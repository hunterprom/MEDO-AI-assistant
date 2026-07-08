"""Speech-to-text with faster-whisper.

Loads once (model load is the expensive part) and transcribes 16 kHz float32
mono audio. On this CPU-only machine ``device=auto`` resolves to CPU with int8
compute, which is the fast path for faster-whisper.

Robustness layers on top of the raw model:

* RMS-target loudness normalization so a quiet mic still transcribes well.
* Per-segment hallucination filtering using Whisper's own ``no_speech_prob``
  and ``avg_logprob`` scores.
* A junk-transcript blacklist that catches the well-known Whisper
  silence hallucinations ("Thank you.", "Thanks for watching!", ...).
* Optional ``hotwords`` biasing (when the installed faster-whisper supports
  it) built from the command phrases in ``initial_prompt``.
"""

from __future__ import annotations

import logging
import string

import numpy as np

from core.config import STTConfig

logger = logging.getLogger(__name__)

# Segments scoring worse than BOTH bounds are almost certainly hallucinated
# from silence/noise, not real speech (thresholds match the OpenAI reference
# heuristic: high no-speech probability combined with a weak decode).
NO_SPEECH_PROB_MAX = 0.66
AVG_LOGPROB_MIN = -0.85

# Full transcripts (lowercased, punctuation stripped) Whisper is known to
# hallucinate on silence, breaths, or music — the classic YouTube-caption set.
JUNK_TRANSCRIPTS: frozenset[str] = frozenset(
    {
        "",
        "you",
        "bye",
        "bye bye",
        "goodbye",
        "so",
        "the",
        "oh",
        "okay",
        "uh",
        "um",
        "hmm",
        "huh",
        "yeah",
        "thank you",
        "thank you thank you",
        "thank you very much",
        "thanks",
        "thanks for watching",
        "thank you for watching",
        "thanks for listening",
        "thank you for listening",
        "please subscribe",
        "subscribe",
        "like and subscribe",
        "see you next time",
        "see you in the next video",
        "see you later",
        "music",
        "applause",
        "laughter",
        "silence",
        "copyright",
        "transcribed by httpsotterai",
        "subtitles by the amaraorg community",
    }
)


def _canonical(text: str) -> str:
    """Lowercase ``text``, strip punctuation, and collapse whitespace."""
    stripped = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(stripped.split())


def filter_transcript(
    segments_data: list[tuple[str, float, float]],
    *,
    no_speech_prob_max: float = NO_SPEECH_PROB_MAX,
    avg_logprob_min: float = AVG_LOGPROB_MIN,
) -> str:
    """Join Whisper segments into one transcript, dropping hallucinations.

    ``segments_data`` is a list of ``(text, no_speech_prob, avg_logprob)``
    tuples — the metadata faster-whisper reports per segment. A segment is
    dropped only when it looks like non-speech on *both* axes: high
    ``no_speech_prob`` AND low ``avg_logprob`` (either alone is common on
    genuine short utterances). If the joined result is a known junk
    transcript (Whisper's silence hallucinations), an empty string is
    returned. Pure function — no model needed, so it is unit-testable.
    """
    kept: list[str] = []
    for text, no_speech_prob, avg_logprob in segments_data:
        if no_speech_prob > no_speech_prob_max and avg_logprob < avg_logprob_min:
            logger.debug(
                "dropping hallucinated segment %r (no_speech=%.2f logprob=%.2f)",
                text,
                no_speech_prob,
                avg_logprob,
            )
            continue
        text = text.strip()
        if text:
            kept.append(text)
    transcript = " ".join(kept).strip()
    if _canonical(transcript) in JUNK_TRANSCRIPTS:
        if transcript:
            logger.debug("dropping junk transcript %r", transcript)
        return ""
    return transcript


def build_hotwords(initial_prompt: str) -> str:
    """Extract a hotwords string from an ``initial_prompt``-style sentence.

    The configured prompt looks like ``"Commands for a voice assistant: what
    time is it, open chrome, ..."`` — the useful decoder bias is the comma
    separated command phrases after the colon, not the framing text.
    """
    _, _, tail = initial_prompt.partition(":")
    phrases = [p.strip(" .!") for p in (tail or initial_prompt).split(",")]
    return ", ".join(p for p in phrases if p)


def _cuda_runtime_ok() -> bool:
    """True when CUDA's math libraries are actually loadable, not just the driver.

    ``get_cuda_device_count`` succeeds with only the display driver installed,
    but encoding needs cuBLAS (``cublas64_12.dll``) — its absence surfaces as a
    RuntimeError on the *first transcription*, which used to freeze a voice
    session on "PROCESSING". Preflight it so auto-selection is honest.
    """
    import ctypes

    try:
        ctypes.WinDLL("cublas64_12")
        return True
    except OSError:
        logger.info("CUDA present but cublas64_12.dll not loadable — using CPU for STT")
        return False
    except Exception:
        return True  # non-Windows or unexpected: let ctranslate2 decide


def _resolve_device(device: str, compute_type: str) -> tuple[str, str]:
    """Turn ``auto`` into concrete faster-whisper (device, compute_type) values."""
    if device == "auto":
        try:  # prefer CUDA only if a working GPU build is present
            import ctranslate2

            device = (
                "cuda"
                if ctranslate2.get_cuda_device_count() > 0 and _cuda_runtime_ok()
                else "cpu"
            )
        except Exception:
            device = "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


class Transcriber:
    """Wraps a loaded Whisper model; call :meth:`transcribe` per utterance."""

    def __init__(self, config: STTConfig) -> None:
        import inspect

        from faster_whisper import WhisperModel

        self._config = config
        self._language = None if config.language in ("", "auto") else config.language
        device, compute_type = _resolve_device(config.device, config.compute_type)
        logger.info(
            "loading faster-whisper %r on %s/%s", config.model, device, compute_type
        )
        self._device = device
        self._model = WhisperModel(config.model, device=device, compute_type=compute_type)
        # Newer faster-whisper releases accept a `hotwords` decoder bias; pass
        # the command vocabulary through it when available.
        self._hotwords: str | None = None
        try:
            supports_hotwords = (
                "hotwords" in inspect.signature(self._model.transcribe).parameters
            )
        except (TypeError, ValueError):
            supports_hotwords = False
        if supports_hotwords and config.initial_prompt:
            self._hotwords = build_hotwords(config.initial_prompt) or None

    @staticmethod
    def _normalize(
        audio: np.ndarray,
        target_rms: float = 0.06,
        max_gain: float = 10.0,
    ) -> np.ndarray:
        """Boost quiet audio toward ``target_rms`` (leaves loud audio and silence alone).

        RMS-target normalization tracks perceived loudness better than peak
        normalization (one click no longer defeats the boost). The gain is
        capped at ``max_gain`` so the noise floor isn't blown up, never
        attenuates already-loud audio, and is limited so samples stay in
        [-1, 1]. Near-silence is returned untouched.
        """
        if audio.size == 0:
            return audio
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < 1e-4:  # near-silence: nothing worth amplifying
            return audio
        gain = min(target_rms / rms, max_gain)
        if gain <= 1.0:
            return audio  # already at or above target loudness
        peak = float(np.max(np.abs(audio)))
        if peak > 0.0:
            gain = min(gain, 0.99 / peak)  # keep samples clip-free
        return (audio * np.float32(max(gain, 1.0))).astype(np.float32)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a mono 16 kHz float32 waveform to a stripped string.

        Self-heals a broken GPU stack: CUDA can *enumerate* fine at load time
        (driver present) while its math libraries are missing — e.g.
        ``cublas64_12.dll`` — which only surfaces as a RuntimeError on the
        FIRST encode. That used to kill the whole voice loop mid-"PROCESSING";
        now the model is rebuilt on CPU/int8 once and the utterance retried.
        """
        try:
            return self._transcribe(audio)
        except RuntimeError as exc:
            if self._device == "cpu":
                raise
            logger.warning(
                "GPU transcription failed (%s) — rebuilding on CPU/int8", exc
            )
            from faster_whisper import WhisperModel

            self._device = "cpu"
            self._model = WhisperModel(
                self._config.model, device="cpu", compute_type="int8"
            )
            return self._transcribe(audio)

    def _transcribe(self, audio: np.ndarray) -> str:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio = self._normalize(audio)
        c = self._config
        kwargs: dict[str, object] = {}
        if self._hotwords is not None:
            kwargs["hotwords"] = self._hotwords
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=c.beam_size,
            vad_filter=c.vad_filter,
            condition_on_previous_text=c.condition_on_previous_text,
            initial_prompt=c.initial_prompt or None,
            no_speech_threshold=getattr(c, "no_speech_threshold", 0.6),
            **kwargs,
        )
        segments_data = [
            (seg.text, float(seg.no_speech_prob), float(seg.avg_logprob))
            for seg in segments
        ]
        if getattr(c, "filter_hallucinations", True):
            return filter_transcript(segments_data)
        return " ".join(text.strip() for text, _, _ in segments_data).strip()
