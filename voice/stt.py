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
import unicodedata

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
        # Non-English silence hallucinations — with language auto-detect on,
        # Whisper invents these on silence too (a "Gracias"/"De nada" exchange
        # from nothing is the classic case). Belt-and-suspenders alongside
        # narrowing stt.allowed_languages to what you actually speak.
        "gracias",              # es
        "muchas gracias",
        "gracias por ver el video",
        "gracias por ver",
        "de nada",
        "adios",
        "hasta luego",
        "merci",                # fr
        "merci beaucoup",
        "merci davoir regarde",
        "danke",                # de
        "vielen dank",
        "danke schon",
        "grazie",               # it
        "grazie per lattenzione",
        "obrigado",             # pt
        "obrigada",
        "gracias por su atencion",
        "фала",                 # mk
        "фала многу",
        "благодарам",
        "довидување",
        "продолжуваме",
    }
)


def _canonical(text: str) -> str:
    """Lowercase ``text``, strip punctuation, and collapse whitespace.

    Strips ALL Unicode punctuation (category ``P*``), not just ASCII — so
    Spanish '¡Gracias!' and other non-English phantoms normalize to their bare
    form and match the junk set.
    """
    lowered = text.lower()
    stripped = "".join(
        c for c in lowered if not unicodedata.category(c).startswith("P")
    )
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


def pick_forced_language(detected: str | None, allowed: list[str]) -> str | None:
    """The language to force a re-transcription with, or None to keep the decode.

    Whisper's per-utterance language ID regularly mistakes spoken Macedonian
    for a neighboring language (Bulgarian/Serbian/Slovenian — or Russian on
    short clips); the decode then uses the wrong tokenizer context and comes
    out garbled, which reads as "it doesn't understand complex sentences".
    For a bilingual assistant the fix is a clamp: when the detected language
    falls outside ``allowed``, force the first non-English allowed language
    (English detections are reliable; the misdetections are Slavic-on-Slavic).
    Empty ``allowed`` disables the clamp. Pure — unit-testable without audio.
    """
    if not allowed or detected in allowed:
        return None
    return next((lang for lang in allowed if lang != "en"), allowed[0])


def build_hotwords(initial_prompt: str) -> str:
    """Extract a hotwords string from an ``initial_prompt``-style sentence.

    The configured prompt looks like ``"Commands for a voice assistant: what
    time is it, open chrome, ..."`` — the useful decoder bias is the comma
    separated command phrases after the colon, not the framing text.
    """
    _, _, tail = initial_prompt.partition(":")
    phrases = [p.strip(" .!") for p in (tail or initial_prompt).split(",")]
    return ", ".join(p for p in phrases if p)


def _add_pip_cuda_dll_dirs() -> None:
    """Make pip-installed NVIDIA runtime wheels loadable on Windows.

    ``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12`` ship the exact DLLs
    ctranslate2 needs (cuBLAS 12, cuDNN 9) inside site-packages — but Windows
    doesn't look there. Register their bin dirs with the DLL loader AND prepend
    them to PATH (ctranslate2 resolves through PATH), so a plain ``pip install``
    is enough to turn GPU STT on. No-op when the wheels aren't installed.
    """
    import os
    import site
    from pathlib import Path

    candidates = []
    try:
        candidates += site.getsitepackages()
    except Exception:
        pass
    for base in candidates:
        for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
            p = Path(base) / sub
            if p.is_dir():
                try:
                    os.add_dll_directory(str(p))
                    os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
                except Exception:  # best effort — preflight below stays honest
                    pass


def _cuda_runtime_ok() -> bool:
    """True when CUDA's math libraries are actually loadable, not just the driver.

    ``get_cuda_device_count`` succeeds with only the display driver installed,
    but encoding needs cuBLAS (``cublas64_12.dll``) and the conv layers need
    cuDNN 9 — their absence surfaces as a RuntimeError on the *first
    transcription*, which used to freeze a voice session on "PROCESSING".
    Preflight both so auto-selection is honest. Pip-wheel DLL dirs are
    registered first, so installing the nvidia wheels is all it takes.
    """
    import ctypes

    _add_pip_cuda_dll_dirs()
    try:
        ctypes.WinDLL("cublas64_12")
    except OSError:
        logger.info("CUDA present but cublas64_12.dll not loadable — using CPU for STT "
                    "(pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 to enable)")
        return False
    except Exception:
        return True  # non-Windows or unexpected: let ctranslate2 decide
    try:
        ctypes.WinDLL("cudnn64_9")
    except OSError:
        logger.info("cuBLAS found but cuDNN 9 (cudnn64_9.dll) is not loadable — "
                    "using CPU for STT (pip install nvidia-cudnn-cu12 to enable)")
        return False
    except Exception:
        return True
    return True


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
        # Bilingual clamp for auto-detect (see pick_forced_language).
        allowed = [
            lang.strip().lower()
            for lang in getattr(config, "allowed_languages", []) or []
            if lang.strip()
        ]
        if not allowed:
            # Fall back to the supported set rather than to "anything Whisper
            # knows": an unclamped decode is what turned spoken Macedonian into
            # garbled Bulgarian, and the same trap waits for every language
            # here with a close neighbour.
            from core import languages

            allowed = languages.codes()
        self._allowed = allowed
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
        return self.transcribe_with_language(audio)[0]

    def transcribe_with_language(self, audio: np.ndarray) -> tuple[str, str | None]:
        """Like :meth:`transcribe` but also returns the detected language code.

        Used by interpreter mode to know which way to translate. The same
        GPU-stack self-heal applies.
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

    def _decode(self, audio: np.ndarray, language: str | None):
        """One faster-whisper pass; returns (segments_data, detected_language)."""
        c = self._config
        kwargs: dict[str, object] = {}
        if self._hotwords is not None:
            kwargs["hotwords"] = self._hotwords
        segments, info = self._model.transcribe(
            audio,
            language=language,
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
        return segments_data, getattr(info, "language", None)

    def _transcribe(self, audio: np.ndarray) -> tuple[str, str | None]:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio = self._normalize(audio)
        segments_data, detected = self._decode(audio, self._language)
        language = self._language or detected
        # Auto-detect landed outside the allowed set (e.g. Macedonian heard as
        # Bulgarian): the decode used the wrong tokenizer context. Redo it with
        # the language forced — one extra pass, only on misdetection.
        if self._language is None:
            forced = pick_forced_language(detected, self._allowed)
            if forced is not None:
                logger.info(
                    "detected language %r not in allowed %s — re-transcribing as %r",
                    detected, self._allowed, forced,
                )
                segments_data, _ = self._decode(audio, forced)
                language = forced
        if getattr(self._config, "filter_hallucinations", True):
            return filter_transcript(segments_data), language
        text = " ".join(t.strip() for t, _, _ in segments_data).strip()
        return text, language
