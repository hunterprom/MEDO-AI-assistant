"""Swappable text-to-speech engines behind one interface.

Nothing in MEDO calls a TTS engine directly any more — the same rule the LLM
layer already follows for brains (:mod:`llm.client`). A provider takes text
plus the language it is in and returns PCM; which engine that is, and whether
it runs on CPU, GPU or someone else's server, is a configuration question.

Three ship:

* :class:`SherpaProvider` — the workhorse. Piper (VITS) and Kokoro ONNX voices
  under sherpa-onnx, which is Apache-2.0 and runs entirely on CPU. Fifteen of
  MEDO's sixteen languages, no VRAM, measured RTF 0.057.
* :class:`EdgeProvider` — Microsoft's cloud voices. Only reachable for
  languages that have no local voice at all (today: Macedonian), and only
  while ``tts.allow_cloud`` is on.
* :class:`LegacyPiperProvider` — the ``piper-tts`` package MEDO used to call
  directly. Kept so nothing breaks on an install that has it, but it is
  **GPL-3.0-or-later**, which is a licensing problem for a product that may be
  sold. SherpaProvider plays the same voice files under Apache-2.0; when this
  provider is no longer needed, dropping the dependency clears that.

:class:`VoiceBox` chains them: the first provider that can speak the language
wins, and a provider that fails hands on to the next rather than costing the
turn its voice.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from voice import voices
from voice.voices import VoiceSpec

logger = logging.getLogger(__name__)

#: Empty audio, in the shape every provider returns.
SILENCE: tuple[np.ndarray, int] = (np.zeros(0, dtype=np.int16), 22050)

#: Target RMS for a spoken reply, as a fraction of full scale. Voices trained
#: on different corpora arrive at wildly different levels: measured on the same
#: sentence, es_ES-miro-high peaked at 3870 and tr_TR-fahrettin at 32746 — an
#: eight-fold difference, and the loud one within a hair of clipping. Left
#: alone, changing language changes the volume, which reads as MEDO being
#: broken rather than as a property of the voice model.
TARGET_RMS = 0.10
#: Never let normalisation push a peak past this. Speech is peaky; scaling a
#: quiet-but-spiky clip up to the target RMS would clip it.
PEAK_CEILING = 0.95


def normalize(samples: np.ndarray) -> np.ndarray:
    """Bring float32 audio to a consistent speaking level, without clipping.

    RMS-based, not peak-based: peak normalisation just equalises the loudest
    transient, which for speech is usually a plosive and tells you nothing
    about how loud the voice sounds.
    """
    if samples.size == 0:
        return samples
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if rms <= 1e-6:                      # digital silence — nothing to scale
        return samples
    gain = TARGET_RMS / rms
    peak = float(np.abs(samples).max())
    if peak * gain > PEAK_CEILING:
        gain = PEAK_CEILING / peak
    return samples * gain


@runtime_checkable
class TTSProvider(Protocol):
    """One engine that can turn text into speech."""

    name: str

    def supports(self, language: str | None) -> bool:
        """Can this provider voice that language, right now, as configured?"""
        ...

    async def synthesize(self, text: str,
                         language: str | None = None) -> tuple[np.ndarray, int]:
        """``text`` as (int16 mono PCM, sample_rate). Empty audio on failure."""
        ...


class SherpaProvider:
    """Piper + Kokoro ONNX voices via sherpa-onnx. Local, CPU, no VRAM.

    Models load lazily — the first Japanese reply pays for downloading and
    loading the Japanese voice, and a MEDO that is only ever spoken to in
    English never fetches the other fourteen. Loaded models are cached by
    ARCHIVE (not language), so Japanese and Korean share the one Kokoro model
    rather than holding two copies of it.
    """

    name = "sherpa"

    def __init__(self, root: Path, *, max_loaded: int = 3,
                 num_threads: int = 2, offline_only: bool = True,
                 auto_download: bool = True, normalize_loudness: bool = True,
                 use_fallback_voices: bool = False, speed: float = 1.0,
                 overrides: dict | None = None) -> None:
        self._root = Path(root)
        # config.yaml's tts.voices applied over the built-in map, so changing
        # how MEDO sounds is an edit to config, not to code.
        self._table = voices.load_voices(overrides)
        # tts.speed existed in config and was being ignored by this path.
        self._speed = float(speed) if speed and speed > 0 else 1.0
        self._max_loaded = max(1, int(max_loaded))
        self._threads = max(1, int(num_threads))
        self._offline_only = offline_only
        self._auto_download = auto_download
        self._normalize = normalize_loudness
        # Off by default. On, this provider will speak a language using
        # another language's voice (Macedonian in the Serbian voice). That is
        # a deliberate choice a person makes, never a default — see
        # voices.spec_for for the bug this flag exists to prevent.
        self._use_fallback = use_fallback_voices
        self._loaded: OrderedDict[str, object] = OrderedDict()
        self._failed: set[str] = set()

    # -- capability ---------------------------------------------------------

    def spec_for(self, language: str | None) -> VoiceSpec | None:
        spec = voices.spec_for(language, offline_only=self._offline_only,
                               allow_fallback=self._use_fallback,
                               table=self._table)
        if spec is None or spec.engine == "edge":
            return None
        return spec

    def supports(self, language: str | None) -> bool:
        spec = self.spec_for(language)
        if spec is None or spec.archive in self._failed:
            return False
        # An uninstalled voice is only "supported" if we're allowed to fetch
        # it. Claiming otherwise would make VoiceBox skip the cloud fallback
        # and then produce silence.
        return self._auto_download or voices.is_installed(self._root, spec)

    # -- synthesis ----------------------------------------------------------

    async def synthesize(self, text: str,
                         language: str | None = None) -> tuple[np.ndarray, int]:
        spec = self.spec_for(language)
        if spec is None or not (text or "").strip():
            return SILENCE
        return await asyncio.to_thread(self._speak, text, spec)

    def _speak(self, text: str, spec: VoiceSpec) -> tuple[np.ndarray, int]:
        engine = self._engine(spec)
        if engine is None:
            return SILENCE
        try:
            audio = engine.generate(text, sid=spec.speaker, speed=self._speed)
        except Exception:
            logger.warning("sherpa synthesis failed for %s", spec.archive,
                           exc_info=True)
            return SILENCE
        samples = np.asarray(audio.samples, dtype=np.float32)
        if samples.size == 0:
            return SILENCE
        if self._normalize:
            samples = normalize(samples)
        # sherpa hands back float32 in [-1, 1]; the audio stack wants int16.
        # Clip first: a sample above 1.0 wraps to a loud crack when cast.
        pcm = np.clip(samples, -1.0, 1.0) * 32767.0
        return pcm.astype(np.int16), int(audio.sample_rate)

    def _engine(self, spec: VoiceSpec):
        """The loaded sherpa model for this voice, loading/fetching it once."""
        cached = self._loaded.get(spec.archive)
        if cached is not None:
            self._loaded.move_to_end(spec.archive)
            return cached
        if spec.archive in self._failed:
            return None
        model = voices.ensure(self._root, spec) if self._auto_download else (
            voices.model_path(self._root, spec)
            if voices.is_installed(self._root, spec) else None)
        if model is None or not Path(model).exists():
            self._failed.add(spec.archive)
            return None
        try:
            engine = self._build(spec, Path(model))
        except Exception:
            logger.warning("couldn't load voice %s", spec.archive, exc_info=True)
            self._failed.add(spec.archive)
            return None
        self._loaded[spec.archive] = engine
        while len(self._loaded) > self._max_loaded:
            evicted, _ = self._loaded.popitem(last=False)
            logger.debug("evicted voice %s from the model cache", evicted)
        return engine

    def _build(self, spec: VoiceSpec, model: Path):
        import sherpa_onnx

        folder = model.parent
        espeak = self._root / voices.SHARED_ESPEAK
        tokens = str(folder / "tokens.txt")
        if spec.engine == "sherpa-kokoro":
            model_config = sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(model),
                    voices=str(folder / "voices.bin"),
                    tokens=tokens,
                    data_dir=str(espeak),
                    dict_dir=str(folder / "dict"),
                    lexicon=",".join(
                        str(folder / n) for n in ("lexicon-us-en.txt",
                                                  "lexicon-zh.txt")
                        if (folder / n).exists()),
                    # Without this Kokoro phonemizes every language as en-us.
                    # It is not cosmetic: Japanese came out at 12.9 s for a
                    # sentence that takes 4.2 s, because it was being read as
                    # if the characters were English.
                    lang=spec.lang,
                ),
                num_threads=self._threads, provider="cpu")
        else:
            model_config = sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model), tokens=tokens, data_dir=str(espeak)),
                num_threads=self._threads, provider="cpu")
        logger.info("loading voice %s (%s)", spec.archive, spec.quality)
        return sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(model=model_config))


class EdgeProvider:
    """Microsoft's cloud neural voices — the exception, not the rule.

    Only offered for a language with no local voice at all, and only while
    ``allow_cloud`` is set. Everything else it could speak is already covered
    locally by :class:`SherpaProvider`, and routing a language to the cloud
    that doesn't need to go there is exactly the drift this class exists to
    prevent.
    """

    name = "edge"

    def __init__(self, edge, *, allow_cloud: bool = True) -> None:
        self._edge = edge                 # voice.tts.EdgeTTS, or None
        self._allow = allow_cloud

    def supports(self, language: str | None) -> bool:
        if self._edge is None or not self._allow:
            return False
        spec = voices.spec_for(language)
        return spec is not None and spec.engine == "edge"

    async def synthesize(self, text: str,
                         language: str | None = None) -> tuple[np.ndarray, int]:
        if self._edge is None:
            return SILENCE
        try:
            return await self._edge.synthesize(text, language)
        except Exception:
            logger.warning("edge-tts failed for %r", language, exc_info=True)
            return SILENCE


class LegacyPiperProvider:
    """The ``piper-tts`` package, called directly. English only, GPL-3.0.

    Last in the chain and English-only on purpose: this is the engine that
    produced the "reads Japanese out character by character" bug when it was
    handed text it had no voice for.
    """

    name = "piper-legacy"

    def __init__(self, tts) -> None:
        self._tts = tts                   # voice.tts.TextToSpeech, or None

    def supports(self, language: str | None) -> bool:
        code = (language or "en").strip().lower()[:2]
        return self._tts is not None and code == "en"

    async def synthesize(self, text: str,
                         language: str | None = None) -> tuple[np.ndarray, int]:
        if self._tts is None:
            return SILENCE
        try:
            return await asyncio.to_thread(self._tts.synthesize, text)
        except Exception:
            logger.warning("piper synthesis failed", exc_info=True)
            return SILENCE


class VoiceBox:
    """The chain: the first provider that can speak the language wins.

    A provider that returns no audio is treated as a failure and the next one
    is tried, so a voice that won't download costs that language its preferred
    engine rather than costing the user the reply. When nothing can speak it,
    the caller gets empty audio and must SHOW the reply instead — never voice
    it in the wrong language.
    """

    def __init__(self, providers: list[TTSProvider]) -> None:
        self._providers = [p for p in providers if p is not None]

    @property
    def providers(self) -> list[TTSProvider]:
        return list(self._providers)

    def provider_for(self, language: str | None) -> TTSProvider | None:
        for provider in self._providers:
            if provider.supports(language):
                return provider
        return None

    def can_speak(self, language: str | None) -> bool:
        return self.provider_for(language) is not None

    async def synthesize(self, text: str,
                         language: str | None = None) -> tuple[np.ndarray, int]:
        for provider in self._providers:
            if not provider.supports(language):
                continue
            audio, rate = await provider.synthesize(text, language)
            if getattr(audio, "size", 0):
                return audio, rate
            logger.info("provider %s gave no audio for %r — trying the next",
                        provider.name, language)
        return SILENCE
