"""Text-to-speech with Piper (fast local neural TTS).

Loads a ``.onnx`` voice once and synthesizes an utterance into a single int16
waveform. The voice's own sample rate (22.05 kHz for the default lessac-medium)
is returned so the :class:`~voice.audio.Speaker` can play it back correctly.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np

from core.config import PROJECT_ROOT, TTSConfig

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Wraps a loaded Piper voice; call :meth:`synthesize` to get audio."""

    def __init__(self, config: TTSConfig) -> None:
        from piper import PiperVoice

        if not config.voice_model:
            raise ValueError("tts.voice_model is not set in config.yaml")
        model_path = Path(config.voice_model).expanduser()
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Piper voice not found: {model_path}")
        logger.info("loading Piper voice %s", model_path.name)
        self._voice = PiperVoice.load(str(model_path))
        self._speed = config.speed

    @property
    def sample_rate(self) -> int:
        return int(self._voice.config.sample_rate)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize ``text`` into (int16 mono waveform, sample_rate)."""
        chunks: list[np.ndarray] = []
        for chunk in self._voice.synthesize(text):
            chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
        if not chunks:
            return np.zeros(0, dtype=np.int16), self.sample_rate
        return np.concatenate(chunks), self.sample_rate

    def synthesize_to_wav(self, text: str, path: str | Path) -> Path:
        """Synthesize straight to a ``.wav`` file (used for offline testing)."""
        path = Path(path)
        with wave.open(str(path), "wb") as wav:
            self._voice.synthesize_wav(text, wav)
        return path
