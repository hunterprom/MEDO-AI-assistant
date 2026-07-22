"""Text-to-speech: Piper (fast local neural TTS) + edge-tts for Macedonian.

Piper loads a ``.onnx`` voice once and synthesizes an utterance into a single
int16 waveform; its own sample rate (22.05 kHz for lessac-medium) is returned
so playback is pitched correctly. Piper has no Macedonian voice, so Cyrillic
replies are handed to :class:`EdgeTTS` — Microsoft's free neural voices
(``mk-MK-MarijaNeural``) — with Piper as the offline fallback. The mp3 stream
edge-tts returns is decoded to PCM with the local ffmpeg.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import wave
from pathlib import Path

import numpy as np

from core.config import PROJECT_ROOT, TTSConfig

logger = logging.getLogger(__name__)

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def contains_cyrillic(text: str) -> bool:
    """True when the text has any Cyrillic — the 'speak this in Macedonian' cue."""
    return bool(_CYRILLIC.search(text or ""))


#: End of a sentence: terminal punctuation, optionally followed by a closing
#: quote/bracket, then whitespace.
_SENTENCE_END = re.compile(r"([.!?…]+[\"')\]]?)\s")


def drain_sentences(buffer: str, min_len: int = 24) -> tuple[list[str], str]:
    """Pull complete sentences off a streaming text buffer.

    Returns ``(sentences, remainder)``. Sentences shorter than ``min_len`` are
    merged with the following one ("Dr." or "1." must not be spoken alone), so
    the TTS gets natural chunks. The remainder holds the trailing incomplete
    sentence — flush it yourself when the stream ends.
    """
    sentences: list[str] = []
    while True:
        emitted = False
        for m in _SENTENCE_END.finditer(buffer):
            candidate = buffer[: m.end(1)].strip()
            if len(candidate) >= min_len:
                sentences.append(candidate)
                buffer = buffer[m.end(1):].lstrip()
                emitted = True
                break
        if not emitted:
            return sentences, buffer


def find_ffmpeg() -> str | None:
    """Locate ffmpeg (PATH, then the usual per-OS install spots). None if absent.

    PATH alone isn't enough: an app launched from Finder/Explorer does NOT
    inherit the shell's PATH, so a Homebrew ffmpeg in /opt/homebrew/bin is
    invisible and Macedonian replies silently fall back to the English voice.
    """
    path = shutil.which("ffmpeg")
    if path:
        return path
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if base.exists():
        hits = sorted(base.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe"))
        if hits:
            return str(hits[-1])
    for candidate in ("/opt/homebrew/bin/ffmpeg",   # macOS (Apple Silicon brew)
                      "/usr/local/bin/ffmpeg",      # macOS (Intel brew) / Linux
                      "/usr/bin/ffmpeg"):           # Linux distro package
        if Path(candidate).exists():
            return candidate
    return None


class EdgeTTS:
    """Microsoft edge-tts neural voices — one per supported language.

    Free, no key; needs internet (the voice runs in Microsoft's cloud) and a
    local ffmpeg to decode the mp3 stream. Callers should catch exceptions and
    fall back to Piper — offline operation must never depend on this class.

    The voice is chosen per utterance from the language Whisper detected, not
    fixed at construction: one instance serves every language MEDO speaks.
    """

    SAMPLE_RATE = 24000

    def __init__(self, voice: str) -> None:
        import edge_tts  # noqa: F401  (fail fast if the package is missing)

        self._voice = voice          # fallback when no language is given
        self._ffmpeg = find_ffmpeg()
        if not self._ffmpeg:
            raise RuntimeError("ffmpeg not found — needed to decode edge-tts audio")
        logger.info("edge-tts ready (voice %s)", voice)

    def voice_for(self, language: str | None) -> str:
        """The voice to speak a reply in, given the detected language."""
        from core import languages

        return languages.voice_for(language, self._voice)

    async def synthesize(self, text: str,
                         language: str | None = None) -> tuple[np.ndarray, int]:
        """Synthesize ``text`` into (int16 mono waveform, sample_rate).

        ``language`` is the ISO code Whisper reported for the user's utterance;
        the reply is spoken by that language's voice so MEDO answers in the
        language it was addressed in.
        """
        import asyncio

        import edge_tts

        voice = self.voice_for(language)
        mp3 = bytearray()
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk["type"] == "audio":
                mp3.extend(chunk["data"])
        if not mp3:
            return np.zeros(0, dtype=np.int16), self.SAMPLE_RATE
        proc = await asyncio.create_subprocess_exec(
            self._ffmpeg, "-v", "error", "-i", "pipe:0",
            "-f", "s16le", "-ar", str(self.SAMPLE_RATE), "-ac", "1", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        pcm, _ = await proc.communicate(bytes(mp3))
        return np.frombuffer(pcm, dtype=np.int16), self.SAMPLE_RATE


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
