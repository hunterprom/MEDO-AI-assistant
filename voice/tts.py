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


#: End of a sentence. Three families, because "terminal punctuation then a
#: space" is an alphabetic-script assumption and MEDO speaks sixteen languages:
#:
#: * Latin/Cyrillic/Greek — ``.!?`` plus an optional closing quote, and the
#:   whitespace AFTER it is what proves the sentence ended. Without that
#:   lookahead "Dr. Smith" and "3.5" split mid-phrase.
#: * CJK — ``。！？`` are full-width and are themselves the boundary; Chinese
#:   and Japanese put no space after them. Requiring one meant zh and ja NEVER
#:   drained a sentence: measured, both waited for the entire reply before a
#:   single word was spoken.
#: * Indic — the danda ``।`` ends a Devanagari sentence. Hindi had the same
#:   total failure for the same reason.
_SENTENCE_END = re.compile(
    r"[.!?…]+[\"')\]]?(?=\s)"          # alphabetic scripts: space follows
    r"|[。．！？]+[」』）】〉]?"           # CJK: the mark itself ends it
    r"|[।॥]+(?=\s|$)"                  # Devanagari danda / double danda
)

#: Characters that carry far more speech than a Latin letter does. Measured on
#: the same sentence: 16 Chinese characters and 44 English ones both produced
#: ~2.8 s of audio, so one CJK/Devanagari character is worth about three.
#: Without this weighting a ``min_len`` tuned for English holds three Japanese
#: sentences back waiting for a chunk "long enough".
_DENSE = re.compile(r"[぀-ヿ㐀-鿿가-힯ऀ-ॿ]")


def speech_weight(text: str) -> int:
    """Roughly how much SPEECH a string is worth, in Latin-character units."""
    dense = len(_DENSE.findall(text or ""))
    return len(text or "") + dense * 2


def drain_sentences(buffer: str, min_len: int = 24) -> tuple[list[str], str]:
    """Pull complete sentences off a streaming text buffer.

    Returns ``(sentences, remainder)``. Chunks worth less speech than
    ``min_len`` are merged with the following one ("Dr." or "1." must not be
    spoken alone). The remainder holds the trailing incomplete sentence —
    flush it yourself when the stream ends.

    Length is measured in :func:`speech_weight`, not characters, so the
    threshold means the same amount of talking in every script.
    """
    sentences: list[str] = []
    while True:
        emitted = False
        for m in _SENTENCE_END.finditer(buffer):
            candidate = buffer[: m.end()].strip()
            if speech_weight(candidate) >= min_len:
                sentences.append(candidate)
                buffer = buffer[m.end():].lstrip()
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
        """The voice to speak a reply in, given the detected language.

        Falls back to the construction voice (the primary active language's, set
        by the loop) when a language has no registered voice — warning once so a
        silent wrong-voice is visible without spamming the log every utterance.
        """
        from core import languages

        voice = languages.voice_for(language, "")
        if not voice:
            if language and not getattr(self, "_warned_fallback", False):
                logger.warning("no edge-tts voice for %r — using fallback voice %r",
                               language, self._voice)
                self._warned_fallback = True
            return self._voice
        return voice

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


#: One neutral sentence per language, for --demo. Same content everywhere so
#: voices are compared on delivery, not on what they happen to be saying.
DEMO_LINES = {
    "en": "It's twenty-two degrees and clear in Skopje right now. The wind is "
          "light from the north-west, and it should stay dry until the evening.",
    "mk": "Дваесет и два степени е и ведро во Скопје. Ветерот е слаб од "
          "северозапад, и до вечерта ќе биде суво.",
    "de": "Es sind zweiundzwanzig Grad und klar in Skopje. Der Wind weht "
          "schwach aus Nordwesten, und bis zum Abend bleibt es trocken.",
    "fr": "Il fait vingt-deux degrés et le ciel est dégagé à Skopje. Le vent "
          "est faible et il devrait rester sec jusqu'au soir.",
    "es": "Hace veintidós grados y está despejado en Skopie. El viento es "
          "flojo y debería seguir seco hasta la noche.",
    "it": "Ci sono ventidue gradi e il cielo è sereno a Skopje. Il vento è "
          "debole e dovrebbe restare asciutto fino a sera.",
    "pt": "Estão vinte e dois graus e o céu está limpo em Skopje. O vento é "
          "fraco e deve continuar seco até à noite.",
    "nl": "Het is tweeëntwintig graden en helder in Skopje. De wind is zwak "
          "en het blijft droog tot de avond.",
    "pl": "Jest dwadzieścia dwa stopnie i bezchmurnie w Skopje. Wiatr jest "
          "słaby i do wieczora powinno być sucho.",
    "ru": "Сейчас двадцать два градуса, в Скопье ясно. Ветер слабый, до "
          "вечера будет сухо.",
    "tr": "Üsküp'te hava yirmi iki derece ve açık. Rüzgâr hafif, akşama kadar "
          "kuru kalması bekleniyor.",
    "el": "Είναι είκοσι δύο βαθμοί και αίθριος καιρός στα Σκόπια. Ο άνεμος "
          "είναι ασθενής και θα παραμείνει στεγνό μέχρι το βράδυ.",
    "zh": "斯科普里现在二十二度，天气晴朗。风力较弱，傍晚之前都会保持干燥。",
    "ja": "スコピエは今、気温二十二度で晴れています。風は弱く、夕方までは乾燥した天気が続きます。",
    "ko": "스코페는 지금 이십이 도이고 맑습니다. 바람은 약하고 저녁까지는 건조하겠습니다.",
    "hi": "स्कोप्ये में इस समय बाईस डिग्री है और मौसम साफ़ है। हवा हल्की है और शाम तक मौसम सूखा रहेगा।",
}


def _demo(argv: list[str]) -> int:
    """Speak (or write) the demo line for a language, so voices can be A/B'd.

        python -m voice.tts --demo en
        python -m voice.tts --demo en --voice kokoro:af_heart
        python -m voice.tts --demo all --write voice-samples
    """
    import argparse
    import asyncio
    import wave as _wave
    from pathlib import Path as _Path

    parser = argparse.ArgumentParser(prog="voice.tts")
    parser.add_argument("--demo", metavar="LANG", required=True,
                        help="language code, or 'all'")
    parser.add_argument("--voice", default="",
                        help="override, e.g. kokoro:bf_emma or piper:de_DE-thorsten-high")
    parser.add_argument("--text", default="", help="say this instead")
    parser.add_argument("--write", metavar="DIR", default="",
                        help="write .wav files here instead of playing")
    args = parser.parse_args(argv)

    from core.config import PROJECT_ROOT, load_settings
    from voice.providers import SherpaProvider
    from voice import voices as _voices

    settings = load_settings()
    codes = list(DEMO_LINES) if args.demo == "all" else [args.demo]
    overrides = dict(settings.tts.voices or {})
    if args.voice:
        engine, _, name = args.voice.partition(":")
        for code in codes:
            overrides[code] = {"engine": engine, "voice": name}
    provider = SherpaProvider(
        PROJECT_ROOT / settings.tts.voices_dir,
        max_loaded=settings.tts.max_loaded_voices,
        num_threads=settings.tts.num_threads,
        speed=settings.tts.speed, overrides=overrides)

    async def run() -> int:
        table = _voices.load_voices(overrides)
        failures = 0
        for code in codes:
            text = args.text or DEMO_LINES.get(code, "")
            spec = table.get(code)
            if not text or spec is None:
                print(f"  {code:3}  no demo line or no voice")
                failures += 1
                continue
            if spec.engine == "edge":
                print(f"  {code:3}  {spec.archive} (cloud — not played here)")
                continue
            audio, rate = await provider.synthesize(text, code)
            if not audio.size:
                print(f"  {code:3}  FAILED  {spec.archive}")
                failures += 1
                continue
            label = (f"{spec.archive} sid={spec.speaker}"
                     if spec.engine == "sherpa-kokoro" else spec.archive)
            print(f"  {code:3}  {audio.size / rate:4.1f}s  [{spec.quality}]  {label}")
            if args.write:
                out = _Path(args.write)
                out.mkdir(parents=True, exist_ok=True)
                with _wave.open(str(out / f"{code}.wav"), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(rate)
                    w.writeframes(audio.tobytes())
            else:
                from voice.audio import AudioIO

                AudioIO(settings.audio).play(audio, rate)
        return failures

    return asyncio.run(run())


if __name__ == "__main__":       # pragma: no cover - operator entry point
    import sys

    raise SystemExit(_demo(sys.argv[1:]))
