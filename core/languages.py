"""The languages MEDO speaks — one table, used by STT, TTS, the LLM and the HUD.

MEDO was bilingual by construction: Whisper was clamped to en/mk and the voice
was chosen by a single "is this Cyrillic?" test. Neither scales — script is not
language (Russian and Macedonian share an alphabet; Japanese and Korean don't
share one with anything here), so the routing key has to be the language code
Whisper actually detected.

Every voice name below was read from ``edge_tts.list_voices()`` rather than
guessed, and pinned to a primary locale on purpose: asking for "German" and
getting Austrian, or "Chinese" and getting Cantonese, is the kind of detail
that makes an assistant feel careless.

Two engines sit behind this:

* **Piper** — local, offline, and English-only in this install. Still the
  fallback whenever the network is down.
* **edge-tts** — Microsoft's free neural voices, one per language here. Needs
  internet. That trade-off is the price of speaking sixteen languages without
  shipping sixteen voice models.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    """One spoken language: how to hear it, how to say it, what to call it."""

    code: str            # ISO-639-1, the code faster-whisper reports
    english: str         # "German"
    native: str          # "Deutsch" — what the HUD picker shows
    voice: str           # edge-tts ShortName, primary locale
    #: Right-to-left scripts need the HUD to flip direction. None here yet, but
    #: the field exists so adding Arabic or Hebrew is data, not a refactor.
    rtl: bool = False


#: The curated set. Deliberately NOT "everything Whisper supports": the
#: language clamp in voice/stt.py is what stops Macedonian being decoded as
#: Bulgarian, and every language added widens the field it can be confused
#: with. These sixteen are far apart enough to stay reliable.
LANGUAGES: tuple[Language, ...] = (
    Language("en", "English", "English", "en-US-AvaNeural"),
    Language("mk", "Macedonian", "Македонски", "mk-MK-MarijaNeural"),
    # --- European ---
    Language("de", "German", "Deutsch", "de-DE-SeraphinaMultilingualNeural"),
    Language("fr", "French", "Français", "fr-FR-VivienneMultilingualNeural"),
    Language("es", "Spanish", "Español", "es-ES-XimenaNeural"),
    Language("it", "Italian", "Italiano", "it-IT-ElsaNeural"),
    Language("pt", "Portuguese", "Português", "pt-PT-RaquelNeural"),
    Language("nl", "Dutch", "Nederlands", "nl-NL-ColetteNeural"),
    Language("pl", "Polish", "Polski", "pl-PL-ZofiaNeural"),
    Language("ru", "Russian", "Русский", "ru-RU-SvetlanaNeural"),
    Language("tr", "Turkish", "Türkçe", "tr-TR-EmelNeural"),
    Language("el", "Greek", "Ελληνικά", "el-GR-AthinaNeural"),
    # --- Asian ---
    Language("zh", "Chinese", "中文", "zh-CN-XiaoxiaoNeural"),
    Language("ja", "Japanese", "日本語", "ja-JP-NanamiNeural"),
    Language("ko", "Korean", "한국어", "ko-KR-SunHiNeural"),
    Language("hi", "Hindi", "हिन्दी", "hi-IN-SwaraNeural"),
)

_BY_CODE = {lang.code: lang for lang in LANGUAGES}


def get(code: str | None) -> Language | None:
    """Look up a language by the code Whisper reported. None if unsupported."""
    if not code:
        return None
    return _BY_CODE.get(str(code).strip().lower()[:2])


def voice_for(code: str | None, default: str = "") -> str:
    """The edge-tts voice for a detected language, or ``default``."""
    lang = get(code)
    return lang.voice if lang is not None else default


def english_name(code: str | None, default: str = "English") -> str:
    """"ja" -> "Japanese". Used to tell the LLM which language to answer in."""
    lang = get(code)
    return lang.english if lang is not None else default


def detect_script(text: str) -> str | None:
    """Guess the language of ``text`` from its script alone. None if Latin.

    The safety net for replies with no detected language behind them — typed
    HUD input, an announcement, a skill's own words. Without it a Japanese
    reply reaches the English voice, which reads it out character by character.

    Script is a weak signal and this only claims what a script can actually
    settle: kana means Japanese, Hangul means Korean, unaccompanied Han means
    Chinese. Latin returns None rather than guessing between a dozen languages
    that share it — the caller's default is better than a coin flip.
    """
    if not text:
        return None
    for ch in text:
        code = ord(ch)
        if 0x3040 <= code <= 0x30FF:                 # hiragana / katakana
            return "ja"
        if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
            return "ko"                              # hangul
        if 0x0900 <= code <= 0x097F:
            return "hi"                              # devanagari
        if 0x0370 <= code <= 0x03FF:
            return "el"                              # greek
    # Han without kana: Chinese. Checked second so Japanese kanji mixed with
    # kana is caught above as Japanese, which is what it is.
    if any(0x4E00 <= ord(ch) <= 0x9FFF for ch in text):
        return "zh"
    if any(0x0400 <= ord(ch) <= 0x04FF for ch in text):
        return "mk"          # Cyrillic: MEDO's own is Macedonian
    return None


def codes() -> list[str]:
    """Every supported code — the STT clamp list."""
    return [lang.code for lang in LANGUAGES]


def enabled(only: list[str] | None = None) -> tuple[Language, ...]:
    """The languages switched on in config; all of them when unset."""
    if not only:
        return LANGUAGES
    wanted = {c.strip().lower() for c in only if c.strip()}
    # English is never dropped: it is the fallback voice and the language the
    # system prompt is written in, so losing it would break the default path.
    wanted.add("en")
    return tuple(lang for lang in LANGUAGES if lang.code in wanted)
