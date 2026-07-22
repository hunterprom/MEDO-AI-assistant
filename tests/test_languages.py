"""The multilingual layer: the language table, voice routing, and the HUD strings.

MEDO was bilingual by construction — Whisper clamped to en/mk, and the voice
chosen by a single "is this Cyrillic?" test. Script is not language (Russian
and Macedonian share an alphabet), so the routing key had to become the code
Whisper detected. These tests pin that, and pin the voice names to real
edge-tts locales rather than plausible-looking ones.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from core import languages
from core.config import load_settings

HUD = Path("ui/web/index.html")


# --- the table -----------------------------------------------------------------


def test_every_language_is_complete():
    for lang in languages.LANGUAGES:
        assert len(lang.code) == 2, lang.code
        assert lang.english and lang.native and lang.voice
        # The voice must be a real edge-tts ShortName: locale + name + Neural.
        assert re.match(r"^[a-z]{2}-[A-Z]{2}-\w+Neural$", lang.voice), lang.voice


def test_codes_are_unique():
    codes = languages.codes()
    assert len(codes) == len(set(codes))


def test_the_curated_set_covers_european_and_asian():
    codes = set(languages.codes())
    assert {"de", "fr", "es", "it", "pt", "nl", "pl", "ru", "tr", "el"} <= codes
    assert {"zh", "ja", "ko", "hi"} <= codes
    assert {"en", "mk"} <= codes, "the original pair must survive"


def test_voices_are_pinned_to_primary_locales():
    """Asking for German and getting Austrian is the kind of sloppiness that
    makes an assistant feel careless."""
    expected = {"de": "de-DE", "fr": "fr-FR", "es": "es-ES", "zh": "zh-CN",
                "pt": "pt-PT", "en": "en-US", "mk": "mk-MK"}
    for code, locale in expected.items():
        assert languages.get(code).voice.startswith(locale), code


# --- lookups -------------------------------------------------------------------


def test_get_is_forgiving_about_case_and_region():
    assert languages.get("JA").code == "ja"
    assert languages.get("pt-BR").code == "pt"          # region suffix ignored
    assert languages.get("  de  ").code == "de"


def test_unknown_and_empty_codes_return_nothing():
    for probe in ("xx", "", None, "klingon"):
        assert languages.get(probe) is None


def test_voice_for_falls_back_rather_than_raising():
    assert languages.voice_for("ja") == "ja-JP-NanamiNeural"
    assert languages.voice_for("xx", "fallback") == "fallback"
    assert languages.voice_for(None, "fallback") == "fallback"


def test_english_name_is_what_the_llm_is_told():
    assert languages.english_name("zh") == "Chinese"
    assert languages.english_name("xx", "") == ""


def test_enabled_filters_but_never_drops_english():
    subset = languages.enabled(["ja", "de"])
    codes = {lang.code for lang in subset}
    assert codes == {"en", "ja", "de"}, "English is the fallback voice"
    assert languages.enabled([]) == languages.LANGUAGES
    assert languages.enabled(None) == languages.LANGUAGES


# --- wiring --------------------------------------------------------------------


def test_the_stt_clamp_ships_the_whole_set():
    """An unclamped decode is what garbled Macedonian in the first place."""
    allowed = set(load_settings().stt.allowed_languages)
    assert set(languages.codes()) <= allowed


def test_edge_tts_picks_the_voice_per_utterance():
    """One EdgeTTS instance serves every language; the voice is per utterance."""
    from voice.tts import EdgeTTS

    # Constructed without __init__ so no network or ffmpeg is needed; only the
    # fallback voice matters to voice_for().
    engine = object.__new__(EdgeTTS)
    engine._voice = "mk-MK-MarijaNeural"
    assert engine.voice_for("ja") == "ja-JP-NanamiNeural"
    assert engine.voice_for("de") == "de-DE-SeraphinaMultilingualNeural"
    # An unsupported language keeps the configured fallback rather than failing.
    assert engine.voice_for("xx") == "mk-MK-MarijaNeural"


# --- the HUD strings -----------------------------------------------------------


def _i18n_table() -> dict:
    """Pull the I18N object out of the page and parse it as JSON."""
    html = HUD.read_text(encoding="utf-8")
    start = html.index("const I18N = {") + len("const I18N = ")
    depth, end = 0, start
    for i, ch in enumerate(html[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0 and ch == "}":
            end = i + 1
            break
    blob = html[start:end]
    blob = re.sub(r"(\w+):", r'"\1":', blob)        # bare keys -> quoted
    blob = blob.replace("'", '"').replace(",\n  }", "\n  }")
    blob = re.sub(r",(\s*[}\]])", r"\1", blob)      # trailing commas
    return json.loads(blob)


def test_the_hud_offers_every_supported_language():
    html = HUD.read_text(encoding="utf-8")
    for lang in languages.LANGUAGES:
        assert f'value="{lang.code}"' in html, f"{lang.code} missing from the picker"
        assert lang.native in html, f"{lang.native} missing from the picker"


def test_every_language_translates_every_string():
    table = _i18n_table()
    assert set(table) == set(languages.codes()), "picker and table disagree"
    keys = set(table["en"])
    for code, strings in table.items():
        missing = keys - set(strings)
        assert not missing, f"{code} is missing {sorted(missing)}"
        assert all(str(v).strip() for v in strings.values()), f"{code} has a blank"


def test_translations_are_not_just_copied_english():
    """A table that silently falls back to English is worse than no table."""
    table = _i18n_table()
    english = table["en"]
    for code in ("de", "fr", "ja", "zh", "ko", "hi", "ru", "el"):
        same = [k for k, v in table[code].items() if v == english[k]]
        # A couple of shared proper nouns are fine; wholesale copying is not.
        assert len(same) <= 2, f"{code} looks untranslated: {same}"


@pytest.mark.parametrize("key", ["lion", "council", "pc_control", "ui_language"])
def test_the_new_controls_are_translated_everywhere(key):
    table = _i18n_table()
    for code, strings in table.items():
        assert strings.get(key), f"{code} has no {key!r}"


def test_every_tagged_label_has_a_key_in_the_table():
    html = HUD.read_text(encoding="utf-8")
    tagged = set(re.findall(r'data-i18n="(\w+)"', html))
    tagged |= set(re.findall(r'data-i18n-ph="(\w+)"', html))
    known = set(_i18n_table()["en"])
    assert tagged <= known, f"tagged but untranslated: {sorted(tagged - known)}"
