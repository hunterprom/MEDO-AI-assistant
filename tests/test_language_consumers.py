"""Constrained two-language mode — S3: language-dependent features follow the
ACTIVE set (no hardcoded en/mk).

Covers the three consumers the master prompt calls out: the confirmation-word
gate, persona quips, and TTS voice selection.
"""

from __future__ import annotations

import random

import pytest

from core.config import PROJECT_ROOT, PersonalityConfig
from core.persona import QUIPS, Persona


# --- confirmation-word banks -------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_default_bank():
    # Every test starts from the shipped default; restore it afterward so the
    # module-global bank never leaks between tests.
    import core.safety as safety
    safety.configure_confirm_words(["en", "mk"], "en")
    yield
    safety.configure_confirm_words(["en", "mk"], "en")


def test_shipped_banks_exist():
    for code in ("en", "mk", "es", "de"):
        assert (PROJECT_ROOT / "lang" / "confirm_words" / f"{code}.yaml").exists()


def test_default_active_matches_en_and_mk():
    import core.safety as safety
    assert safety.is_affirmative("yes") and safety.is_affirmative("да")
    assert safety.is_negative("no") and safety.is_negative("не")


def test_switching_to_non_mk_pair_drops_mk_words():
    import core.safety as safety
    safety.configure_confirm_words(["en", "es"], "en")
    # Macedonian no longer matches; Spanish does; English (shared) still does.
    assert not safety.is_affirmative("да")
    assert safety.is_affirmative("sí")
    assert safety.is_affirmative("yes")
    assert safety.is_negative("cancela")
    assert not safety.is_negative("откажи")


def test_single_active_language_excludes_the_other():
    import core.safety as safety
    safety.configure_confirm_words(["en"], "en")
    assert safety.is_affirmative("yes")
    assert not safety.is_affirmative("да")


def test_missing_bank_falls_back_to_primary_plus_english():
    import core.safety as safety
    # fr has no shipped bank -> warn + fall back so a yes/no still parses.
    safety.configure_confirm_words(["fr"], "fr")
    assert safety.is_affirmative("yes")   # English floor keeps the gate working
    assert safety.is_negative("no")


def test_load_confirm_words_is_active_only():
    from core.safety import load_confirm_words
    aff, neg = load_confirm_words(["en", "mk"], "en")
    assert "да" in aff and "yes" in aff
    aff2, _ = load_confirm_words(["en", "de"], "en")
    assert "ja" in aff2 and "да" not in aff2


# --- persona quips -----------------------------------------------------------

def _persona(active, primary="en"):
    cfg = PersonalityConfig(style="dry_wit", wit_level=1.0, quips_language="match")
    return Persona(cfg, rng=random.Random(0), active=active, primary=primary)


def test_quip_uses_active_language():
    p = _persona(["en", "mk"])
    out = p.decorate("Timer set.", skill_name="timers", user_text="", language="mk")
    quip = out.replace("Timer set. ", "")
    assert quip in QUIPS["timers"]["mk"]


def test_quip_degrades_to_primary_when_language_not_active():
    # mk is detected but NOT active -> must quip in primary (en), never mk.
    p = _persona(["en", "es"], primary="en")
    out = p.decorate("Timer set.", skill_name="timers", user_text="", language="mk")
    quip = out.replace("Timer set. ", "")
    assert quip in QUIPS["timers"]["en"]
    assert quip not in QUIPS["timers"]["mk"]


def test_quip_degrades_when_no_bank_for_active_language():
    # es is active but has no quip bank -> degrade to primary en.
    p = _persona(["en", "es"], primary="en")
    out = p.decorate("Opening.", skill_name="apps", user_text="", language="es")
    quip = out.replace("Opening. ", "")
    assert quip in QUIPS["apps"]["en"]


# --- TTS voice selection -----------------------------------------------------

def test_edge_voice_follows_language_with_fallback(monkeypatch):
    import voice.tts as tts
    monkeypatch.setattr(tts, "find_ffmpeg", lambda: "ffmpeg")
    edge = tts.EdgeTTS("en-US-AvaNeural")     # primary (en) voice as fallback
    from core import languages
    assert edge.voice_for("mk") == languages.get("mk").voice
    assert edge.voice_for("de") == languages.get("de").voice
    # an unknown language falls back to the construction (primary) voice, once.
    assert edge.voice_for("zz") == "en-US-AvaNeural"
    assert edge._warned_fallback is True


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
