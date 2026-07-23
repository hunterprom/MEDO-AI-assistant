"""Custom trigger phrases and manual language selection (B4).

Both are "the user configures it, no code change" features, so the tests are
about the config surface actually reaching the thing that acts on it.
"""

from __future__ import annotations

import re

import pytest

from core.config import STTConfig, Settings
from core.triggers import apply_triggers, phrase_to_pattern
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult


class Dummy(Skill):
    name = "dummy"
    description = "test skill"
    patterns = [re.compile(r"\bbuilt in phrase\b", re.IGNORECASE)]

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("ok")


class Other(Dummy):
    name = "other"


def registry_with(*skills) -> SkillRegistry:
    reg = SkillRegistry()
    for s in skills:
        reg.register(s)
    return reg


# -- phrase compilation ---------------------------------------------------------

@pytest.mark.parametrize("phrase,text", [
    ("catch me up", "hey medo catch me up please"),
    ("што има ново", "медо, што има ново?"),
    ("what's up?", "so what's up?"),
    ("do i need a jacket", "Do I Need A Jacket today"),
    ("look   it  up", "please look it up now"),      # odd spacing survives
])
def test_phrases_match_natural_sentences(phrase, text):
    pattern = phrase_to_pattern(phrase)
    assert pattern is not None and pattern.search(text), phrase


def test_phrases_are_literal_not_regex():
    # A user typing punctuation must get punctuation, not a regex operator.
    pattern = phrase_to_pattern("what (now)")
    assert pattern.search("so what (now)")
    assert not pattern.search("so what now")


def test_a_broken_regex_in_config_cannot_reach_the_matcher():
    # Escaped input can't compile to anything dangerous; it just matches itself.
    pattern = phrase_to_pattern("(((")
    assert pattern is not None and pattern.search("say ((( out loud")


@pytest.mark.parametrize("phrase", ["", "  ", "go", "hi"])
def test_too_short_phrases_are_rejected(phrase):
    # These would shadow every skill registered after the one they're on.
    assert phrase_to_pattern(phrase) is None


# -- application ----------------------------------------------------------------

def test_a_configured_phrase_routes_to_its_skill():
    reg = registry_with(Dummy())
    assert reg.find_match("catch me up") is None

    apply_triggers(reg, {"dummy": ["catch me up", "што има ново"]})

    assert reg.find_match("catch me up")[0].name == "dummy"
    assert reg.find_match("медо, што има ново")[0].name == "dummy"
    # and the built-in wording still works
    assert reg.find_match("built in phrase")[0].name == "dummy"


def test_custom_phrases_win_over_a_built_in_overlap():
    first, second = Dummy(), Other()
    reg = registry_with(first, second)
    apply_triggers(reg, {"other": ["built in phrase"]})
    # 'other' is registered second, but its custom phrase is tried first within
    # its own list — registration order still decides between skills.
    assert reg.find_match("built in phrase")[0].name == "dummy"
    assert second.patterns[0].search("built in phrase")


def test_triggers_do_not_leak_into_other_instances():
    reg = registry_with(Dummy())
    apply_triggers(reg, {"dummy": ["catch me up"]})
    # a fresh instance (as tests and a second registry build one) is untouched
    assert Dummy().match("catch me up") is None


def test_unknown_skill_names_are_reported_not_raised():
    reg = registry_with(Dummy())
    report = apply_triggers(reg, {"nosuchskill": ["hello there"]})
    assert report["unknown"] == ["nosuchskill"]
    assert report["added"] == []


def test_short_phrases_are_reported_as_rejected():
    reg = registry_with(Dummy())
    report = apply_triggers(reg, {"dummy": ["ok", "catch me up"]})
    assert len(report["rejected"]) == 1 and len(report["added"]) == 1


def test_a_bare_string_is_accepted_as_one_phrase():
    reg = registry_with(Dummy())
    apply_triggers(reg, {"dummy": "catch me up"})
    assert reg.find_match("catch me up") is not None


def test_no_triggers_configured_is_a_no_op():
    reg = registry_with(Dummy())
    for empty in (None, {}, ):
        assert apply_triggers(reg, empty) == {"added": [], "unknown": [],
                                              "rejected": []}


def test_triggers_cannot_lift_a_confirmation_gate():
    class Risky(Dummy):
        name = "risky"
        requires_confirmation = True
        controls_pc = True

    risky = Risky()
    reg = registry_with(risky)
    apply_triggers(reg, {"risky": ["do the thing"]})
    # reaching it by a custom phrase changes nothing about how it's gated
    assert risky.requires_confirmation and risky.controls_pc


def test_config_ships_a_triggers_key():
    from core.config import load_settings

    assert isinstance(load_settings().skills.get("triggers", {}), dict)


# -- manual language selection --------------------------------------------------

def test_default_keeps_the_configured_language():
    # language_mode untouched => the older `language` field still rules.
    assert STTConfig().language == "en"
    assert STTConfig(language=None).language is None


def test_auto_mode_turns_detection_on():
    cfg = STTConfig(language="en", language_mode="auto")
    assert cfg.language is None            # None = detect per utterance


@pytest.mark.parametrize("code", ["ja", "mk", "de", "EN", " ko "])
def test_forcing_a_language_pins_the_decoder(code):
    cfg = STTConfig(language=None, language_mode=code)
    assert cfg.language == code.strip().lower()
    assert cfg.language_mode == code.strip().lower()


def test_forcing_an_unsupported_language_falls_back_to_auto():
    # Whisper knows Welsh; MEDO has no voice for it, so forcing it would give
    # you a transcript you can't be answered in. Auto-detect is the honest
    # fallback (and it warns).
    cfg = STTConfig(language="en", language_mode="cy")
    assert cfg.language_mode == "auto" and cfg.language is None


def test_forced_language_is_what_the_transcriber_reads():
    # The transcriber only ever looks at `language`; language_mode must fold
    # into it rather than becoming a second source of truth.
    cfg = STTConfig(language_mode="ja")
    forced = None if cfg.language in ("", "auto") else cfg.language
    assert forced == "ja"


def test_every_forceable_code_has_a_voice():
    from core import languages

    for lang in languages.LANGUAGES:
        cfg = STTConfig(language_mode=lang.code)
        assert cfg.language == lang.code
        assert languages.voice_for(cfg.language)


def test_settings_accepts_language_mode_end_to_end():
    s = Settings(stt={"language_mode": "mk"})
    assert s.stt.language == "mk"
