"""Hallucination guards: MEDO must not answer what it was not asked.

FAST tier — everyday speech must not trigger skills. A hallucination-audit pass
found three skills whose bare keyword patterns fired on unrelated utterances:
the destructive PowerSkill on "I need some sleep", the WeatherSkill on "sales
forecast", the SystemInfoSkill on "improve my memory". These lock in that the
tightened patterns reject the false positives while still matching the real
commands.

LLM tier (second section) — a reply that SOUNDS like an answer but is not one:
a placeholder token spoken as fact, and a canned pleasantry returned for an
utterance that was never a pleasantry.
"""

from __future__ import annotations

import random

import pytest

from core import speech_text
from core.router import _empty_reply, _is_non_answer, pick_model_tier
from llm.tools import _synthesize_text, dispatch_tool
from skills.base import SkillRegistry, SkillRequest
from skills.lion import LionModeSkill
from skills.media import MediaSkill
from skills.smalltalk import SmallTalkSkill
from skills.system import PowerSkill, SystemInfoSkill
from skills.weather import WeatherSkill, _looks_like_city


def _matches(skill_cls, text: str) -> bool:
    return any(p.search(text) for p in skill_cls.patterns)


@pytest.mark.parametrize("text", [
    "I need to get some sleep tonight",
    "let's restart the discussion",
    "I had to restart the game",
    "shut off the music",
])
def test_power_ignores_everyday_speech(text):
    assert not _matches(PowerSkill, text)


@pytest.mark.parametrize("text", [
    "put the computer to sleep", "sleep the pc", "restart the computer",
    "reboot", "shut down", "shutdown", "shutdown the pc", "lock the screen",
])
def test_power_still_matches_real_commands(text):
    assert _matches(PowerSkill, text)


@pytest.mark.parametrize("text", [
    "how do I improve my memory",
    "the battery in my car is dead",
])
def test_systeminfo_ignores_unrelated(text):
    assert not _matches(SystemInfoSkill, text)


@pytest.mark.parametrize("text", [
    "how much battery do I have", "what's my battery percentage",
    "how much memory am I using", "what's my cpu usage", "system memory",
    "free memory", "available memory", "how much free memory do I have",
])
def test_systeminfo_still_matches_real_questions(text):
    assert _matches(SystemInfoSkill, text)


@pytest.mark.parametrize("text", [
    "what temperature should I cook chicken at",
    "what's my GPU temperature",
    "sales forecast for Q3",
    "what's the sales forecast",
])
def test_weather_ignores_non_weather(text):
    assert not _matches(WeatherSkill, text)


@pytest.mark.parametrize("text", [
    "what's the temperature", "what's the temperature outside",
    "what's the temperature tomorrow", "what will the temperature be tomorrow",
    "what's the temperature for tomorrow",
    "how hot is it", "how cold will it be", "what's the forecast",
    "weather forecast",
])
def test_weather_still_matches_real_questions(text):
    assert _matches(WeatherSkill, text)


# --- Round-3 audit regressions -------------------------------------------

@pytest.mark.parametrize("text", [
    "кое е најдоброто место",           # "which is the best place" — bare место
    "нема доволно место во собата",     # "not enough room in the room"
    "open my google drive",             # bare "drive" means Google Drive, not disk
    "take me for a drive",
])
def test_systeminfo_ignores_bare_place_and_drive(text):
    assert not _matches(SystemInfoSkill, text)


@pytest.mark.parametrize("text", [
    "колку простор имам",               # framed MK disk question
    "слободен простор на дискот",
    "how much disk space", "hard drive", "storage usage",
])
def test_systeminfo_still_matches_framed_disk(text):
    assert _matches(SystemInfoSkill, text)


@pytest.mark.parametrize("candidate,ok", [
    ("work", False), ("home", False), ("next quarter", False),
    ("the office", False), ("my house", False),
    ("London", True), ("New York", True), ("Скопје", True),
])
def test_weather_nonplace_capture_falls_back(candidate, ok):
    assert _looks_like_city(candidate) is ok


@pytest.mark.parametrize("candidate", ["The Hague", "the Netherlands", "London"])
def test_weather_group_path_keeps_determiner_led_real_cities(candidate):
    # The captured "in/for/at <city>" group frames a proper name, so a leading
    # article must NOT drop a real city (regression guard for the group path).
    assert _looks_like_city(candidate, allow_article=True) is True


@pytest.mark.parametrize("candidate", ["work", "my house", "next quarter",
                                       "the office", "the gym"])
def test_weather_group_path_still_drops_nonplaces(candidate):
    # "the office"/"the gym" mean the current location, not a city to geocode:
    # a leading article is stripped before the non-place check.
    assert _looks_like_city(candidate, allow_article=True) is False


def _lion_dir(text):
    m = next((p.search(text) for p in LionModeSkill.patterns if p.search(text)), None)
    assert m is not None, text
    gd = m.groupdict()
    return bool(gd.get("on") or gd.get("on2")), bool(gd.get("off") or gd.get("off2"))


@pytest.mark.parametrize("text,want_on", [
    ("enable lion mode", True), ("activate lion mode", True),
    ("disable lion mode", False), ("deactivate lion mode", False),
])
def test_lion_reversed_order_keeps_direction(text, want_on):
    on, off = _lion_dir(text)
    # Reversed word order ("enable lion mode") must set a direction, not fall
    # through to the bare-"lion mode" toggle branch.
    assert on is want_on and off is (not want_on)


# --------------------------------------------------------------------------- #
# LLM-TIER hallucination guards.
#
# From one live Macedonian exchange (bugs/bug-2026-09-06-112349): asked where
# the village Другово is, MEDO thought for 24 seconds and said "Nil."; asked
# again it produced a sentence with the place name TRANSLATED ("во Other");
# then, on "Твоја бога.", it answered "At your service." — in English, to a
# greeting that never happened. The first and third are guarded here.
# --------------------------------------------------------------------------- #

def _tool_request(text: str, utterance: str) -> SkillRequest:
    """A request as the LLM tool path builds it: synthetic text, real utterance."""
    return SkillRequest(text=text, context={"via": "tool", "utterance": utterance})


def test_utterance_prefers_the_real_words_over_the_synthetic_stand_in():
    req = _tool_request(_synthesize_text("smalltalk", {}), "Твоја бога.")
    assert req.text == "smalltalk"          # what dispatch_tool invents
    assert req.utterance == "Твоја бога."   # what the user actually said


def test_utterance_falls_back_to_text_on_the_fast_path():
    # No context: the fast path's text IS the utterance, so nothing changes.
    assert SkillRequest(text="how are you").utterance == "how are you"
    assert SkillRequest(text="здраво", context={"lang": "mk"}).utterance == "здраво"


async def test_smalltalk_answers_macedonian_in_macedonian_via_the_tool_path():
    """The reported bug: the synthetic English stand-in picked the language."""
    skill = SmallTalkSkill(rng=random.Random(0))
    req = _tool_request(_synthesize_text("smalltalk", {}), "Здраво, Медо.")
    result = await skill.execute(req)
    assert result.speech in SmallTalkSkill._REPLIES["greet"]["mk"]
    assert "At your service" not in result.speech


async def test_smalltalk_declines_what_is_not_small_talk():
    """"Твоја бога." is not a greeting; a canned hello is a fabricated reply."""
    skill = SmallTalkSkill(rng=random.Random(0))
    result = await skill.execute(
        _tool_request(_synthesize_text("smalltalk", {}), "Твоја бога."))
    assert not result.success                      # -> the model answers instead
    assert result.speech not in SmallTalkSkill._REPLIES["greet"]["en"]
    assert result.speech not in SmallTalkSkill._REPLIES["greet"]["mk"]


@pytest.mark.parametrize("text,category", [
    ("how are you?", "how_are_you"), ("hello", "greet"), ("thanks", "thanks"),
    ("good night", "night"), ("како си", "how_are_you"), ("здраво", "greet"),
])
async def test_smalltalk_still_answers_real_pleasantries(text, category):
    skill = SmallTalkSkill(rng=random.Random(0))
    result = await skill.execute(SkillRequest(text=text))
    assert result.success
    assert result.data["smalltalk"] == category


async def test_unknown_tool_still_reports_cleanly():
    # The decline path must not be mistaken for a missing tool.
    result = await dispatch_tool(SkillRegistry(), "no_such_tool", {}, {})
    assert not result.success


@pytest.mark.parametrize("reply", [
    "Nil.", "nil", "NULL", "N/A", "n/a.", "NaN", "undefined", "-", "--",
    "...", "…", "?", "  Nil  ",
])
def test_placeholder_replies_are_not_answers(reply):
    assert _is_non_answer(reply)


@pytest.mark.parametrize("reply", [
    # Each of these is a legitimate spoken answer and must survive untouched.
    "No.", "Nothing.", "None of them.", "Unknown, sorry.", "Nothing on today.",
    "Nil desperandum is the phrase you're after.",
    "Другово е село во близина на Кичево.", "It's 27 degrees and clear.",
    "no", "nothing",
])
def test_real_answers_are_left_alone(reply):
    assert not _is_non_answer(reply)


def test_empty_reply_is_spoken_in_the_language_of_the_turn():
    assert _empty_reply("mk") != _empty_reply("en")
    assert _empty_reply("mk") == _empty_reply("MK")      # code case/'mk-MK'
    # An unknown language falls back to English rather than to nothing.
    assert _empty_reply("ja") == _empty_reply(None) == _empty_reply("en")

if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])


# --- the tier that made all of this likely -----------------------------------

@pytest.mark.parametrize("text", [
    "Твоја бога.",
    "Дали може да ми кажеш каде се наоѓа селото Другово?",
    "здраво",
    "какво е времето",
])
def test_macedonian_never_goes_to_the_small_english_model(text):
    """`fast_model` is a small English-first model; it misreads Macedonian and
    still gets the whole tool list. Live, that turned "Твоја бога." into
    power(action=lock) and a village into locate_in_app(app="Other App")."""
    assert pick_model_tier(text) == "strong"


@pytest.mark.parametrize("text", [
    "what time is it", "hello", "turn the volume up", "what's the weather",
])
def test_simple_english_turns_still_take_the_fast_model(text):
    assert pick_model_tier(text) == "fast"


@pytest.mark.parametrize("text", [
    "why does this happen", "explain the tradeoffs", "debug this python function",
])
def test_complex_english_turns_still_take_the_strong_model(text):
    assert pick_model_tier(text) == "strong"


def test_utterance_is_empty_rather_than_the_models_own_words():
    """A tool call with no utterance has none — the stand-in is not evidence.

    `power` asks whether the user said "lock"; the stand-in for that very call
    is "power lock", so falling back to it confirmed the model against itself
    and the screen locked anyway.
    """
    req = SkillRequest(text="power lock", args={"action": "lock"},
                       context={"via": "tool"})
    assert req.utterance == ""


# --- a bare Macedonian transport verb had no pattern -------------------------

@pytest.mark.parametrize("text", [
    "Паузирај.", "Паузирај", "Продолжи.", "Паузирај го.", "Паузирај ја.",
    "Следна.", "Претходна.",
])
def test_bare_macedonian_transport_verbs_reach_the_media_skill(text):
    """"Паузирај." matched nothing (the MK pattern needs a following noun), so
    it fell to the LLM and came back as media_volume_down — a pause that turned
    the volume down."""
    assert _matches(MediaSkill, text)


@pytest.mark.parametrize("text", [
    "паузирај ја песната", "пушти музика", "следна песна", "претходна нумера",
])
def test_macedonian_media_phrases_still_match(text):
    assert _matches(MediaSkill, text)


# --- written answers read aloud ----------------------------------------------

@pytest.mark.parametrize("reply", [
    "| Number | Square | Cube |\n|---|---|---|\n| 1 | 1 | 1 |",   # a table
    "Use $(10 + n)^2 = 100 + 20n + n^2$ for 11 to 15.",           # LaTeX
    "$$x^2 + y^2 = z^2$$",
    "--- \nSquares (1-15)",                                        # rule
    "## Squares\nLearn them by rote.",                             # heading
    "- Flashcards.\n- Group by patterns.\n- Practice daily.",      # a list
    "1. Learn the basics.\n2. Then the patterns.",
    "```python\nprint(1)\n```",                                    # still caught
])
def test_written_answers_are_recognised_as_written(reply):
    """The voice read a six-minute markdown essay aloud, tables and LaTeX and
    all, because only fences and JSON counted as "written"."""
    assert speech_text.looks_like_markup(reply)


@pytest.mark.parametrize("reply", [
    "It's 27 degrees and clear.",
    "Другово е село во близина на Кичево.",
    "That costs $5, or $10 with shipping.",          # money is not LaTeX
    "I paused the music - it was getting loud.",     # a stray dash
    "Yes - I can do that for you.",
    "The file is called notes_2026.txt.",
])
def test_ordinary_spoken_answers_are_not_flagged(reply):
    assert not speech_text.looks_like_markup(reply)


@pytest.mark.parametrize("text", [
    "Паузирај а музиката, не да гузга сис компјутерот.",   # the live transcript
    "паузирај а музиката",
    "пушти а музиката",
])
def test_whisper_clitic_artifact_still_reaches_media(text):
    """Whisper writes the clitic "ја" as a bare "а". Without it, "pause the
    music" reached the LLM and came back offering to sleep the machine."""
    assert _matches(MediaSkill, text)


@pytest.mark.parametrize("text", [
    "а музиката е гласна",          # no verb: not a command
    "што а тоа",
])
def test_the_clitic_artifact_does_not_widen_the_pattern(text):
    assert not _matches(MediaSkill, text)
