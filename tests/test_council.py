"""The specialist council, MEDO Lion Mode, and the wiring helper."""

from __future__ import annotations

import pytest

from core.config import load_settings
from core.council import (
    COUNCIL,
    enabled_council,
    find_specialist,
    load_council,
    rank_specialists,
    system_prompt,
)
from skills.base import SkillRequest
from skills.circuit import CircuitSkill, extract_code, split_sections
from skills.council import AskSpecialistSkill, ConveneCouncilSkill
from skills.lion import LionModeSkill


@pytest.fixture
def settings():
    return load_settings()


def _expert(reply="answer"):
    async def ask(system, user):
        return reply
    return ask


# --- the roster ---------------------------------------------------------------


def test_find_specialist_by_key_and_title():
    assert find_specialist("electrical").key == "electrical"
    assert find_specialist("the electrical engineer").key == "electrical"
    assert find_specialist("roboticist").key == "robotics"
    assert find_specialist("nobody at all") is None


@pytest.mark.parametrize("spoken,key", [
    # Macedonian glues the article on, so these are alias + suffix.
    ("електроинженерот", "electrical"),
    ("електроинженер", "electrical"),
    ("адвокатот", "law"),
    ("правникот", "law"),
    ("физичарот", "physics"),
    ("математичарот", "mathematics"),
    ("роботичарот", "robotics"),
    ("програмерот", "software"),
    ("економистот", "economics"),
    ("машинскиот инженер", "mechanical"),
])
def test_find_specialist_resolves_macedonian_titles(spoken, key):
    # The Macedonian "прашај го …" pattern has always existed; until aliases
    # were added it matched and then resolved to nobody, so every Macedonian
    # specialist request dead-ended in "I don't have that specialist".
    found = find_specialist(spoken)
    assert found is not None and found.key == key


def test_macedonian_request_reaches_the_specialist_skill():
    skill = AskSpecialistSkill(load_settings(), _expert())
    assert skill.match("прашај го електроинженерот за заземјување") is not None


@pytest.mark.parametrize("text", [
    "what does this page say about batteries",
    "what does this article say about tariffs",
    "прашај ја оваа страница за батерии",
    "ask nobody about anything",
])
def test_a_name_that_is_not_a_specialist_is_not_claimed(text):
    # The patterns must capture a free-form name, which made them greedy:
    # "this page" was captured as an expert and the fast path stopped there,
    # so web_fetch never got the chance to read the page.
    skill = AskSpecialistSkill(load_settings(), _expert())
    assert skill.match(text) is None


def test_config_can_add_aliases_for_a_custom_specialist():
    council = load_council({"chef": {"prompt": "You cook.", "title": "the chef",
                                     "aliases": ["готвач"]}})
    assert find_specialist("готвачот", council).key == "chef"


def test_rank_specialists_picks_the_relevant_fields():
    assert "electrical" in [s.key for s in rank_specialists(
        "what resistor do I need for this led circuit")]
    assert "robotics" in [s.key for s in rank_specialists(
        "what torque does the servo need")]


def test_rank_specialists_is_empty_off_topic():
    assert rank_specialists("what is the weather in Skopje") == []


def test_rank_specialists_is_bounded_and_stable():
    q = "the circuit voltage torque motor equation contract inflation"
    first = rank_specialists(q, COUNCIL, limit=3)
    assert len(first) == 3
    assert [s.key for s in first] == [s.key for s in rank_specialists(q, COUNCIL, 3)]


def test_council_can_be_extended_and_disabled():
    extended = load_council({"metallurgy": {"title": "the metallurgist",
                                            "prompt": "You are a metallurgist.",
                                            "triggers": ["alloy"]}})
    assert find_specialist("metallurgy", extended) is not None
    # A malformed entry is skipped, never fatal at startup.
    assert load_council({"junk": {"title": "no prompt"}}) == COUNCIL
    trimmed = enabled_council(COUNCIL, ["law", "finance"])
    assert {s.key for s in trimmed}.isdisjoint({"law", "finance"})


def test_system_prompt_carries_house_style_and_language():
    prompt = system_prompt(COUNCIL[0], macedonian=True)
    assert "no markdown" in prompt and "Macedonian" in prompt


# --- asking -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_specialist_names_who_answered(settings):
    skill = AskSpecialistSkill(settings, _expert("Use a 220 ohm resistor."))
    phrase = "ask the electrical engineer about the led resistor"
    r = await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert r.success and "electrical engineer" in r.speech
    assert "220 ohm" in r.speech


@pytest.mark.asyncio
async def test_unknown_specialist_lists_the_real_ones(settings):
    skill = AskSpecialistSkill(settings, _expert())
    phrase = "ask the astrologer about mercury"
    r = await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert r.success is False and "I have" in r.speech


@pytest.mark.asyncio
async def test_no_brain_degrades_cleanly(settings):
    skill = AskSpecialistSkill(settings, None)
    phrase = "ask the physicist about momentum"
    r = await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert r.success is False


@pytest.mark.asyncio
async def test_convene_asks_several_then_synthesizes(settings):
    asked = []

    async def ask(system, user):
        asked.append(system[:40])
        return "a note"

    async def synth(question, notes, macedonian=False):
        return "combined answer"

    skill = ConveneCouncilSkill(settings, ask, synth)
    phrase = "convene the council on motor torque and battery voltage"
    r = await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert r.success and "combined answer" in r.speech
    assert len(asked) >= 2                       # more than one field consulted
    assert len(r.data["members"]) == len(asked)


@pytest.mark.asyncio
async def test_one_dead_specialist_does_not_kill_the_council(settings):
    calls = {"n": 0}

    async def flaky(system, user):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("that one is down")
        return "still here"

    async def synth(question, notes, macedonian=False):
        return "combined"

    skill = ConveneCouncilSkill(settings, flaky, synth)
    phrase = "convene the council on motor torque and battery voltage"
    r = await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert r.success


@pytest.mark.asyncio
async def test_convene_falls_back_when_no_field_matches(settings):
    skill = ConveneCouncilSkill(settings, _expert("generalist note"), None)
    phrase = "convene the council on what to have for lunch"
    r = await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert r.success and r.data["members"]


# --- lion mode ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_lion_mode_turns_on_without_a_confirmation(settings):
    # It lowers no guardrail, so gating it behind a prompt would only teach the
    # habit of clicking through security dialogs.
    settings.mode.lion = False
    skill = LionModeSkill(settings)
    r = await skill.execute(SkillRequest(text="lion mode on",
                                         match=skill.match("lion mode on")))
    assert r.success and r.needs_confirmation is False
    assert settings.mode.lion is True


@pytest.mark.asyncio
async def test_lion_on_reply_states_the_safety_gate_is_unchanged(settings):
    settings.mode.lion = False
    skill = LionModeSkill(settings)
    r = await skill.execute(SkillRequest(text="lion mode on",
                                         match=skill.match("lion mode on")))
    # the spoken reply must not imply any relaxation of safety
    assert "unchanged" in r.speech.lower() or "still ask" in r.speech.lower()
    assert "stop asking" not in r.speech.lower()


@pytest.mark.asyncio
async def test_lion_mode_off_never_asks(settings):
    settings.mode.lion = True
    skill = LionModeSkill(settings)
    r = await skill.execute(SkillRequest(text="lion mode off",
                                         match=skill.match("lion mode off")))
    assert r.success and r.needs_confirmation is False
    assert settings.mode.lion is False


@pytest.mark.asyncio
async def test_lion_mode_macedonian(settings):
    settings.mode.lion = True
    skill = LionModeSkill(settings)
    r = await skill.execute(SkillRequest(text="исклучи лав мод",
                                         match=skill.match("исклучи лав мод")))
    assert settings.mode.lion is False and "Лав" in r.speech


def test_lion_mode_defaults_off_every_start():
    # It is a mode for a task, not a setting you leave behind.
    assert load_settings().mode.lion is False


# --- wiring helper ------------------------------------------------------------


ANSWER = (
    "CONNECTIONS:\n"
    "Battery + -> Arduino VIN\n"
    "Battery - -> Arduino GND\n"
    "Arduino D9 -> 220 ohm resistor -> LED anode\n"
    "LED cathode -> Arduino GND\n"
    "WATCH OUT:\n"
    "Without the 220 ohm resistor the LED and the pin both die.\n"
    "CODE:\n"
    "```cpp\n"
    "void setup() { pinMode(9, OUTPUT); }\n"
    "void loop() { digitalWrite(9, HIGH); }\n"
    "```\n"
)


def test_split_sections_and_extract_code():
    parts = split_sections(ANSWER)
    assert "Battery + -> Arduino VIN" in parts["connections"]
    assert "220 ohm resistor" in parts["watchout"]
    code = extract_code(parts["code"])
    assert code.startswith("void setup()") and "`" not in code


def test_extract_code_handles_none():
    assert extract_code("none") == ""
    assert extract_code("") == ""


def test_split_sections_survives_a_missing_heading():
    assert split_sections("just prose, no headings at all") == {}


@pytest.mark.asyncio
async def test_circuit_returns_connections_and_saves_a_sketch(settings, tmp_path):
    settings.conversation.dictation_dir = str(tmp_path)
    skill = CircuitSkill(settings, _expert(ANSWER),
                         _expert("Battery to the Arduino, LED on D9."))
    phrase = "how do I wire an led to an arduino with code"
    r = await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert r.success
    assert "Battery + -> Arduino VIN" in r.data["connections"]
    assert r.data["code"].startswith("void setup()")
    sketches = list(tmp_path.glob("*.ino"))
    assert sketches and "digitalWrite" in sketches[0].read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_circuit_without_asking_for_code_writes_no_file(settings, tmp_path):
    settings.conversation.dictation_dir = str(tmp_path)
    skill = CircuitSkill(settings, _expert(ANSWER), None)
    phrase = "how do I wire an led to an arduino"
    r = await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert r.success and r.data["sketch_path"] is None
    assert list(tmp_path.glob("*.ino")) == []


@pytest.mark.asyncio
async def test_circuit_with_no_brain_says_so(settings):
    skill = CircuitSkill(settings, None, None)
    phrase = "how do I wire an led to an arduino"
    r = await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert r.success is False
