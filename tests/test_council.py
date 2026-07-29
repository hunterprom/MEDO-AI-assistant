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


# --- the broadened bench (marketing, design, data, cybersecurity, ...) --------

@pytest.mark.parametrize("spoken,key", [
    ("marketing", "marketing"),
    ("the marketing strategist", "marketing"),
    ("marketer", "marketing"),
    ("маркетинг", "marketing"),
    ("designer", "design"),
    ("the product designer", "design"),
    ("the data scientist", "data"),
    ("ml engineer", "data"),
    ("the security engineer", "cybersecurity"),
    ("pentester", "cybersecurity"),
    ("chemist", "chemistry"),
    ("doctor", "medicine"),
    ("the physician", "medicine"),
    ("biologist", "biology"),
    ("the writer", "writing"),
])
def test_new_specialists_resolve_by_name(spoken, key):
    found = find_specialist(spoken)
    assert found is not None and found.key == key


@pytest.mark.parametrize("question,key", [
    ("help me with brand positioning and the marketing funnel", "marketing"),
    ("my neural network is overfitting the dataset", "data"),
    ("is there an injection vulnerability or exploit in this", "cybersecurity"),
    ("what epoxy adhesive should I use", "chemistry"),
    ("proofread this paragraph and fix the grammar", "writing"),
])
def test_new_specialists_are_routed_by_subject(question, key):
    assert key in [s.key for s in rank_specialists(question)]


def test_find_specialist_ignores_tiny_fragments():
    # "have a look at my screen" -> the who-capture yields the bare article "a",
    # which must NOT substring-match into "electricAl engineer" and hijack the
    # turn from the vision skill. A fragment under 4 chars resolves to nobody.
    assert find_specialist("a") is None
    assert find_specialist("the") is None
    skill = AskSpecialistSkill(load_settings())
    assert skill.match("have a look at my screen") is None
    assert skill.match("have a look at this") is None


def test_rank_specialists_no_longer_prefix_matches_unrelated_words():
    # Latin triggers are whole-word (plural-tolerant), so these common words in
    # unrelated questions no longer drag in the wrong specialist.
    assert rank_specialists("what is the team currently working on") == []   # not electrical
    assert rank_specialists("should we investigate the lawn drainage") == []  # not finance/law
    assert "physics" not in [s.key for s in
                             rank_specialists("the part was forced into place")]
    # ...and the economics 'market' trigger no longer fires on 'marketing'.
    assert "economics" not in [s.key for s in
                               rank_specialists("help with the marketing funnel")]


def test_rank_specialists_stem_triggers_still_inflect():
    for q, key in [("improve our advertising campaign", "marketing"),
                   ("any injection vulnerability or exploitation", "cybersecurity"),
                   ("explain quantum entanglement", "quantum"),
                   ("my model keeps overfitting", "data"),
                   ("the return on this investment", "finance")]:
        assert key in [s.key for s in rank_specialists(q)], q
    # a plain plural still matches its whole-word trigger
    assert "electrical" in [s.key for s in rank_specialists("what resistors do I need")]
    assert "medicine" in [s.key for s in rank_specialists("what are the symptoms")]


def test_finance_routes_on_invest_and_investor_but_not_investigate():
    # "invest" is safe as a whole word (matches invest/invests, not "investigate"),
    # so it and "investor" route to finance again.
    assert "finance" in [s.key for s in rank_specialists("should I invest in stocks")]
    assert "finance" in [s.key for s in rank_specialists("is he a good investor")]
    assert "finance" not in [s.key for s in
                             rank_specialists("we should investigate the lawn")]


def test_tightened_triggers_do_not_fire_on_everyday_phrases():
    # "reaction"->"chemical reaction" and dropping bare "brand" stop the two
    # worst everyday false positives.
    assert "chemistry" not in [s.key for s in
                               rank_specialists("what was the reaction to our launch")]
    assert "marketing" not in [s.key for s in
                               rank_specialists("should I buy a brand new servo")]
    # ...while the intended terms still route.
    assert "chemistry" in [s.key for s in rank_specialists("explain this chemical reaction")]
    assert "marketing" in [s.key for s in rank_specialists("help with our branding")]


def test_broadened_bench_keeps_the_roster_deterministic():
    # New majors must not perturb the two rank invariants the router relies on.
    assert rank_specialists("what is the weather in Skopje") == []
    q = "the circuit voltage torque motor equation contract inflation"
    assert len(rank_specialists(q, COUNCIL, 3)) == 3
    # No two specialists share a trigger word (a shared trigger would make a
    # convene non-deterministic about who it pulls in).
    seen: dict[str, str] = {}
    for s in COUNCIL:
        for t in s.triggers:
            assert t not in seen, f"trigger {t!r} shared by {seen.get(t)} and {s.key}"
            seen[t] = s.key


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


@pytest.mark.asyncio
async def test_reversed_order_enable_honours_direction_not_toggle(settings):
    # "activate lion mode" must turn it ON even when it is already on, not
    # toggle it off.
    settings.mode.lion = True
    skill = LionModeSkill(settings)
    r = await skill.execute(SkillRequest(text="activate lion mode",
                                         match=skill.match("activate lion mode")))
    assert settings.mode.lion is True and r.data["lion_mode"] is True


@pytest.mark.asyncio
async def test_reversed_order_disable_honours_direction_not_toggle(settings):
    # "disable lion mode" must turn it OFF even when it is already off, not
    # toggle it on.
    settings.mode.lion = False
    skill = LionModeSkill(settings)
    r = await skill.execute(SkillRequest(text="disable lion mode",
                                         match=skill.match("disable lion mode")))
    assert settings.mode.lion is False and r.data["lion_mode"] is False


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


# --- council brain selection (make the agents actually work) ------------------
# The council role-plays experts on a text prompt. When the selected brain is a
# CLI agent, convening several in parallel would spawn a pile of slow processes
# (the "stuck" failure); Router.council_brain borrows the fast local tool-brain
# instead. Ollama users keep the model they picked.

def _router(settings):
    from core.events import EventBus
    from core.router import Router
    from llm.client import OllamaClient
    from skills.base import SkillRegistry

    return Router(settings, SkillRegistry(), OllamaClient(settings.llm), EventBus())


def test_council_brain_uses_selected_ollama_brain(settings):
    from llm.client import OllamaClient

    settings.llm.provider = "ollama"
    r = _router(settings)
    r.model = "qwen3:30b"
    client, model = r.council_brain()
    assert isinstance(client, OllamaClient) and client is r._llm
    assert model == "qwen3:30b"


def test_council_brain_borrows_local_when_cli_agent(settings, monkeypatch):
    from llm.client import OllamaClient

    monkeypatch.setattr(OllamaClient, "is_available", lambda self: True)
    settings.llm.provider = "claude-code"
    settings.llm.tool_brain_model = "llama3.2:3b"
    r = _router(settings)
    r.model = "claude-sonnet-5"
    client, model = r.council_brain()
    assert client is not r._llm            # a separate LOCAL client, not the CLI
    assert model == "llama3.2:3b"
    assert r._tool_brain_ok is True        # probe memoized — no re-probe per expert
    # A second call must not re-probe (would blow up if it did).
    monkeypatch.setattr(OllamaClient, "is_available",
                        lambda self: (_ for _ in ()).throw(AssertionError("re-probed")))
    assert r.council_brain()[1] == "llama3.2:3b"


def test_council_brain_falls_back_to_cli_when_local_absent(settings, monkeypatch):
    from llm.client import OllamaClient

    monkeypatch.setattr(OllamaClient, "is_available", lambda self: False)
    settings.llm.provider = "claude-code"
    r = _router(settings)
    r.model = "claude-sonnet-5"
    client, model = r.council_brain()
    assert client is r._llm and model == "claude-sonnet-5"   # nothing to borrow
    assert r._tool_brain_ok is False


def test_council_brain_probes_a_down_toolbrain_at_most_once(settings, monkeypatch):
    # The blocking ~2s probe must not fire per specialist in a parallel convene —
    # even on the FAILURE path (tool-brain down). Bounded to once per TTL.
    from llm.client import OllamaClient

    calls = {"n": 0}

    def probe(self):
        calls["n"] += 1
        return False

    monkeypatch.setattr(OllamaClient, "is_available", probe)
    settings.llm.provider = "claude-code"
    r = _router(settings)
    r.model = "claude-sonnet-5"
    for _ in range(4):                       # four specialists resolve the brain
        client, _model = r.council_brain()
        assert client is r._llm              # down -> CLI fallback each time
    assert calls["n"] == 1                   # probed ONCE, not per specialist
