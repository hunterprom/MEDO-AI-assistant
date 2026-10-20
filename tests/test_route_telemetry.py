"""P0-3: making the fast path's mistakes visible.

A regex that fires on the wrong sentence answers in under a millisecond, with
total confidence, and looks identical to a correct route in every number MEDO
records. The only witness is the user, who says "no, I meant the weather" a few
seconds later. These tests cover writing that down: what got routed, how much
of the sentence the pattern actually matched, and whether it was taken back.
"""

from __future__ import annotations

import re

import pytest

from core.config import load_settings
from core.corrections import (
    DEFAULT_WINDOW_S,
    corrects_previous,
    is_correction,
    is_refinement,
    starts_with_pivot,
)
from core.events import EventBus
from core.metrics import MetricsStore
from core.router import Router
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult


class _Timer(Skill):
    name = "timers"
    description = "set a timer"
    patterns = [re.compile(r"\bset a timer\b", re.IGNORECASE)]

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("Timer set.")


class _Weather(Skill):
    name = "weather"
    description = "the weather"
    patterns = [re.compile(r"\bweather\b", re.IGNORECASE)]

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("Sunny.")


def _router(tmp_path, *skills: Skill) -> tuple[Router, MetricsStore]:
    settings = load_settings()
    settings.logging.route_telemetry = True
    registry = SkillRegistry()
    for skill in skills or (_Timer(), _Weather()):
        registry.register(skill)
    router = Router(settings, registry, None, EventBus())  # type: ignore[arg-type]
    router.model = None
    store = MetricsStore(tmp_path / "metrics.db")
    router.metrics = store
    return router, store


def _events(store: MetricsStore) -> list[tuple]:
    with store._connect() as conn:
        return conn.execute(
            "SELECT tier, skill, text, confidence, corrected, correction "
            "FROM route_events ORDER BY id").fetchall()


# --- recognising a correction (bilingual, and narrow on purpose) -------------


@pytest.mark.parametrize("text", [
    "no, I meant the weather",
    "that's not what I meant",
    "I didn't say set a timer",
    "not that",
    "wrong one",
    "no, don't do that",
    "actually, I meant tomorrow",
    "that's not it",
])
def test_english_corrections_are_recognised(text):
    assert is_correction(text)


@pytest.mark.parametrize("text", [
    "не, мислев на времето",
    "не мислев тоа",
    "не тоа",
    "не така",
    "мислев на утре",
    "тоа не е тоа",
    "погрешна команда",
])
def test_macedonian_corrections_are_recognised(text):
    """Written in Macedonian, not translated from the English list.

    Four English-only review rounds once passed while MEDO was broken for
    nearly every Macedonian utterance; a detector that only speaks English
    would report a flat 0% misfire rate for half of what this user says.
    """
    assert is_correction(text)


@pytest.mark.parametrize("text", [
    "no",                              # answering a yes/no question
    "what's wrong with my code",       # a question that contains "wrong"
    "set a timer",                     # an ordinary command
    "what did I mean by that",         # musing, not correcting
    "не",                              # a bare no
    "што е погрешно тука",             # "what's wrong here" — a question
])
def test_ordinary_utterances_are_not_corrections(text):
    """A false positive here poisons the very statistic this produces."""
    assert not is_correction(text)


def test_a_refinement_is_not_a_correction():
    assert is_refinement("make it shorter")
    assert not is_correction("make it shorter")


def test_a_pivot_is_recognised_in_both_languages():
    assert starts_with_pivot("no, shorter")
    assert starts_with_pivot("actually, wait")
    assert starts_with_pivot("не, пократко")
    assert not starts_with_pivot("shorter please")


def test_a_correction_outside_the_window_does_not_count():
    assert corrects_previous("no, I meant the weather", elapsed_s=5.0)
    assert not corrects_previous("no, I meant the weather",
                                 elapsed_s=DEFAULT_WINDOW_S + 1)


def test_a_bare_refinement_is_not_blamed_on_the_route():
    """"Shorter" means the reply was long, not that the skill was wrong."""
    assert not corrects_previous("make it shorter", elapsed_s=2.0)
    assert corrects_previous("no, make it shorter", elapsed_s=2.0)


# --- match confidence --------------------------------------------------------


def test_a_full_match_scores_one():
    text = "set a timer"
    assert Router.match_confidence(text, re.search(r"set a timer", text)) == 1.0


def test_a_sliver_of_a_long_sentence_scores_low():
    """The shape of a fast-path false positive, made into a number."""
    text = "honestly the current approach to this whole problem is fine"
    score = Router.match_confidence(text, re.search(r"current", text))
    assert score < 0.2


def test_confidence_is_one_when_there_is_no_regex():
    assert Router.match_confidence("anything", None) == 1.0


# --- what gets written down --------------------------------------------------


@pytest.mark.asyncio
async def test_a_fast_route_is_logged_with_its_confidence(tmp_path):
    router, store = _router(tmp_path)
    await router.route("set a timer")
    rows = _events(store)
    assert len(rows) == 1
    tier, skill, text, confidence, corrected, _correction = rows[0]
    assert (tier, skill, text) == ("FAST", "timers", "set a timer")
    assert confidence == 1.0
    assert corrected == 0


@pytest.mark.asyncio
async def test_a_correction_marks_the_previous_route(tmp_path):
    router, store = _router(tmp_path)
    await router.route("set a timer")
    await router.route("no, I meant the weather")
    first = _events(store)[0]
    assert first[1] == "timers"
    assert first[4] == 1                       # corrected
    assert first[5] == "no, I meant the weather"


@pytest.mark.asyncio
async def test_a_macedonian_correction_marks_it_too(tmp_path):
    router, store = _router(tmp_path)
    await router.route("set a timer")
    await router.route("не, мислев на времето")
    assert _events(store)[0][4] == 1


@pytest.mark.asyncio
async def test_an_unrelated_follow_up_marks_nothing(tmp_path):
    router, store = _router(tmp_path)
    await router.route("set a timer")
    await router.route("weather")
    assert [row[4] for row in _events(store)] == [0, 0]


@pytest.mark.asyncio
async def test_only_the_immediately_previous_route_can_be_corrected(tmp_path):
    """A correction blames the last turn, not everything before it."""
    router, store = _router(tmp_path)
    await router.route("set a timer")
    await router.route("weather")
    await router.route("no, I meant something else")
    rows = _events(store)
    # Three rows now: the correction utterance is itself a turn, and the
    # dashboard shows every tier. Only the SECOND is flagged — the route the
    # user was actually taking back.
    assert [row[4] for row in rows[:2]] == [0, 1]
    assert rows[1][1] == "weather"


@pytest.mark.asyncio
async def test_a_correction_outside_the_window_is_not_attributed(tmp_path):
    router, store = _router(tmp_path)
    router._settings.logging.correction_window_s = 0.0
    await router.route("set a timer")
    await router.route("no, I meant the weather")
    assert _events(store)[0][4] == 0


@pytest.mark.asyncio
async def test_telemetry_can_be_switched_off_entirely(tmp_path):
    router, store = _router(tmp_path)
    router._settings.logging.route_telemetry = False
    await router.route("set a timer")
    assert _events(store) == []


@pytest.mark.asyncio
async def test_a_broken_metrics_store_does_not_break_routing(tmp_path):
    """Telemetry is never allowed to cost a reply."""
    router, _store = _router(tmp_path)

    class _Broken:
        def record(self, *a, **kw):
            raise RuntimeError("disk on fire")

        def record_route_event(self, *a, **kw):
            raise RuntimeError("disk on fire")

        def mark_route_corrected(self, *a, **kw):
            raise RuntimeError("disk on fire")

    router.metrics = _Broken()
    result = await router.route("set a timer")
    # `in`, not `==`: the persona layer may decorate a successful fast-path
    # reply, and what this test is about is that the reply happened at all.
    assert "Timer set." in result.speech


# --- the report --------------------------------------------------------------


def test_the_routing_report_shows_a_misfire_rate(tmp_path):
    store = MetricsStore(tmp_path / "m.db")
    good = store.record_route_event("FAST", "timers", "set a timer", 1.0)
    bad = store.record_route_event("FAST", "power", "the current approach", 0.2)
    assert good and bad
    store.mark_route_corrected(bad, "no, I meant the weather")
    report = store.routing_report()
    assert "power" in report and "50.0%" in report or "100.0%" in report
    assert "no, I meant the weather" in report


def test_the_routing_report_says_so_when_empty(tmp_path):
    assert "No routing telemetry" in MetricsStore(tmp_path / "m.db").routing_report()


def test_pruning_drops_old_telemetry(tmp_path):
    store = MetricsStore(tmp_path / "m.db")
    store.record_route_event("FAST", "timers", "set a timer", 1.0)
    with store._writing() as conn:
        conn.execute("UPDATE route_events SET ts = datetime('now', '-200 days')")
    assert store.prune(keep_days=90) == 1
    assert _events(store) == []


def test_pruning_keeps_recent_telemetry(tmp_path):
    store = MetricsStore(tmp_path / "m.db")
    store.record_route_event("FAST", "timers", "set a timer", 1.0)
    assert store.prune(keep_days=90) == 0
    assert len(_events(store)) == 1


# --- council selection disagreements -----------------------------------------


def test_council_disagreements_are_recorded(tmp_path):
    store = MetricsStore(tmp_path / "m.db")
    store.record_council_selection("what resistor", "llm_vs_wordlist",
                                   "electrical", "electrical", agreed=True)
    store.record_council_selection("is this contract fair", "llm_vs_wordlist",
                                   "law", "finance", agreed=False)
    report = store.routing_report()
    assert "llm_vs_wordlist" in report
    assert "50.0%" in report


@pytest.mark.asyncio
async def test_the_council_logs_when_the_llm_picks_a_different_expert(tmp_path,
                                                                     monkeypatch):
    """A model naming an expert is a free A/B against the word list."""
    from skills.council import AskSpecialistSkill

    settings = load_settings()
    settings.logging.route_telemetry = True
    monkeypatch.setattr(settings.memory, "db_path", str(tmp_path / "m.db"))

    async def ask(system, user):
        return "Use a 220 ohm resistor."

    skill = AskSpecialistSkill(settings, ask=ask)
    await skill.execute(SkillRequest(
        text="ask about this",
        args={"specialist": "lawyer",
              "question": "what resistor do I need for this LED circuit"}))

    store = MetricsStore(tmp_path / "m.db")
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT method, chosen, alternative, agreed "
            "FROM council_selections").fetchall()
    assert len(rows) == 1
    method, chosen, alternative, agreed = rows[0]
    assert method == "llm_vs_wordlist"
    assert chosen == "law"
    assert alternative == "electrical"        # the word list saw "resistor"
    assert agreed == 0


@pytest.mark.asyncio
async def test_the_council_logs_agreement_too(tmp_path, monkeypatch):
    """Disagreements alone have no denominator — agreement is the other half."""
    from skills.council import AskSpecialistSkill

    settings = load_settings()
    settings.logging.route_telemetry = True
    monkeypatch.setattr(settings.memory, "db_path", str(tmp_path / "m.db"))

    async def ask(system, user):
        return "220 ohms."

    skill = AskSpecialistSkill(settings, ask=ask)
    await skill.execute(SkillRequest(
        text="ask about this",
        args={"specialist": "electrical engineer",
              "question": "what resistor do I need for this LED circuit"}))
    store = MetricsStore(tmp_path / "m.db")
    with store._connect() as conn:
        rows = conn.execute("SELECT agreed FROM council_selections").fetchall()
    assert rows == [(1,)]


@pytest.mark.asyncio
async def test_a_fast_path_council_call_is_not_logged_as_a_comparison(tmp_path,
                                                                      monkeypatch):
    """No model named anyone, so there is no second opinion to compare against."""
    from skills.council import AskSpecialistSkill

    settings = load_settings()
    settings.logging.route_telemetry = True
    monkeypatch.setattr(settings.memory, "db_path", str(tmp_path / "m.db"))

    async def ask(system, user):
        return "Answer."

    skill = AskSpecialistSkill(settings, ask=ask)
    match = skill.match("ask the electrical engineer about grounding")
    await skill.execute(SkillRequest(
        text="ask the electrical engineer about grounding", match=match))
    store = MetricsStore(tmp_path / "m.db")
    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM council_selections").fetchone()[0] == 0
