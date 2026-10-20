"""P2-5: "actually, make it shorter" — correcting an answer while it happens.

Barge-in already stopped MEDO talking. What it did not do was understand WHY:
the next utterance was routed as a brand-new request, so cutting in with "make
it shorter" produced a confused answer to the words "make it shorter" and the
original question was gone.

The rule this implements: an utterance that follows an interruption, and reads
as a correction or a refinement, is folded back into the ORIGINAL request and
re-answered. Everything else — including an ordinary command shouted over the
top — routes normally.
"""

from __future__ import annotations

import re

import pytest

from core.config import load_settings
from core.events import EventBus
from core.router import CORRECTION_PROMPT, Router
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult


class _Clock(Skill):
    name = "clock"
    description = "tell the time"
    patterns = [re.compile(r"\bwhat time is it\b", re.IGNORECASE)]

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("It's noon.")


class _RecordingLLM:
    """Captures the prompt it was asked to answer."""

    model = "m"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def chat(self, model, messages, tools=None, **kw):
        user = [m for m in messages if m.get("role") == "user"]
        if user:
            self.prompts.append(user[-1]["content"])
        return {"content": "A shorter answer."}


class _Asker(Skill):
    """Asks a question and wants the next utterance as its answer."""

    name = "asker"
    description = "fix something"
    patterns = [re.compile(r"\bfix something\b", re.IGNORECASE)]

    async def execute(self, request: SkillRequest) -> SkillResult:
        if request.context.get("captured_reply"):
            return SkillResult("Right you are.")
        return SkillResult("What should I fix?", await_reply=True)


def _router(*extra: Skill) -> tuple[Router, _RecordingLLM]:
    llm = _RecordingLLM()
    registry = SkillRegistry()
    registry.register(_Clock())
    for skill in extra:
        registry.register(skill)
    settings = load_settings()
    router = Router(settings, registry, llm, EventBus())
    return router, llm


# --- after an interruption -----------------------------------------------------


@pytest.mark.asyncio
async def test_a_refinement_after_cutting_in_re_answers_the_original():
    router, llm = _router()
    router.note_interruption("explain how a diode works",
                             "A diode is a two-terminal semiconductor")
    await router.route("actually, make it shorter")
    assert llm.prompts, "the model should have been asked again"
    prompt = llm.prompts[-1]
    assert "explain how a diode works" in prompt
    assert "make it shorter" in prompt
    assert "two-terminal semiconductor" in prompt      # what it had said


@pytest.mark.asyncio
async def test_a_bare_refinement_counts_once_the_user_has_cut_in():
    """Nobody interrupts to make an unrelated remark."""
    router, llm = _router()
    router.note_interruption("explain how a diode works", "")
    await router.route("shorter")
    assert "explain how a diode works" in llm.prompts[-1]


@pytest.mark.asyncio
async def test_the_model_is_told_not_to_apologise():
    """A local model otherwise spends its first sentence saying sorry — to
    someone who cut in precisely to save time."""
    assert "not" in CORRECTION_PROMPT and "apologise" in CORRECTION_PROMPT


@pytest.mark.asyncio
async def test_a_real_command_after_cutting_in_is_not_a_correction():
    router, llm = _router()
    router.note_interruption("explain how a diode works", "A diode is")
    result = await router.route("what time is it")
    assert result.skill_name == "clock"
    assert llm.prompts == []


@pytest.mark.asyncio
async def test_the_arming_lasts_exactly_one_turn():
    """A stale interruption would turn a later question into a strange re-answer."""
    router, llm = _router()
    router.note_interruption("explain how a diode works", "A diode is")
    await router.route("what time is it")            # consumes the arming
    await router.route("make it shorter")
    assert llm.prompts == [] or "diode" not in llm.prompts[-1]


@pytest.mark.asyncio
async def test_an_empty_question_does_not_arm_anything():
    router, llm = _router()
    router.note_interruption("", "partial")
    await router.route("make it shorter")
    assert llm.prompts == [] or "make it shorter" == llm.prompts[-1]


# --- without an interruption ---------------------------------------------------


@pytest.mark.asyncio
async def test_an_explicit_correction_works_without_an_interruption():
    """"No, I meant the weather" disowns the last turn whether or not you cut in."""
    router, llm = _router()
    router.conversation.add_turn("what time is it", "It's noon.")
    await router.route("no, I meant the weather")
    assert "what time is it" in llm.prompts[-1]
    assert "I meant the weather" in llm.prompts[-1]


@pytest.mark.asyncio
async def test_a_calm_refinement_is_not_treated_as_a_correction():
    """Said calmly after an answer, "make it shorter" could start something new;
    guessing wrong loses the request entirely."""
    router, llm = _router()
    router.conversation.add_turn("explain diodes", "A diode is a thing.")
    await router.route("make it shorter")
    assert llm.prompts == [] or "explain diodes" not in llm.prompts[-1]


@pytest.mark.asyncio
async def test_a_correction_with_no_history_routes_normally():
    router, llm = _router()
    result = await router.route("no, I meant the weather")
    assert result is not None                # answered somehow, did not crash


# --- conversation state survives -----------------------------------------------


@pytest.mark.asyncio
async def test_the_conversation_records_what_the_user_actually_said():
    """Not the rewritten prompt: the transcript must stay a transcript."""
    router, _llm = _router()
    router.note_interruption("explain how a diode works", "A diode is")
    await router.route("actually, make it shorter")
    assert router.conversation.last_question() == "actually, make it shorter"


@pytest.mark.asyncio
async def test_earlier_turns_are_not_lost():
    router, _llm = _router()
    router.conversation.add_turn("hello", "hi")
    router.note_interruption("explain how a diode works", "A diode is")
    await router.route("actually, make it shorter")
    messages = router.conversation.recent_messages()
    assert messages[0]["content"] == "hello"
    assert len(messages) >= 4


@pytest.mark.asyncio
async def test_a_correction_reaches_the_llm_not_the_fast_path():
    """"Make it shorter" must not be routed by a regex that happens to match."""
    router, llm = _router()
    router.note_interruption("what time is it in Tokyo", "It is")
    await router.route("no, I meant shorter")
    assert llm.prompts, "the correction went to a skill instead of the model"


def test_note_interruption_is_safe_to_call_repeatedly():
    router, _llm = _router()
    router.note_interruption("first", "a")
    router.note_interruption("second", "b")
    assert router._interrupted_turn == ("second", "b")


@pytest.mark.asyncio
async def test_a_turn_answered_by_a_gate_still_spends_the_arming():
    """One turn means ONE turn — including a turn that never reaches the
    correction check because a pending gate claimed it first.

    Cutting in over a question ("what should I fix?") and then answering it is
    an ordinary exchange. The arming was taken inside the correction check,
    which that turn returns above, so it survived — and the next refinement,
    about something else entirely, was folded back into the interrupted
    question instead of being routed on its own.
    """
    router, llm = _router(_Asker())
    await router.route("fix something")                  # arms the reply capture
    router.note_interruption("explain how a diode works", "A diode is")
    answer = await router.route("the login bug")         # the gate takes this turn
    assert answer.speech == "Right you are."
    assert router._interrupted_turn is None
    await router.route("make it shorter")
    assert llm.prompts == [] or "diode" not in llm.prompts[-1]


class _Confirms(Skill):
    """Destructive: it arms the confirmation gate and waits for a yes."""

    name = "eraser"
    description = "erase the disk"
    patterns = [re.compile(r"\berase the disk\b", re.IGNORECASE)]
    requires_confirmation = True

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("Erased.")


@pytest.mark.asyncio
async def test_a_correction_survives_cancelling_a_pending_confirmation():
    """An ambiguous answer to "are you sure?" cancels the action and re-routes
    the words — and that re-route is still THIS turn, so the interruption it
    followed is still the thing being corrected.
    """
    router, llm = _router(_Confirms())
    router._settings.safety.confirm_destructive = True
    router._settings.security.yolo = False
    await router.route("erase the disk")                 # arms the confirmation
    assert router.awaiting_confirmation
    router.note_interruption("explain how a diode works", "A diode is")
    await router.route("actually, make it shorter")
    assert not router.awaiting_confirmation              # nothing left armed
    assert llm.prompts and "diode" in llm.prompts[-1]
    assert router._interrupted_turn is None
