"""Router reply-capture: a skill can ask a question and get the next utterance.

When a skill returns ``await_reply=True``, the router routes the very next
utterance back to it with context['captured_reply'] — UNLESS that utterance is
itself another command (matches a fast-path skill), in which case the command
wins and the capture is dropped. Same-source scoped, like the confirmation gate.
"""

from __future__ import annotations

import re

import pytest

from core.config import load_settings
from core.events import EventBus
from core.router import Router
from llm.client import OllamaClient
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult


class _Asker(Skill):
    """Asks once, then echoes whatever answer it's handed."""

    name = "asker"
    description = "asks for input"
    patterns = [re.compile(r"\bask me\b", re.IGNORECASE)]

    def __init__(self):
        self.answer = None

    async def execute(self, request: SkillRequest) -> SkillResult:
        if request.context.get("captured_reply"):
            self.answer = request.text
            return SkillResult(f"got: {request.text}")
        return SkillResult("what's the answer?", await_reply=True)


class _Clock(Skill):
    name = "clock"
    description = "time"
    patterns = [re.compile(r"\bwhat time is it\b", re.IGNORECASE)]

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("it's noon")


class _Offerer(Skill):
    """Makes a yes/no OFFER ('want the breakdown?'), then echoes captured replies.

    Unlike _Asker (free-text capture), it sets reply_is_offer=True: only a yes or
    a no is an answer, so a fresh command re-routes instead of being swallowed.
    """

    name = "offerer"
    description = "offers a breakdown"
    patterns = [re.compile(r"\bcheck that\b", re.IGNORECASE)]

    def __init__(self):
        self.answer = None

    async def execute(self, request: SkillRequest) -> SkillResult:
        if request.context.get("captured_reply"):
            self.answer = request.text
            return SkillResult("okay")
        return SkillResult("want the breakdown?", await_reply=True,
                           reply_is_offer=True)


def _router(*skills):
    reg = SkillRegistry()
    for s in skills:
        reg.register(s)
    settings = load_settings()
    r = Router(settings, reg, OllamaClient(settings.llm), EventBus())
    r.model = None
    return r


@pytest.mark.asyncio
async def test_next_utterance_is_captured_as_the_answer():
    asker = _Asker()
    router = _router(asker)
    r1 = await router.route("ask me", context={"source": "voice"})
    assert "answer" in r1.speech.lower()
    r2 = await router.route("forty two", context={"source": "voice"})
    assert asker.answer == "forty two" and r2.speech == "got: forty two"


@pytest.mark.asyncio
async def test_a_real_command_wins_over_capture():
    asker, clock = _Asker(), _Clock()
    router = _router(asker, clock)
    await router.route("ask me", context={"source": "voice"})
    # the follow-up is itself a command -> it runs, and the capture is dropped
    r = await router.route("what time is it", context={"source": "voice"})
    assert r.speech == "it's noon" and asker.answer is None


@pytest.mark.asyncio
async def test_capture_is_scoped_to_the_source():
    asker = _Asker()
    router = _router(asker)
    await router.route("ask me", context={"source": "voice"})
    # a different channel's turn must not be swallowed as the answer
    await router.route("hello there", context={"source": "remote"})
    assert asker.answer is None


# --- yes/no OFFER captures (reply_is_offer) ----------------------------------

@pytest.mark.asyncio
async def test_offer_captures_a_yes():
    off = _Offerer()
    router = _router(off)
    await router.route("check that", context={"source": "voice"})
    r = await router.route("yes", context={"source": "voice"})
    assert off.answer == "yes" and r.speech == "okay"


@pytest.mark.asyncio
async def test_offer_captures_a_no_as_a_decline():
    # A 'no' is still an answer to the offer — it reaches the skill (which drops
    # the breakdown), so declines are NOT re-routed to the LLM.
    off = _Offerer()
    router = _router(off)
    await router.route("check that", context={"source": "voice"})
    r = await router.route("no thanks", context={"source": "voice"})
    assert off.answer == "no thanks" and r.speech == "okay"


@pytest.mark.asyncio
async def test_offer_reroutes_a_fresh_request_instead_of_swallowing_it():
    # The bug this fixes: a yes/no OFFER swallowed a brand-new request and
    # answered "okay", silently dropping it (the "I want you to use the
    # electrical engineer for are you sure?" -> bland "Okay." transcript). Now a
    # reply that is neither yes nor no re-routes as a fresh command (here: to the
    # offline LLM), and the offer is NOT captured.
    from core.events import RoutePath

    off = _Offerer()
    router = _router(off)
    await router.route("check that", context={"source": "voice"})
    r = await router.route("actually use the electrical engineer to verify it",
                           context={"source": "voice"})
    assert off.answer is None                     # NOT swallowed
    assert r.path == RoutePath.LLM                 # re-routed as a fresh turn


@pytest.mark.asyncio
async def test_free_text_capture_still_takes_any_answer():
    # A NON-offer capture (self-dev's "what should I fix?") is unchanged: it
    # takes ANY utterance, even a full sentence, because it set no offer flag.
    asker = _Asker()
    router = _router(asker)
    await router.route("ask me", context={"source": "voice"})
    r = await router.route("make the weather skill handle an empty city",
                           context={"source": "voice"})
    assert asker.answer == "make the weather skill handle an empty city"
    assert r.speech.startswith("got:")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
