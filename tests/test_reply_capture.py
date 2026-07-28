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


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
