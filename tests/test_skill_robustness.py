"""Skills must not crash the turn on model-supplied args, and a skill that does
raise must degrade to a spoken error — never take down the text REPL / /ask.

A bug-audit pass found the LLM tool path could feed a skill a string where it
expected a number ("4.7k" for a "number" ohms arg), and that the router's three
dispatch sites didn't wrap ``skill.execute`` — so an unguarded skill exception
propagated out of ``route()`` (a silent turn in voice, a HARD crash in the text
REPL and the companion ``/ask`` endpoint).
"""

from __future__ import annotations

import pytest

from core.events import EventBus
from core.router import Router
from llm.client import OllamaClient
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult
from skills.resistor import ResistorSkill


@pytest.mark.asyncio
async def test_resistor_handles_string_ohms_from_the_model():
    """A model sends ohms="4.7k" (a string) for a "number" arg — no crash."""
    skill = ResistorSkill()
    result = await skill.execute(SkillRequest(text="", args={"ohms": "4.7k"}))
    assert isinstance(result, SkillResult)
    assert result.success                       # parsed 4.7k, produced the bands
    assert result.data.get("ohms") == 4700.0


class _Boom(Skill):
    name = "boom"
    description = "always raises"
    patterns = []

    async def execute(self, request: SkillRequest) -> SkillResult:
        raise RuntimeError("kaboom")


@pytest.mark.asyncio
async def test_safe_execute_turns_a_crash_into_a_spoken_error():
    from core.config import load_settings

    settings = load_settings()
    router = Router(settings, SkillRegistry(), OllamaClient(settings.llm), EventBus())
    out = await router._safe_execute(_Boom(), SkillRequest(text="hi"))
    assert isinstance(out, SkillResult)
    assert out.success is False                  # a graceful error, not a raise
    assert out.speech                            # something is spoken


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
