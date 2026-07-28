"""A destructive confirmation is scoped to the channel that armed it.

Bug-audit finding (HIGH): the router's single ``_pending`` confirmation slot was
shared across the voice loop, the companion ``/ask`` endpoint, and routines with
no isolation — so a remote ``/ask`` arriving between a voice "are you sure?" and
the spoken "yes" could resolve (or, with an ambiguous word, silently cancel) a
destructive action the voice user never confirmed. The gate is now source-scoped.
"""

from __future__ import annotations

import re

import pytest

from core.events import EventBus
from core.router import Router
from llm.client import OllamaClient
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult


class _Wipe(Skill):
    """A destructive skill: asks first, only acts once its request is confirmed."""

    name = "wipe"
    description = "delete everything"
    patterns = [re.compile(r"\bdelete everything\b", re.IGNORECASE)]
    requires_confirmation = True

    def __init__(self) -> None:
        self.executed = 0

    async def execute(self, request: SkillRequest) -> SkillResult:
        if not request.context.get("confirmed"):
            return SkillResult("Are you sure?", needs_confirmation=True)
        self.executed += 1
        return SkillResult("Done.")


def _router():
    from core.config import load_settings

    settings = load_settings()
    settings.safety.confirm_destructive = True
    reg = SkillRegistry()
    skill = _Wipe()
    reg.register(skill)
    router = Router(settings, reg, OllamaClient(settings.llm), EventBus())
    router.model = None
    return router, skill


@pytest.mark.asyncio
async def test_same_channel_confirmation_resolves():
    router, skill = _router()
    r1 = await router.route("delete everything", context={"source": "voice"})
    assert r1.speech == "Are you sure?"
    assert router._pending is not None
    r2 = await router.route("yes", context={"source": "voice"})
    assert skill.executed == 1 and r2.skill_name == "wipe"
    assert router._pending is None


@pytest.mark.asyncio
async def test_other_channel_cannot_resolve():
    router, skill = _router()
    await router.route("delete everything", context={"source": "voice"})
    assert router._pending is not None
    # A remote "yes" must NOT execute the voice user's pending action…
    await router.route("yes", context={"source": "remote"})
    assert skill.executed == 0
    assert router._pending is not None            # …and the confirmation still stands
    # …then the voice user's own "yes" still resolves it.
    await router.route("yes", context={"source": "voice"})
    assert skill.executed == 1


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
