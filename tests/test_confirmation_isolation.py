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
from core.router import PC_CONTROL_OFF_REPLY, Router
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


class _PcWipe(_Wipe):
    """Destructive AND PC-controlling: asks first, and touches the machine."""

    name = "pc_wipe"
    description = "wipe the disk"
    patterns = [re.compile(r"\bwipe the disk\b", re.IGNORECASE)]
    controls_pc = True


class _Clock(Skill):
    """A harmless query skill — the fresh command an ambiguous reply becomes."""

    name = "clock"
    description = "tell the time"
    patterns = [re.compile(r"what time is it", re.IGNORECASE)]

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("It's 3 PM.")


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


@pytest.mark.asyncio
async def test_confirmed_action_re_checks_pc_control_gate():
    # Arm a destructive PC-controlling action while PC control is ON…
    from core.config import load_settings

    settings = load_settings()
    settings.safety.confirm_destructive = True
    settings.safety.pc_control_enabled = True
    reg = SkillRegistry()
    skill = _PcWipe()
    reg.register(skill)
    router = Router(settings, reg, OllamaClient(settings.llm), EventBus())
    router.model = None

    r1 = await router.route("wipe the disk", context={"source": "voice"})
    assert r1.speech == "Are you sure?"
    assert router._pending is not None
    # …then flip the HUD PC CONTROL switch OFF before saying "yes". The confirmed
    # path must re-apply the gate the fast/LLM paths enforce, not run the action.
    settings.safety.pc_control_enabled = False
    r2 = await router.route("yes", context={"source": "voice"})
    assert skill.executed == 0                     # the action must NOT run…
    assert r2.speech == PC_CONTROL_OFF_REPLY       # …and MEDO says why
    assert router._pending is None                 # …and the confirmation clears


@pytest.mark.asyncio
async def test_ambiguous_reply_that_is_a_command_is_recorded():
    # A reply that is neither yes nor no cancels the pending wipe and is re-routed
    # as a genuine command — which must be recorded as a normal conversation turn,
    # so a later follow-up ("and tomorrow?") can find it.
    from core.config import load_settings

    settings = load_settings()
    settings.safety.confirm_destructive = True
    reg = SkillRegistry()
    wipe, clock = _Wipe(), _Clock()
    reg.register(wipe)
    reg.register(clock)
    router = Router(settings, reg, OllamaClient(settings.llm), EventBus())
    router.model = None

    await router.route("delete everything", context={"source": "voice"})
    assert router._pending is not None
    r = await router.route("what time is it", context={"source": "voice"})
    assert wipe.executed == 0 and r.speech == "It's 3 PM."   # re-routed, not wiped
    # The re-routed command IS the latest turn — not the suppressed "delete
    # everything"/"Are you sure?" turn it followed.
    assert router.conversation.last_question() == "what time is it"
    assert router.conversation.last_reply() == "It's 3 PM."


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
