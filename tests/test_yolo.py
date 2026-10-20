"""YOLO / dev mode: turn the whole security layer off for development.

When security.yolo is on the policy engine allows everything AND skill-driven
confirmations auto-accept, so a destructive action runs with no prompt. Off by
default — the normal path still confirms.
"""

from __future__ import annotations

import re

import pytest

from core.config import load_settings
from core.events import EventBus
from core.router import Router
from llm.client import OllamaClient
from security.capabilities import Capability
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult


class _Destructive(Skill):
    name = "boom"
    controls_pc = True
    capabilities = frozenset({Capability.WRITE_FILES})
    patterns = [re.compile(r"\bboom\b", re.IGNORECASE)]

    def __init__(self):
        self.ran = False

    async def execute(self, request):
        if not request.context.get("confirmed"):
            return SkillResult("Are you sure? Say yes.", needs_confirmation=True)
        self.ran = True
        return SkillResult("boom done")


def _router(yolo: bool):
    reg = SkillRegistry()
    skill = _Destructive()
    reg.register(skill)
    settings = load_settings()
    settings.security.yolo = yolo
    r = Router(settings, reg, OllamaClient(settings.llm), EventBus())
    r.model = None
    return r, skill


def test_yolo_defaults_off():
    assert load_settings().security.yolo is False


@pytest.mark.asyncio
async def test_normal_mode_still_confirms():
    r, skill = _router(yolo=False)
    res = await r.route("boom", context={"source": "voice"})
    assert skill.ran is False and "sure" in res.speech.lower()
    assert r._pending is not None                     # armed for a yes


@pytest.mark.asyncio
async def test_yolo_runs_destructive_action_without_confirmation():
    r, skill = _router(yolo=True)
    res = await r.route("boom", context={"source": "voice"})
    assert skill.ran is True and "done" in res.speech.lower()
    assert r._pending is None                          # nothing waiting


@pytest.mark.asyncio
async def test_yolo_ignores_pc_control_off():
    r, skill = _router(yolo=True)
    r._settings.safety.pc_control_enabled = False       # would normally block it
    await r.route("boom", context={"source": "voice"})
    assert skill.ran is True                            # yolo overrides the gate


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
