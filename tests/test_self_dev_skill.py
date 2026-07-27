"""The voice/background skill over the self-programming engine.

A fake engine (no real agent, no git) and a fake announcer let us verify the
control flow: a request starts a BACKGROUND proposal and reports it when ready,
apply goes through the yes/no confirmation gate, discard drops it, a failed
proposal is cleaned up + announced, and the trigger needs a self-marker so a
normal "fix the weather skill" isn't hijacked into self-editing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.self_dev import Proposal
from skills.base import SkillRequest
from skills.self_dev_skill import _START, SelfDevSkill


class _FakeEngine:
    def __init__(self, proposal=None, raise_exc=None):
        self._proposal = proposal
        self._raise = raise_exc
        self.applied: list = []
        self.discarded: list = []

    async def propose(self, request):
        if self._raise is not None:
            raise self._raise
        return self._proposal

    async def apply(self, proposal):
        self.applied.append(proposal)
        return True

    async def discard(self, proposal):
        self.discarded.append(proposal)


class _FakeAnnouncer:
    def __init__(self):
        self.messages: list[str] = []

    async def __call__(self, text: str) -> None:
        self.messages.append(text)


def _proposal(ok: bool = True) -> Proposal:
    return Proposal(
        request="x", branch="medo/self-dev/x-abcd1234",
        worktree=Path("/tmp/x/tree"), base="0" * 40,
        diff="+ADDED = True" if ok else "",
        files_changed=["skills/x.py"] if ok else [],
        tests_ok=ok, lint_ok=ok)


# --- the trigger --------------------------------------------------------------

def test_self_dev_surfaces_only_in_lion_mode():
    from core.config import load_settings

    settings = load_settings()
    skill = SelfDevSkill(_FakeEngine(proposal=_proposal(ok=True)), None, settings)
    settings.mode.lion = False
    assert skill.match("fix your code so the timer works") is None
    settings.mode.lion = True
    assert skill.match("fix your code so the timer works") is not None


def test_trigger_requires_a_self_marker():
    assert _START.search("fix your code so the timer works")
    assert _START.search("program yourself to send emails")
    assert _START.search("add a feature to yourself")
    # ...but a normal request must NOT be hijacked into self-editing:
    assert not _START.search("fix the weather skill")
    assert not _START.search("add a note about memory")


# --- start / announce ---------------------------------------------------------

@pytest.mark.asyncio
async def test_start_runs_in_background_and_announces_a_ready_proposal():
    ann = _FakeAnnouncer()
    skill = SelfDevSkill(_FakeEngine(proposal=_proposal(ok=True)), ann)

    result = await skill.execute(SkillRequest(text="fix your code so X works"))
    assert result.success and "background" in result.speech.lower()

    await skill._task                       # let the background proposal finish
    assert skill._pending is not None
    assert any("proposal" in m.lower() for m in ann.messages)


@pytest.mark.asyncio
async def test_bare_trigger_asks_what_to_do():
    skill = SelfDevSkill(_FakeEngine(), None)
    result = await skill.execute(SkillRequest(text="program yourself"))
    assert result.success is False and "tell me" in result.speech.lower()


@pytest.mark.asyncio
async def test_failed_proposal_is_discarded_and_reported():
    ann = _FakeAnnouncer()
    engine = _FakeEngine(proposal=_proposal(ok=False))
    skill = SelfDevSkill(engine, ann)

    await skill.execute(SkillRequest(text="fix your code so the alarm works"))
    await skill._task

    assert skill._pending is None
    assert engine.discarded                 # the not-mergeable proposal was cleaned up
    assert any("set it aside" in m.lower() for m in ann.messages)


# --- apply / discard ----------------------------------------------------------

@pytest.mark.asyncio
async def test_apply_needs_confirmation_then_merges():
    engine = _FakeEngine(proposal=_proposal(ok=True))
    skill = SelfDevSkill(engine, None)
    await skill.execute(SkillRequest(text="fix your code so the light turns on"))
    await skill._task

    prompt = await skill.execute(SkillRequest(text="apply your change"))
    assert prompt.needs_confirmation and not engine.applied

    done = await skill.execute(SkillRequest(text="yes", context={"confirmed": True}))
    assert engine.applied and skill._pending is None
    assert "merged" in done.speech.lower()


@pytest.mark.asyncio
async def test_apply_with_nothing_pending():
    result = await SelfDevSkill(_FakeEngine(), None).execute(
        SkillRequest(text="apply your change"))
    assert result.success is False and "don't have" in result.speech.lower()


@pytest.mark.asyncio
async def test_discard_drops_the_pending_proposal():
    engine = _FakeEngine(proposal=_proposal(ok=True))
    skill = SelfDevSkill(engine, None)
    await skill.execute(SkillRequest(text="work on your code to add Q"))
    await skill._task

    result = await skill.execute(SkillRequest(text="discard your change"))
    assert engine.discarded and skill._pending is None
    assert "discarded" in result.speech.lower()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
