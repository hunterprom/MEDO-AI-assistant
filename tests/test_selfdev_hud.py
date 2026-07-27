"""The companion-API surface behind the HUD's self-dev panel.

Reads the pending proposal (diff + gate) and applies/discards it via a real
SelfDevSkill (with a fake engine) registered in the router's registry, so the
GET /selfdev + POST /control/selfdev/{apply,discard} endpoints are exercised
end-to-end without a real coding agent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import load_settings
from core.events import EventBus, StateMachine
from core.router import Router
from core.self_dev import Proposal
from llm.client import OllamaClient
from remote.server import RemoteServer
from skills.base import SkillRegistry
from skills.self_dev_skill import SelfDevSkill


class _FakeEngine:
    def __init__(self):
        self.applied: list = []
        self.discarded: list = []

    async def apply(self, proposal):
        self.applied.append(proposal)
        return True

    async def discard(self, proposal):
        self.discarded.append(proposal)


def _proposal(ok=True):
    return Proposal(
        request="fix the timer", branch="medo/self-dev/fix-1",
        worktree=Path("/tmp/x/tree"), base="0" * 40,
        diff="+ADDED = True", files_changed=["skills/timers.py"],
        tests_ok=ok, lint_ok=ok)


def _server(with_skill=True, pending=None):
    settings = load_settings()
    reg = SkillRegistry()
    engine = _FakeEngine()
    if with_skill:
        skill = SelfDevSkill(engine)
        skill._pending = pending
        reg.register(skill)
    router = Router(settings, reg, OllamaClient(settings.llm), EventBus())
    router.model = None
    return RemoteServer(settings, router, StateMachine(EventBus())), engine


async def _client(server):
    from aiohttp.test_utils import TestClient, TestServer
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_status_when_self_dev_is_off():
    server, _ = _server(with_skill=False)
    client = await _client(server)
    try:
        d = await (await client.get("/selfdev")).json()
        assert d["enabled"] is False and d["proposal"] is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_status_reports_the_pending_proposal():
    server, _ = _server(pending=_proposal(ok=True))
    client = await _client(server)
    try:
        d = await (await client.get("/selfdev")).json()
        assert d["enabled"] is True
        p = d["proposal"]
        assert p["ok"] is True and p["files"] == ["skills/timers.py"]
        assert "+ADDED = True" in p["diff"] and p["request"] == "fix the timer"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_apply_endpoint_merges_and_clears():
    server, engine = _server(pending=_proposal(ok=True))
    client = await _client(server)
    try:
        r = await (await client.post("/control/selfdev/apply")).json()
        assert r["ok"] is True and engine.applied
        # proposal is gone afterwards
        d = await (await client.get("/selfdev")).json()
        assert d["proposal"] is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_discard_endpoint():
    server, engine = _server(pending=_proposal(ok=True))
    client = await _client(server)
    try:
        r = await (await client.post("/control/selfdev/discard")).json()
        assert r["ok"] is True and engine.discarded
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_apply_when_off_is_409():
    server, _ = _server(with_skill=False)
    client = await _client(server)
    try:
        assert (await client.post("/control/selfdev/apply")).status == 409
    finally:
        await client.close()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
