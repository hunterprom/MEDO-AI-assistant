"""The HUD DEVELOPER MODE switch (security.yolo) — toggle + session-only.

User-authorized dev bypass, exposed as a HUD toggle (mirrors PC control). The
guarantees pinned here: the toggle flips the live flag (policy engine then allows
what it would otherwise deny), /status reports it, and it is SESSION-ONLY — never
persisted, so a restart returns to secure.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from core.config import load_settings
from core.events import EventBus, StateMachine
from core.router import Router
from llm.client import OllamaClient
from remote.server import RemoteServer
from security.capabilities import Capability
from security.policy import ActionRequest, Actor, PolicyEngine
from skills.base import SkillRegistry


def _make_server() -> RemoteServer:
    settings = load_settings()
    bus = EventBus()
    router = Router(settings, SkillRegistry(), OllamaClient(settings.llm), bus)
    router.model = None
    return RemoteServer(settings, router, StateMachine(bus))


@pytest_asyncio.fixture
async def server_client() -> AsyncIterator[tuple[TestClient, RemoteServer]]:
    server = _make_server()
    tc = TestClient(TestServer(server.build_app()))
    await tc.start_server()
    yield tc, server
    await tc.close()


@pytest.mark.asyncio
async def test_toggle_flips_the_live_flag(server_client):
    tc, server = server_client
    assert server._settings.security.yolo is False          # secure by default
    r = await tc.post("/control/dev-mode", json={"on": True})
    assert r.status == 200 and (await r.json())["on"] is True
    assert server._settings.security.yolo is True            # live

    # the engine now allows what it would otherwise DENY (pc off + owner-voice on)
    server._settings.safety.pc_control_enabled = False
    server._settings.security.owner_voice = True
    engine = PolicyEngine(server._settings.security, server._settings.safety)
    d = engine.check(ActionRequest(actor=Actor.LOCAL_USER,
                                   capability=Capability.POWER_CONTROL))
    assert d.allowed() and d.code == "yolo"

    off = await tc.post("/control/dev-mode", json={"on": False})
    assert (await off.json())["on"] is False
    assert server._settings.security.yolo is False


@pytest.mark.asyncio
async def test_status_reports_dev_mode(server_client):
    tc, server = server_client
    d = await (await tc.get("/status")).json()
    assert d["dev_mode"] is False
    await tc.post("/control/dev-mode", json={"on": True})
    d2 = await (await tc.get("/status")).json()
    assert d2["dev_mode"] is True


@pytest.mark.asyncio
async def test_bad_body_is_rejected(server_client):
    tc, _server = server_client
    r = await tc.post("/control/dev-mode", json={"nope": 1})
    assert r.status == 400


@pytest.mark.asyncio
async def test_dev_mode_is_never_persisted(server_client, monkeypatch):
    # SESSION-ONLY: the handler must not write it to secrets.local.yaml, so a
    # restart returns to secure. Assert no persistence helper is invoked.
    import core.config as cfg
    called = {"n": 0}
    monkeypatch.setattr(cfg, "_write_local", lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    tc, _server = server_client
    await tc.post("/control/dev-mode", json={"on": True})
    assert called["n"] == 0                                  # nothing written to disk


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
