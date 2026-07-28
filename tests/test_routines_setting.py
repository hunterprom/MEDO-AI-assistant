"""Proactive scheduled routines edited from the HUD: 'at 08:00, ask these things'.

Routines are validated (HH:MM time, at least one thing to ask), persisted per
machine to the git-ignored overrides, and re-applied at startup. The scheduler is
built at boot, so a change takes a restart — the endpoint reports that.
"""

from __future__ import annotations

import pytest

from core.config import (
    apply_local_secrets,
    load_settings,
    resolve_routines,
    save_routines,
)


# --- validation --------------------------------------------------------------

def test_resolve_normalizes_time_and_days():
    items = resolve_routines([
        {"name": "brief", "at": "8:05", "ask": ["the weather"], "days": ["Monday"]}])
    assert items[0].at == "08:05"
    assert items[0].days == ["mon"]
    assert items[0].ask == ["the weather"]


def test_resolve_rejects_bad_time_and_empty_ask():
    with pytest.raises(ValueError):
        resolve_routines([{"name": "x", "at": "99:99", "ask": ["hi"]}])
    with pytest.raises(ValueError):
        resolve_routines([{"name": "x", "at": "25:00", "ask": ["hi"]}])
    with pytest.raises(ValueError):
        resolve_routines([{"name": "x", "at": "08:00", "ask": []}])


# --- persistence roundtrip ---------------------------------------------------

def test_save_and_apply_roundtrip(tmp_path):
    path = tmp_path / "secrets.local.yaml"
    save_routines([{"name": "brief", "at": "08:00",
                    "ask": ["the weather", "the news"]}], path=path)
    s = apply_local_secrets(load_settings(), path)
    r = next(x for x in s.routines if x.name == "brief")
    assert r.at == "08:00" and r.ask == ["the weather", "the news"]


# --- the companion endpoints -------------------------------------------------

@pytest.mark.asyncio
async def test_routines_status_and_set_endpoints():
    from aiohttp.test_utils import TestClient, TestServer

    from core.events import EventBus, StateMachine
    from core.router import Router
    from llm.client import OllamaClient
    from remote.server import RemoteServer
    from skills.base import SkillRegistry

    settings = load_settings()
    router = Router(settings, SkillRegistry(), OllamaClient(settings.llm), EventBus())
    router.model = None
    server = RemoteServer(settings, router, StateMachine(EventBus()))   # persist off
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    try:
        resp = await client.post("/control/routines", json={"routines": [
            {"name": "brief", "at": "08:00", "ask": ["the weather"]}]})
        assert resp.status == 200
        assert any(r.name == "brief" for r in settings.routines)

        got = await (await client.get("/routines")).json()
        assert any(r["name"] == "brief" and r["at"] == "08:00"
                   for r in got["routines"])

        # a bad time is a 400, not a 500
        bad = await client.post("/control/routines", json={"routines": [
            {"name": "x", "at": "nope", "ask": ["hi"]}]})
        assert bad.status == 400
    finally:
        await client.close()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
