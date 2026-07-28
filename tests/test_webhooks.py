"""Outbound event webhooks: POST a JSON payload to a URL when MEDO does things.

An HUD-added webhook may carry an optional bearer token, so — like MCP servers —
it's persisted to the git-ignored overrides (never config.yaml) and re-applied at
startup. The WebhookManager maps four lifecycle events onto EventBus types and
fires each matching hook fire-and-forget so a dead endpoint never stalls a turn.
"""

from __future__ import annotations

import pytest

from core.config import (
    WebhookConfig,
    apply_local_secrets,
    load_settings,
    remove_webhook,
    save_webhook,
)
from core.events import AssistantState, Event, EventType, RoutePath
from core.router import RouteResult
from core.webhooks import WebhookManager


# --- config field + persistence ---------------------------------------------

def test_webhook_config_defaults():
    h = WebhookConfig(name="n8n", url="http://x/y", auth_token="sek")
    assert h.event == "routed" and h.enabled is True and h.auth_token == "sek"


def test_save_requires_name_url_and_valid_event(tmp_path):
    path = tmp_path / "secrets.local.yaml"
    with pytest.raises(ValueError):
        save_webhook("", event="routed", url="http://x/y", path=path)     # no name
    with pytest.raises(ValueError):
        save_webhook("a", event="routed", url="", path=path)              # no url
    with pytest.raises(ValueError):
        save_webhook("a", event="routed", url="ftp://x", path=path)       # bad scheme
    with pytest.raises(ValueError):
        save_webhook("a", event="nope", url="http://x/y", path=path)      # bad event


def test_save_and_apply_roundtrip(tmp_path):
    path = tmp_path / "secrets.local.yaml"
    summary = save_webhook("logger", event="reply", url="https://x/y",
                           auth_token="tok-9", path=path)
    assert summary == {"name": "logger", "event": "reply", "url": "https://x/y",
                       "has_auth": True}
    s = apply_local_secrets(load_settings(), path)
    hook = next(h for h in s.webhooks if h.name == "logger")
    assert hook.event == "reply" and hook.url == "https://x/y"
    assert hook.auth_token == "tok-9"                       # secret, applied at boot


def test_remove(tmp_path):
    path = tmp_path / "secrets.local.yaml"
    save_webhook("x", event="wake", url="http://x/y", path=path)
    assert remove_webhook("x", path=path) is True
    assert remove_webhook("x", path=path) is False          # already gone


# --- the manager maps events -> payloads -------------------------------------

def _mgr(event="routed"):
    return WebhookManager([WebhookConfig(name="h", event=event, url="http://x/y")])


def test_wake_only_fires_on_listening():
    m = _mgr("wake")
    into = Event(EventType.STATE_CHANGED, (AssistantState.IDLE, AssistantState.LISTENING))
    out = Event(EventType.STATE_CHANGED, (AssistantState.LISTENING, AssistantState.IDLE))
    assert m._payload(into) == {"event": "wake", "state": "LISTENING",
                                "at": m._payload(into)["at"]}
    assert m._payload(out) is None                          # leaving listening: no fire


def test_transcript_and_reply_payloads():
    m = _mgr()
    tp = m._payload(Event(EventType.TRANSCRIPT, "what time is it"))
    assert tp["event"] == "transcript" and tp["text"] == "what time is it"
    rp = m._payload(Event(EventType.RESPONSE, "it is noon"))
    assert rp["event"] == "reply" and rp["text"] == "it is noon"


def test_routed_payload_from_routeresult():
    m = _mgr()
    rr = RouteResult(path=RoutePath.FAST, speech="tick", skill_name="clock")
    p = m._payload(Event(EventType.ROUTED, rr))
    assert p == {"event": "routed", "skill": "clock", "path": "FAST",
                 "speech": "tick", "at": p["at"]}
    # A non-RouteResult ROUTED payload must not fire.
    assert m._payload(Event(EventType.ROUTED, "junk")) is None


def test_manager_filters_disabled_and_unknown():
    m = WebhookManager([
        WebhookConfig(name="ok", event="routed", url="http://x/y"),
        WebhookConfig(name="off", event="routed", url="http://x/y", enabled=False),
        WebhookConfig(name="nourl", event="routed", url=""),
        WebhookConfig(name="badevent", event="bogus", url="http://x/y"),
    ])
    assert [h.name for h in m._hooks] == ["ok"]
    assert m.active is True


# --- the companion endpoints -------------------------------------------------

@pytest.mark.asyncio
async def test_webhook_add_status_and_remove_endpoints():
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
        resp = await client.post("/control/webhook", json={
            "name": "n8n", "event": "reply", "url": "http://localhost:5678/w",
            "auth": "tok"})
        assert resp.status == 200
        hook = next(h for h in settings.webhooks if h.name == "n8n")
        assert hook.event == "reply" and hook.auth_token == "tok"

        # GET status never leaks the token, only whether one is set.
        got = await (await client.get("/webhooks")).json()
        row = next(w for w in got["webhooks"] if w["name"] == "n8n")
        assert row["has_auth"] is True and "auth_token" not in row

        # bad URL scheme -> 400
        bad = await client.post("/control/webhook", json={
            "name": "b", "event": "reply", "url": "ftp://x"})
        assert bad.status == 400

        # remove it
        resp = await client.post("/control/webhook/remove", json={"name": "n8n"})
        assert resp.status == 200
        assert not any(h.name == "n8n" for h in settings.webhooks)
    finally:
        await client.close()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
