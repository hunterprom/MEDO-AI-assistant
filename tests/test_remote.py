"""Companion API tests: liveness, routing, and bearer-token auth over real HTTP."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from core.config import load_settings
from core.events import EventBus, StateMachine
from core.router import Router
from llm.client import OllamaClient
from remote.server import MAX_TEXT_CHARS, RemoteServer
from skills.base import SkillRegistry
from skills.datetime_skill import DateTimeSkill

AUTH_TOKEN = "unit-test-token"


def _build_server(*, auth_enabled: bool = True, token: str = "") -> RemoteServer:
    settings = load_settings()
    settings.remote.auth_enabled = auth_enabled
    settings.remote.token = token
    registry = SkillRegistry()
    registry.register(DateTimeSkill())
    bus = EventBus()
    router = Router(settings, registry, OllamaClient(settings.llm), bus)
    router.model = None  # no LLM in tests; fast path only
    return RemoteServer(settings, router, StateMachine(bus))


async def _start(server: RemoteServer) -> TestClient:
    test_client = TestClient(TestServer(server.build_app()))
    await test_client.start_server()
    return test_client


@pytest_asyncio.fixture
async def client() -> AsyncIterator[TestClient]:
    # Auth is on with a token set, but TestClient connects from 127.0.0.1 —
    # so this fixture also exercises the localhost exemption on every test.
    test_client = await _start(_build_server(token=AUTH_TOKEN))
    yield test_client
    await test_client.close()


@pytest_asyncio.fixture
async def lan_client() -> AsyncIterator[TestClient]:
    """A client whose requests look like they come from another LAN device."""
    server = _build_server(token=AUTH_TOKEN)
    server._peer_is_local = lambda request: False  # not 127.0.0.1
    test_client = await _start(server)
    yield test_client
    await test_client.close()


@pytest.mark.asyncio
async def test_ping(client: TestClient):
    resp = await client.get("/ping")
    assert resp.status == 200
    body = await resp.json()
    assert body["ok"] is True
    assert body["service"] == "jarvis-v2"


@pytest.mark.asyncio
async def test_ask_routes_fast_path(client: TestClient):
    resp = await client.post("/ask", json={"text": "what time is it"})
    assert resp.status == 200
    body = await resp.json()
    assert body["path"] == "FAST"
    assert body["skill"] == "datetime"
    assert "It's" in body["speech"]
    assert body["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_ask_rejects_empty_text(client: TestClient):
    resp = await client.post("/ask", json={"text": "   "})
    assert resp.status == 400


@pytest.mark.asyncio
async def test_ask_rejects_non_json(client: TestClient):
    resp = await client.post("/ask", data=b"not json")
    assert resp.status == 400


@pytest.mark.asyncio
async def test_ask_rejects_oversize_text(client: TestClient):
    resp = await client.post("/ask", json={"text": "x" * (MAX_TEXT_CHARS + 1)})
    assert resp.status == 413


# --- bearer-token auth (remote.auth_enabled) ---


@pytest.mark.asyncio
async def test_auth_valid_token_accepted(lan_client: TestClient):
    resp = await lan_client.post(
        "/ask",
        json={"text": "what time is it"},
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
    )
    assert resp.status == 200
    assert (await resp.json())["path"] == "FAST"


@pytest.mark.asyncio
async def test_auth_query_token_fallback(lan_client: TestClient):
    # For EventSource/MJPEG-style clients that can't set request headers.
    resp = await lan_client.get(f"/ping?token={AUTH_TOKEN}")
    assert resp.status == 200


@pytest.mark.asyncio
async def test_auth_missing_token_rejected(lan_client: TestClient):
    resp = await lan_client.get("/ping")
    assert resp.status == 401
    body = await resp.json()
    assert body["ok"] is False and "token" in body["error"]


@pytest.mark.asyncio
async def test_auth_wrong_token_rejected(lan_client: TestClient):
    resp = await lan_client.get(
        "/ping", headers={"Authorization": "Bearer not-the-token"}
    )
    assert resp.status == 401


@pytest.mark.asyncio
async def test_auth_preflight_needs_no_token(lan_client: TestClient):
    # Browsers never attach Authorization to a CORS preflight.
    resp = await lan_client.options("/ask")
    assert resp.status == 204


@pytest.mark.asyncio
async def test_auth_localhost_exempt(client: TestClient):
    # The `client` fixture has auth on + a token set, but connects from
    # 127.0.0.1 — no token needed (zero-config HUD/sidecar startup).
    resp = await client.get("/ping")
    assert resp.status == 200


@pytest.mark.asyncio
async def test_auth_disabled_restores_open_api():
    server = _build_server(auth_enabled=False)
    server._peer_is_local = lambda request: False
    test_client = await _start(server)
    try:
        resp = await test_client.get("/ping")
        assert resp.status == 200
    finally:
        await test_client.close()


@pytest.mark.asyncio
async def test_auth_fails_closed_without_provisioned_token():
    # Auth on but no token generated yet -> LAN requests are refused, not let in.
    server = _build_server(token="")
    server._peer_is_local = lambda request: False
    test_client = await _start(server)
    try:
        resp = await test_client.get("/ping?token=anything")
        assert resp.status == 401
    finally:
        await test_client.close()
