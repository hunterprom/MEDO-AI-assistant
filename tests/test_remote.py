"""Companion API tests: /ping liveness, /ask routing, and token auth."""

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


def _make_server() -> RemoteServer:
    settings = load_settings()
    registry = SkillRegistry()
    registry.register(DateTimeSkill())
    bus = EventBus()
    router = Router(settings, registry, OllamaClient(settings.llm), bus)
    router.model = None  # no LLM in tests; fast path only
    return RemoteServer(settings, router, StateMachine(bus))


@pytest_asyncio.fixture
async def client() -> AsyncIterator[TestClient]:
    # TestClient connects from 127.0.0.1, so these tests double as proof of
    # the localhost exemption: auth is on by default and nothing sends a token.
    server = _make_server()
    test_client = TestClient(TestServer(server.build_app()))
    await test_client.start_server()
    yield test_client
    await test_client.close()


@pytest_asyncio.fixture
async def lan_client() -> AsyncIterator[TestClient]:
    """A client the server treats as a LAN peer (localhost exemption off)."""
    server = _make_server()
    server._settings.remote.token = "watch-secret"
    server._is_local = lambda request: False  # simulate a non-local peer
    test_client = TestClient(TestServer(server.build_app()))
    await test_client.start_server()
    test_client.medo_server = server  # for per-test config tweaks
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


# --- token auth (LAN clients) ------------------------------------------------


@pytest.mark.asyncio
async def test_localhost_is_exempt_even_with_token_set(client: TestClient):
    # `client` sends no token and still gets 200 everywhere — see fixture note.
    resp = await client.get("/ping")
    assert resp.status == 200


@pytest.mark.asyncio
async def test_lan_valid_bearer_token(lan_client: TestClient):
    resp = await lan_client.get(
        "/ping", headers={"Authorization": "Bearer watch-secret"}
    )
    assert resp.status == 200
    assert (await resp.json())["ok"] is True


@pytest.mark.asyncio
async def test_lan_query_param_token_fallback(lan_client: TestClient):
    resp = await lan_client.get("/ping?token=watch-secret")
    assert resp.status == 200


@pytest.mark.asyncio
async def test_lan_missing_token_is_401(lan_client: TestClient):
    resp = await lan_client.post("/ask", json={"text": "what time is it"})
    assert resp.status == 401
    body = await resp.json()
    assert body["ok"] is False


@pytest.mark.asyncio
async def test_lan_wrong_token_is_401(lan_client: TestClient):
    resp = await lan_client.get(
        "/ping", headers={"Authorization": "Bearer wrong"}
    )
    assert resp.status == 401


@pytest.mark.asyncio
async def test_lan_401_still_carries_cors_headers(lan_client: TestClient):
    # The browser HUD must be able to read the error cross-origin.
    resp = await lan_client.get("/ping")
    assert resp.status == 401
    assert resp.headers.get("Access-Control-Allow-Origin") == "*"


@pytest.mark.asyncio
async def test_lan_no_token_configured_fails_closed(lan_client: TestClient):
    lan_client.medo_server._settings.remote.token = ""
    resp = await lan_client.get("/ping")
    assert resp.status == 401


@pytest.mark.asyncio
async def test_auth_disabled_restores_open_behavior(lan_client: TestClient):
    lan_client.medo_server._settings.remote.auth_enabled = False
    resp = await lan_client.get("/ping")
    assert resp.status == 200


@pytest.mark.asyncio
async def test_options_preflight_never_needs_token(lan_client: TestClient):
    # Browsers strip Authorization from CORS preflights; a 401 here would
    # block every cross-origin call even with a valid token.
    resp = await lan_client.options("/ask")
    assert resp.status == 204


def test_ensure_remote_token_mints_once_and_persists(tmp_path):
    from core.config import apply_local_secrets, ensure_remote_token

    secrets_file = tmp_path / "secrets.local.yaml"
    settings = load_settings()
    token = ensure_remote_token(settings, secrets_file)
    assert token and settings.remote.token == token
    # Second run (fresh settings) loads the same token instead of minting.
    again = load_settings()
    assert ensure_remote_token(again, secrets_file) == token
    # And the normal startup overlay picks it up too.
    overlaid = apply_local_secrets(load_settings(), secrets_file)
    assert overlaid.remote.token == token
