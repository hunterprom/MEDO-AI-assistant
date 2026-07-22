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
async def server_client() -> AsyncIterator[tuple[TestClient, RemoteServer]]:
    """Client plus the server object, so tests can assert on settings too."""
    server = _make_server()
    test_client = TestClient(TestServer(server.build_app()))
    await test_client.start_server()
    yield test_client, server
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
    assert body["service"] == "medo"


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


# --- watch pairing -------------------------------------------------------


async def _start_pairing(lan_client: TestClient) -> str:
    """Kick off pairing and return the code the PC screen would show."""
    resp = await lan_client.post("/pair/start")
    assert resp.status == 200
    return lan_client.medo_server._pair["code"]


@pytest.mark.asyncio
async def test_pairing_is_reachable_without_a_token(lan_client: TestClient):
    # The whole point: a not-yet-paired watch has no token.
    resp = await lan_client.post("/pair/start")
    assert resp.status == 200
    body = await resp.json()
    assert body["ok"] is True and "token" not in body  # code goes to the PC only


@pytest.mark.asyncio
async def test_pair_confirm_right_code_returns_token(lan_client: TestClient):
    code = await _start_pairing(lan_client)
    resp = await lan_client.post("/pair/confirm", json={"code": code})
    assert resp.status == 200
    assert (await resp.json())["token"] == "watch-secret"
    # single use: the same code can't be redeemed twice
    resp = await lan_client.post("/pair/confirm", json={"code": code})
    assert resp.status == 409


@pytest.mark.asyncio
async def test_pair_confirm_wrong_code_401_then_lockout(lan_client: TestClient):
    code = await _start_pairing(lan_client)
    for _ in range(5):
        resp = await lan_client.post("/pair/confirm", json={"code": "000000"})
        assert resp.status == 401
    resp = await lan_client.post("/pair/confirm", json={"code": code})
    assert resp.status == 429  # brute-forced session is dead, even with the code


@pytest.mark.asyncio
async def test_pair_confirm_expired_code(lan_client: TestClient):
    await _start_pairing(lan_client)
    lan_client.medo_server._pair["expires"] = 0.0  # long past
    resp = await lan_client.post("/pair/confirm", json={"code": "123456"})
    assert resp.status == 410


@pytest.mark.asyncio
async def test_pair_confirm_without_start_is_409(lan_client: TestClient):
    resp = await lan_client.post("/pair/confirm", json={"code": "123456"})
    assert resp.status == 409


def test_discovery_protocol_answers_probe():
    from remote.server import DISCOVERY_PROBE, _DiscoveryProtocol

    sent: list[tuple[bytes, tuple]] = []

    class _FakeTransport:
        def sendto(self, data: bytes, addr) -> None:
            sent.append((data, addr))

    proto = _DiscoveryProtocol("MEDO", 8710)
    proto.connection_made(_FakeTransport())
    proto.datagram_received(b"garbage", ("10.0.0.9", 5000))
    assert sent == []  # ignores anything but the probe
    proto.datagram_received(DISCOVERY_PROBE, ("10.0.0.9", 5000))
    assert len(sent) == 1
    import json

    payload = json.loads(sent[0][0])
    assert payload == {"service": "medo", "name": "MEDO", "port": 8710}
    assert sent[0][1] == ("10.0.0.9", 5000)


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


# --- HUD controls for the council and Lion Mode -------------------------------


@pytest.mark.asyncio
async def test_council_endpoint_lists_the_roster(server_client):
    client, server = server_client
    data = await (await client.get("/council")).json()
    assert data["ok"] and data["enabled"] is True
    keys = {m["key"] for m in data["members"]}
    assert {"electrical", "robotics", "law", "finance"} <= keys
    assert all(m["on"] for m in data["members"]), "all on by default"


@pytest.mark.asyncio
async def test_switching_one_specialist_off_and_back_on(server_client):
    client, server = server_client
    data = await (await client.post("/council",
                                    json={"agent": "law", "on": False})).json()
    law = next(m for m in data["members"] if m["key"] == "law")
    assert law["on"] is False
    assert "law" in [d.lower() for d in server._settings.council.disabled]

    data = await (await client.post("/council",
                                    json={"agent": "law", "on": True})).json()
    assert next(m for m in data["members"] if m["key"] == "law")["on"] is True


@pytest.mark.asyncio
async def test_an_unknown_specialist_is_rejected(server_client):
    client, _ = server_client
    resp = await client.post("/council", json={"agent": "astrologer", "on": False})
    assert resp.status == 404


@pytest.mark.asyncio
async def test_the_whole_council_can_be_switched_off(server_client):
    client, server = server_client
    data = await (await client.post("/council", json={"enabled": False})).json()
    assert data["enabled"] is False
    assert server._settings.council.enabled is False


@pytest.mark.asyncio
async def test_lion_mode_toggles_and_is_reported_in_status(server_client):
    client, server = server_client
    assert (await (await client.get("/status")).json())["lion_mode"] is False

    data = await (await client.post("/control/lion", json={"on": True})).json()
    assert data["on"] is True and server._settings.safety.lion_mode is True
    assert (await (await client.get("/status")).json())["lion_mode"] is True

    await client.post("/control/lion", json={"on": False})
    assert server._settings.safety.lion_mode is False


@pytest.mark.asyncio
async def test_lion_mode_rejects_a_malformed_body(server_client):
    client, _ = server_client
    assert (await client.post("/control/lion", data="not json")).status == 400
