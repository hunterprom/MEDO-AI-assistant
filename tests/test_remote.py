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
from remote.server import MAX_TEXT_CHARS, PAIR_MAX_PENDING, RemoteServer
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
async def test_lan_401_echoes_an_allowed_origin_not_wildcard(lan_client: TestClient):
    # The browser HUD (a loopback origin) can still READ the error cross-origin,
    # but only its OWN origin is echoed — never a wildcard.
    resp = await lan_client.get("/ping", headers={"Origin": "http://127.0.0.1:8730"})
    assert resp.status == 401
    assert resp.headers.get("Access-Control-Allow-Origin") == "http://127.0.0.1:8730"
    assert "Access-Control-Allow-Headers" in resp.headers          # static CORS present
    # A non-browser client (no Origin) simply gets no ACAO, which is fine.
    bare = await lan_client.get("/ping")
    assert bare.status == 401
    assert bare.headers.get("Access-Control-Allow-Origin") is None


# --- CSRF / DNS-rebinding defense (the loopback drive-by) --------------------


@pytest.mark.asyncio
async def test_cross_origin_browser_request_is_refused(client: TestClient):
    # A page on the open web POSTing to 127.0.0.1 sends its own Origin; even
    # though the peer is loopback (auth-exempt), the request is refused BEFORE it
    # can act, and no ACAO is echoed so the page can't read the response either.
    resp = await client.post("/ask", json={"text": "what time is it"},
                             headers={"Origin": "https://evil.example"})
    assert resp.status == 403
    assert resp.headers.get("Access-Control-Allow-Origin") is None


@pytest.mark.asyncio
async def test_same_site_origin_and_no_origin_still_work(client: TestClient):
    # The HUD (a loopback origin) and non-browser clients (no Origin) are allowed.
    hud = await client.post("/ask", json={"text": "what time is it"},
                            headers={"Origin": "http://127.0.0.1:8730"})
    assert hud.status == 200
    assert hud.headers.get("Access-Control-Allow-Origin") == "http://127.0.0.1:8730"
    bare = await client.post("/ask", json={"text": "what time is it"})
    assert bare.status == 200


@pytest.mark.asyncio
async def test_allowed_origins_whitelist(server_client):
    tc, server = server_client
    server._settings.remote.allowed_origins = ["https://medo.example"]
    resp = await tc.post("/ask", json={"text": "what time is it"},
                         headers={"Origin": "https://medo.example"})
    assert resp.status == 200
    assert resp.headers.get("Access-Control-Allow-Origin") == "https://medo.example"


@pytest.mark.asyncio
async def test_no_origin_domain_host_is_refused_dns_rebinding(client: TestClient):
    # A same-origin GET (browsers omit Origin) whose Host is an attacker DOMAIN is
    # a DNS-rebinding READ attempt — refuse it even from a loopback, auth-exempt
    # peer, so /dirs, /facts, /search/files can't leak.
    resp = await client.get("/dirs", headers={"Host": "evil.example"})
    assert resp.status == 403
    # ...but a loopback/IP Host with no Origin is a normal client → allowed.
    ok = await client.get("/ping", headers={"Host": "127.0.0.1"})
    assert ok.status == 200


@pytest.mark.asyncio
async def test_whitelisted_hostname_host_is_allowed(server_client):
    # Listing a remote-HUD origin also whitelists its hostname for the Host check,
    # so an mDNS/NetBIOS HUD works once its origin is in remote.allowed_origins.
    tc, server = server_client
    server._settings.remote.allowed_origins = ["http://medo.local:8730"]
    resp = await tc.get("/ping", headers={"Host": "medo.local"})
    assert resp.status == 200


@pytest.mark.asyncio
async def test_provider_base_url_change_requires_api_key(server_client):
    # Redirecting a cloud provider to a NEW base URL without re-supplying the key
    # must be refused, so the stored key is never sent to an attacker URL.
    tc, server = server_client
    server._settings.llm.provider = "openai"
    server._settings.llm.api_key = "sk-secret"
    server._settings.llm.openai_base_url = "https://api.openai.com/v1"
    resp = await tc.post("/provider", json={"provider": "openai",
                                            "base_url": "https://evil.example/v1"})
    assert resp.status == 400
    assert server._settings.llm.openai_base_url == "https://api.openai.com/v1"   # unchanged


@pytest.mark.asyncio
async def test_open_reveals_a_file_never_executes_it(server_client, tmp_path, monkeypatch):
    # /open on a caller-supplied FILE must reveal its folder, never run the file.
    import core.dirs as d

    f = tmp_path / "payload.exe"
    f.write_text("x", encoding="utf-8")
    monkeypatch.setattr(d, "resolve_path", lambda s: f)
    opened: dict = {}
    monkeypatch.setattr(d, "open_path", lambda p: (opened.setdefault("path", p), True)[1])
    tc, _server = server_client
    resp = await tc.post("/open", json={"path": str(f)})
    assert resp.status == 200
    assert opened["path"] == f.parent          # the folder, NOT the executable


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


# --- approve-on-PC pairing (phone + watch, no code typing) ---------------


@pytest.mark.asyncio
async def test_pair_request_approve_delivers_token(server_client):
    client, server = server_client
    server._settings.remote.token = "the-token"
    # 1) the device asks to connect
    resp = await client.post("/pair/request", json={"name": "Pixel", "kind": "phone"})
    assert resp.status == 200
    rid = (await resp.json())["request_id"]
    # 2) it polls: still pending, no token yet
    body = await (await client.get(f"/pair/poll?request_id={rid}")).json()
    assert body["status"] == "pending" and "token" not in body
    # 3) it shows up in the HUD's pending list (with its name + peer)
    pend = (await (await client.get("/pair/pending")).json())["pending"]
    assert any(p["request_id"] == rid and p["name"] == "Pixel"
               and p["kind"] == "phone" for p in pend)
    # 4) a human approves it at the PC
    assert (await client.post("/pair/approve", json={"request_id": rid})).status == 200
    # 5) the next poll returns approved + the token
    body = await (await client.get(f"/pair/poll?request_id={rid}")).json()
    assert body["status"] == "approved" and body["token"] == "the-token"
    # 6) it's no longer pending
    assert (await (await client.get("/pair/pending")).json())["pending"] == []


@pytest.mark.asyncio
async def test_pair_deny_never_hands_over_a_token(server_client):
    client, server = server_client
    server._settings.remote.token = "the-token"
    rid = (await (await client.post(
        "/pair/request", json={"name": "X"})).json())["request_id"]
    assert (await client.post("/pair/deny", json={"request_id": rid})).status == 200
    body = await (await client.get(f"/pair/poll?request_id={rid}")).json()
    assert body["status"] == "denied" and "token" not in body


@pytest.mark.asyncio
async def test_approve_and_pending_require_being_at_the_pc(lan_client: TestClient):
    # A LAN device (no token) may request + poll — the auth-exempt bootstrap...
    rid = (await (await lan_client.post(
        "/pair/request", json={"name": "Rogue"})).json())["request_id"]
    assert (await (await lan_client.get(
        f"/pair/poll?request_id={rid}")).json())["status"] == "pending"
    # ...but it CANNOT approve itself or list the queue without the token: that
    # is exactly what makes "approve at the PC" a real gate, not decoration.
    assert (await lan_client.post(
        "/pair/approve", json={"request_id": rid})).status == 401
    assert (await lan_client.get("/pair/pending")).status == 401
    # An already-trusted client (holds the token) can approve.
    hdr = {"Authorization": "Bearer watch-secret"}
    assert (await lan_client.post(
        "/pair/approve", json={"request_id": rid}, headers=hdr)).status == 200


@pytest.mark.asyncio
async def test_poll_unknown_request_reads_as_expired(client: TestClient):
    body = await (await client.get("/pair/poll?request_id=nonexistent")).json()
    assert body["status"] == "expired"


@pytest.mark.asyncio
async def test_pending_requests_are_capped(server_client):
    client, _ = server_client
    for _ in range(PAIR_MAX_PENDING):
        assert (await client.post("/pair/request", json={"name": "d"})).status == 200
    # one more over the cap is refused, so a LAN peer can't flood the queue
    assert (await client.post("/pair/request", json={"name": "d"})).status == 429


@pytest.mark.asyncio
async def test_expired_request_is_pruned(server_client):
    client, server = server_client
    rid = (await (await client.post(
        "/pair/request", json={"name": "d"})).json())["request_id"]
    server._pair_requests[rid]["created"] -= 10_000  # far past the TTL
    body = await (await client.get(f"/pair/poll?request_id={rid}")).json()
    assert body["status"] == "expired"
    assert (await (await client.get("/pair/pending")).json())["pending"] == []


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
    assert data["on"] is True and server._settings.mode.lion is True
    assert (await (await client.get("/status")).json())["lion_mode"] is True

    await client.post("/control/lion", json={"on": False})
    assert server._settings.mode.lion is False


@pytest.mark.asyncio
async def test_lion_mode_rejects_a_malformed_body(server_client):
    client, _ = server_client
    assert (await client.post("/control/lion", data="not json")).status == 400
