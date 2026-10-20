"""P0-1: TLS, per-caller throttling, and refusal logging on the companion API.

The three together are what makes the LAN surface defensible: TLS stops the
token being readable on the wire, the throttle stops it being guessed, and the
log makes both attempts visible after the fact.
"""

from __future__ import annotations

import ssl
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from core.config import load_settings
from core.events import EventBus, StateMachine
from core.metrics import MetricsStore
from core.router import Router
from llm.client import OllamaClient
from remote import tls
from remote.ratelimit import IDLE_EVICT_S, MAX_BUCKETS, RateLimiter, caller_key
from remote.server import RemoteServer
from skills.base import SkillRegistry
from skills.datetime_skill import DateTimeSkill


def _make_server(tmp_path=None) -> RemoteServer:
    settings = load_settings()
    registry = SkillRegistry()
    registry.register(DateTimeSkill())
    bus = EventBus()
    router = Router(settings, registry, OllamaClient(settings.llm), bus)
    router.model = None
    if tmp_path is not None:
        router.metrics = MetricsStore(tmp_path / "metrics.db")
    return RemoteServer(settings, router, StateMachine(bus))


@pytest_asyncio.fixture
async def lan(tmp_path) -> AsyncIterator[tuple[TestClient, RemoteServer]]:
    """A client the server treats as a LAN peer, with metrics wired up."""
    server = _make_server(tmp_path)
    server._settings.remote.token = "watch-secret"
    server._is_local = lambda request: False
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    yield client, server
    await client.close()


# --- rate limiter (pure) -----------------------------------------------------


def test_bucket_starts_full_so_a_first_request_is_never_dropped():
    limiter = RateLimiter(per_minute=60, burst=5)
    assert limiter.allow("ip:1.2.3.4", now=0.0)


def test_burst_is_spent_then_refused():
    limiter = RateLimiter(per_minute=60, burst=3)
    assert [limiter.allow("k", now=0.0) for _ in range(4)] == [True, True, True, False]


def test_refills_at_the_sustained_rate():
    limiter = RateLimiter(per_minute=60, burst=2)      # one per second
    assert limiter.allow("k", now=0.0)
    assert limiter.allow("k", now=0.0)
    assert not limiter.allow("k", now=0.0)
    assert limiter.allow("k", now=1.0)                 # one second, one token


def test_refill_never_exceeds_the_burst_depth():
    limiter = RateLimiter(per_minute=600, burst=3)
    limiter.allow("k", now=0.0)
    # An hour idle must not bank an hour of quota.
    assert [limiter.allow("k", now=3600.0) for _ in range(4)] == [
        True, True, True, False]


def test_callers_have_independent_buckets():
    limiter = RateLimiter(per_minute=60, burst=1)
    assert limiter.allow("a", now=0.0)
    assert not limiter.allow("a", now=0.0)
    assert limiter.allow("b", now=0.0)


def test_zero_per_minute_disables_the_limiter():
    limiter = RateLimiter(per_minute=0, burst=0)
    assert not limiter.enabled
    assert all(limiter.allow("k", now=0.0) for _ in range(1000))


def test_retry_after_is_at_least_one_second():
    limiter = RateLimiter(per_minute=60, burst=1)
    limiter.allow("k", now=0.0)
    assert limiter.retry_after("k", now=0.0) >= 1


def test_idle_buckets_are_evicted():
    limiter = RateLimiter(per_minute=60, burst=1)
    limiter.allow("old", now=0.0)
    limiter.allow("new", now=IDLE_EVICT_S + 1.0)
    assert "old" not in limiter._buckets


def test_bucket_table_is_capped_under_a_spray():
    limiter = RateLimiter(per_minute=60, burst=1)
    for i in range(MAX_BUCKETS + 50):
        limiter.allow(f"ip:10.0.{i // 256}.{i % 256}", now=float(i) * 0.001)
    assert len(limiter._buckets) <= MAX_BUCKETS


def test_caller_key_never_contains_the_token():
    key = caller_key("super-secret-token", "10.0.0.9")
    assert "super-secret-token" not in key
    assert key.startswith("tok:")


def test_caller_key_falls_back_to_the_peer_address():
    assert caller_key(None, "10.0.0.9") == "ip:10.0.0.9"


# --- throttling, end to end --------------------------------------------------


@pytest.mark.asyncio
async def test_lan_client_is_throttled_after_its_burst(lan):
    client, server = lan
    server._limiter = RateLimiter(per_minute=60, burst=2)
    headers = {"Authorization": "Bearer watch-secret"}
    statuses = [(await client.get("/ping", headers=headers)).status for _ in range(3)]
    assert statuses == [200, 200, 429]


@pytest.mark.asyncio
async def test_a_429_tells_the_client_when_to_come_back(lan):
    client, server = lan
    server._limiter = RateLimiter(per_minute=60, burst=1)
    headers = {"Authorization": "Bearer watch-secret"}
    await client.get("/ping", headers=headers)
    resp = await client.get("/ping", headers=headers)
    assert resp.status == 429
    assert int(resp.headers["Retry-After"]) >= 1


@pytest.mark.asyncio
async def test_guessing_tokens_cannot_buy_a_fresh_bucket_each_try(lan):
    """The attack the throttle exists for: a different wrong token every time.

    Keyed by token, each guess would land in its own full bucket and the
    limiter would never fire. Keyed by address (what an *unauthorized* caller
    gets), the guesses share one bucket and run out.
    """
    client, server = lan
    server._limiter = RateLimiter(per_minute=60, burst=3)
    statuses = []
    for i in range(5):
        resp = await client.get("/ping", headers={"Authorization": f"Bearer guess{i}"})
        statuses.append(resp.status)
    assert statuses[:3] == [401, 401, 401]
    assert statuses[3:] == [429, 429]


@pytest.mark.asyncio
async def test_loopback_is_exempt_so_the_hud_keeps_polling(tmp_path):
    server = _make_server(tmp_path)
    server._limiter = RateLimiter(per_minute=60, burst=1)
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    try:
        statuses = [(await client.get("/ping")).status for _ in range(5)]
        assert statuses == [200] * 5
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_loopback_exemption_can_be_switched_off(tmp_path):
    server = _make_server(tmp_path)
    server._settings.remote.rate_limit_local_exempt = False
    server._limiter = RateLimiter(per_minute=60, burst=1)
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    try:
        assert (await client.get("/ping")).status == 200
        assert (await client.get("/ping")).status == 429
    finally:
        await client.close()


# --- refusal logging ---------------------------------------------------------


@pytest.mark.asyncio
async def test_a_bad_token_is_recorded_in_the_metrics_db(lan):
    client, server = lan
    await client.get("/ping", headers={"Authorization": "Bearer wrong"})
    events = server._metrics().auth_failures()
    assert [e["reason"] for e in events] == ["bad_token"]
    assert events[0]["path"] == "/ping"
    assert events[0]["method"] == "GET"


@pytest.mark.asyncio
async def test_a_missing_token_is_recorded_distinctly(lan):
    client, server = lan
    await client.get("/ping")
    assert [e["reason"] for e in server._metrics().auth_failures()] == ["no_token"]


@pytest.mark.asyncio
async def test_a_throttled_request_is_recorded(lan):
    client, server = lan
    server._limiter = RateLimiter(per_minute=60, burst=1)
    headers = {"Authorization": "Bearer watch-secret"}
    await client.get("/ping", headers=headers)
    await client.get("/ping", headers=headers)
    assert "rate_limited" in {e["reason"] for e in server._metrics().auth_failures()}


@pytest.mark.asyncio
async def test_a_cross_origin_request_is_recorded(lan):
    client, _server = lan
    resp = await client.get("/ping", headers={"Origin": "http://evil.example"})
    assert resp.status == 403
    _c, server = lan
    assert "cross_origin" in {e["reason"] for e in server._metrics().auth_failures()}


@pytest.mark.asyncio
async def test_the_rejected_token_is_never_written_to_the_database(lan):
    client, server = lan
    await client.get("/ping", headers={"Authorization": "Bearer hunter2"})
    dumped = repr(server._metrics().auth_failures())
    assert "hunter2" not in dumped


@pytest.mark.asyncio
async def test_a_successful_request_logs_nothing(lan):
    client, server = lan
    resp = await client.get("/ping", headers={"Authorization": "Bearer watch-secret"})
    assert resp.status == 200
    assert server._metrics().auth_failures() == []


def test_auth_report_groups_by_reason(tmp_path):
    store = MetricsStore(tmp_path / "m.db")
    store.record_auth_event("bad_token", peer="10.0.0.5", path="/ask")
    store.record_auth_event("bad_token", peer="10.0.0.5", path="/ask")
    store.record_auth_event("rate_limited", peer="10.0.0.6", path="/ping")
    report = store.auth_report()
    assert "bad_token" in report and "| 2 |" in report
    assert "10.0.0.5" in report


def test_auth_report_says_so_when_nothing_was_refused(tmp_path):
    assert "No refused" in MetricsStore(tmp_path / "m.db").auth_report()


# --- TLS ---------------------------------------------------------------------


def test_a_certificate_is_minted_on_first_run(tmp_path):
    cert, key = tls.ensure_certificate(tmp_path / "tls")
    assert cert.exists() and key.exists()
    assert cert.read_bytes().startswith(b"-----BEGIN CERTIFICATE-----")


def test_the_certificate_is_reused_on_the_next_run(tmp_path):
    cert, _ = tls.ensure_certificate(tmp_path / "tls")
    first = tls._digest_of(cert)
    cert2, _ = tls.ensure_certificate(tmp_path / "tls")
    assert tls._digest_of(cert2) == first


def test_it_covers_localhost_and_the_loopback_address(tmp_path):
    from cryptography import x509

    cert_path, _ = tls.ensure_certificate(tmp_path / "tls")
    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "localhost" in san.get_values_for_type(x509.DNSName)
    assert "127.0.0.1" in {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}


def test_it_is_reminted_when_the_machine_gains_an_address(tmp_path):
    """A laptop that joins a new network must stop serving a stale certificate."""
    cert, _ = tls.ensure_certificate(
        tmp_path / "tls", hostnames=["localhost"], addresses=["127.0.0.1"])
    first = tls._digest_of(cert)
    cert2, _ = tls.ensure_certificate(
        tmp_path / "tls", hostnames=["localhost"],
        addresses=["127.0.0.1", "192.168.1.50"])
    assert tls._digest_of(cert2) != first


def test_the_private_key_is_not_world_readable(tmp_path):
    import os
    import stat

    _, key = tls.ensure_certificate(tmp_path / "tls")
    if os.name == "nt":       # POSIX mode bits aren't meaningful on Windows
        pytest.skip("POSIX permissions")
    assert not stat.S_IMODE(key.stat().st_mode) & 0o077


def test_the_fingerprint_is_a_stable_sha256(tmp_path):
    cert, _ = tls.ensure_certificate(tmp_path / "tls")
    fp = tls.fingerprint(cert)
    assert len(fp.split(":")) == 32
    assert fp == tls.fingerprint(cert)


def test_an_operator_certificate_is_used_verbatim(tmp_path):
    mine = tmp_path / "mine"
    cert, key = tls.generate_certificate(mine / "c.pem", mine / "k.pem")
    used_cert, used_key = tls.ensure_certificate(
        tmp_path / "unused", certfile=cert, keyfile=key)
    assert (used_cert, used_key) == (cert, key)
    assert not (tmp_path / "unused").exists()


def test_a_missing_operator_certificate_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="don't exist"):
        tls.ensure_certificate(tmp_path, certfile=tmp_path / "nope.pem",
                               keyfile=tmp_path / "nope.key")


def test_the_ssl_context_refuses_obsolete_tls(tmp_path):
    cert, key = tls.ensure_certificate(tmp_path / "tls")
    ctx = tls.ssl_context(cert, key)
    assert ctx.minimum_version >= ssl.TLSVersion.TLSv1_2


@pytest.mark.asyncio
async def test_the_api_actually_serves_over_https(tmp_path):
    """End to end: bind with TLS and complete a real handshake against it.

    The client does not verify the chain, and that is deliberate rather than
    lazy: an antivirus with HTTPS scanning turned on (Avast, Kaspersky, ESET)
    re-signs even loopback connections with its own root, so the certificate a
    local client receives is NOT the one this server presented. Asserting on
    the chain would make this test a detector of the developer's antivirus.
    What TLS has to guarantee here is that the socket is encrypted and the API
    still answers over it — that is what is checked. The certificate's own
    identity is asserted server-side, from ``/tls``, where no proxy can
    interpose.
    """
    import aiohttp

    server = _make_server(tmp_path)
    server._settings.remote.tls_enabled = True
    cert, key = tls.ensure_certificate(tmp_path / "tls")
    server._settings.remote.tls_certfile = str(cert)
    server._settings.remote.tls_keyfile = str(key)
    server._settings.remote.port = 0
    server._settings.remote.host = "127.0.0.1"
    server._settings.remote.discovery_enabled = False
    await server.start()
    try:
        assert server.scheme == "https"
        port = server._runner.addresses[0][1]
        insecure = ssl.create_default_context()
        insecure.check_hostname = False
        insecure.verify_mode = ssl.CERT_NONE
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://127.0.0.1:{port}/ping",
                                   ssl=insecure) as resp:
                assert resp.status == 200
                assert (await resp.json())["service"] == "medo"
            async with session.get(f"https://127.0.0.1:{port}/tls",
                                   ssl=insecure) as resp:
                body = await resp.json()
        assert body["tls"] is True
        assert body["scheme"] == "https"
        assert body["fingerprint_sha256"] == tls.fingerprint(cert)
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_plain_http_is_refused_once_tls_is_on(tmp_path):
    """A client that forgot to switch scheme must fail, not silently downgrade."""
    import aiohttp

    server = _make_server(tmp_path)
    server._settings.remote.tls_enabled = True
    cert, key = tls.ensure_certificate(tmp_path / "tls")
    server._settings.remote.tls_certfile = str(cert)
    server._settings.remote.tls_keyfile = str(key)
    server._settings.remote.port = 0
    server._settings.remote.host = "127.0.0.1"
    server._settings.remote.discovery_enabled = False
    await server.start()
    try:
        port = server._runner.addresses[0][1]
        async with aiohttp.ClientSession() as session:
            with pytest.raises(aiohttp.ClientError):
                await session.get(f"http://127.0.0.1:{port}/ping",
                                  timeout=aiohttp.ClientTimeout(total=5))
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_a_broken_certificate_downgrades_instead_of_refusing_to_boot(tmp_path):
    server = _make_server(tmp_path)
    server._settings.remote.tls_enabled = True
    server._settings.remote.tls_certfile = str(tmp_path / "missing.pem")
    server._settings.remote.tls_keyfile = str(tmp_path / "missing.key")
    assert server._ssl_context() is None
    assert server.scheme == "http"


@pytest.mark.asyncio
async def test_the_tls_endpoint_reports_no_tls_when_plain(tmp_path):
    server = _make_server(tmp_path)
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    try:
        body = await (await client.get("/tls")).json()
        assert body["tls"] is False
        assert body["scheme"] == "http"
        assert body["fingerprint_sha256"] is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_the_tls_endpoint_needs_a_token(lan):
    client, _server = lan
    assert (await client.get("/tls")).status == 401
