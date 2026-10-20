"""P3-2: the local history dashboard.

A searchable record of what was asked and answered, filterable by routing tier.
Two properties matter more than the feature: it must sit behind the SAME token
gate as the rest of the companion API (it is the most private surface MEDO
exposes — every question anyone has asked), and it must add no new way to make
MEDO act. Re-run goes through the ordinary ``/ask``, so a destructive request
re-run from the page meets exactly the confirmation it would have met by voice.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from core.config import load_settings
from core.events import EventBus, StateMachine
from core.metrics import MetricsStore
from core.router import Router
from llm.client import OllamaClient
from remote.server import RemoteServer
from skills.base import SkillRegistry
from skills.datetime_skill import DateTimeSkill


def _server(tmp_path) -> RemoteServer:
    settings = load_settings()
    registry = SkillRegistry()
    registry.register(DateTimeSkill())
    bus = EventBus()
    router = Router(settings, registry, OllamaClient(settings.llm), bus)
    router.model = None
    router.metrics = MetricsStore(tmp_path / "metrics.db")
    return RemoteServer(settings, router, StateMachine(bus))


@pytest_asyncio.fixture
async def client(tmp_path) -> AsyncIterator[tuple[TestClient, RemoteServer]]:
    server = _server(tmp_path)
    test_client = TestClient(TestServer(server.build_app()))
    await test_client.start_server()
    yield test_client, server
    await test_client.close()


def _seed(server: RemoteServer) -> None:
    store = server._metrics()
    store.record_route_event("FAST", "timers", "set a timer for ten minutes",
                             1.0, "Timer set for ten minutes.", 4.0)
    store.record_route_event("LLM", None, "explain how a diode works", None,
                             "A diode conducts in one direction.", 1800.0)
    store.record_route_event("SEMANTIC", "weather", "is it going to rain",
                             0.71, "Cloudy, no rain expected.", 120.0)


# --- the store's query --------------------------------------------------------


def test_history_is_newest_first(tmp_path):
    store = MetricsStore(tmp_path / "m.db")
    store.record_route_event("FAST", "a", "first", 1.0, "one")
    store.record_route_event("FAST", "b", "second", 1.0, "two")
    assert [t["text"] for t in store.history()] == ["second", "first"]


def test_history_searches_the_question(tmp_path):
    store = MetricsStore(tmp_path / "m.db")
    store.record_route_event("FAST", "a", "set a timer", 1.0, "Timer set.")
    store.record_route_event("LLM", None, "explain diodes", None, "A diode...")
    assert [t["text"] for t in store.history(query="diode")] == ["explain diodes"]


def test_history_searches_the_answer_too(tmp_path):
    """"What did MEDO say about guttering" is as good a question as any."""
    store = MetricsStore(tmp_path / "m.db")
    store.record_route_event("LLM", None, "about the roof", None,
                             "You will need new guttering.")
    assert store.history(query="guttering")


def test_the_search_is_case_insensitive(tmp_path):
    store = MetricsStore(tmp_path / "m.db")
    store.record_route_event("LLM", None, "Explain DIODES", None, "x")
    assert store.history(query="diodes")


def test_a_wildcard_in_the_search_is_matched_literally(tmp_path):
    """Otherwise "%" quietly turns a search into "show me everything"."""
    store = MetricsStore(tmp_path / "m.db")
    store.record_route_event("FAST", "a", "battery at 50%", 1.0, "ok")
    store.record_route_event("FAST", "b", "unrelated", 1.0, "ok")
    assert [t["text"] for t in store.history(query="50%")] == ["battery at 50%"]


def test_history_filters_by_tier(tmp_path):
    store = MetricsStore(tmp_path / "m.db")
    store.record_route_event("FAST", "a", "one", 1.0, "x")
    store.record_route_event("LLM", None, "two", None, "y")
    assert [t["text"] for t in store.history(tier="LLM")] == ["two"]
    assert [t["text"] for t in store.history(tier="fast")] == ["one"]


def test_history_honours_the_limit(tmp_path):
    store = MetricsStore(tmp_path / "m.db")
    for i in range(10):
        store.record_route_event("FAST", "a", f"turn {i}", 1.0, "x")
    assert len(store.history(limit=3)) == 3


def test_the_tier_list_comes_from_the_data(tmp_path):
    store = MetricsStore(tmp_path / "m.db")
    store.record_route_event("FAST", "a", "one", 1.0, "x")
    store.record_route_event("LLM", None, "two", None, "y")
    assert store.history_tiers() == ["FAST", "LLM"]


def test_history_survives_a_database_that_predates_the_columns(tmp_path):
    """An existing install must not have to delete its history for this."""
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE route_events (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                 " ts TEXT, tier TEXT NOT NULL, skill TEXT, text TEXT,"
                 " confidence REAL, corrected INTEGER DEFAULT 0, correction TEXT)")
    conn.execute("INSERT INTO route_events (ts, tier, text) "
                 "VALUES ('2026-01-01', 'FAST', 'an old turn')")
    conn.commit()
    conn.close()

    store = MetricsStore(path)
    turns = store.history()
    assert [t["text"] for t in turns] == ["an old turn"]
    assert turns[0]["reply"] is None            # the column was added empty
    store.record_route_event("FAST", "a", "a new turn", 1.0, "with a reply")
    assert store.history()[0]["reply"] == "with a reply"


# --- the endpoints ------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_dashboard_page_is_served(client):
    test_client, _server = client
    resp = await test_client.get("/dashboard")
    assert resp.status == 200
    assert resp.content_type == "text/html"
    body = await resp.text()
    assert "MEDO" in body and "history" in body


@pytest.mark.asyncio
async def test_the_history_endpoint_returns_the_turns(client):
    test_client, server = client
    _seed(server)
    body = await (await test_client.get("/history")).json()
    assert body["ok"] is True
    assert len(body["turns"]) == 3
    assert set(body["tiers"]) == {"FAST", "LLM", "SEMANTIC"}


@pytest.mark.asyncio
async def test_the_history_endpoint_searches(client):
    test_client, server = client
    _seed(server)
    body = await (await test_client.get("/history?q=diode")).json()
    assert [t["text"] for t in body["turns"]] == ["explain how a diode works"]


@pytest.mark.asyncio
async def test_the_history_endpoint_filters_by_tier(client):
    test_client, server = client
    _seed(server)
    body = await (await test_client.get("/history?tier=SEMANTIC")).json()
    assert [t["skill"] for t in body["turns"]] == ["weather"]


@pytest.mark.asyncio
async def test_a_nonsense_limit_does_not_error(client):
    test_client, server = client
    _seed(server)
    assert (await test_client.get("/history?limit=banana")).status == 200


@pytest.mark.asyncio
async def test_the_limit_is_capped(client):
    """A page that can ask for a million rows is a way to stall the event loop."""
    test_client, server = client
    for i in range(20):
        server._metrics().record_route_event("FAST", "a", f"turn {i}", 1.0, "x")
    body = await (await test_client.get("/history?limit=999999")).json()
    assert len(body["turns"]) <= 500


@pytest.mark.asyncio
async def test_history_says_so_when_telemetry_is_off(client):
    test_client, server = client
    server._router.metrics = None
    body = await (await test_client.get("/history")).json()
    assert body["turns"] == []
    assert "telemetry is off" in body["note"]


# --- the auth model -----------------------------------------------------------


@pytest.mark.asyncio
async def test_the_dashboard_needs_a_token_from_the_lan(tmp_path):
    """It is the most private surface MEDO has: every question ever asked."""
    server = _server(tmp_path)
    server._settings.remote.token = "watch-secret"
    server._is_local = lambda request: False
    test_client = TestClient(TestServer(server.build_app()))
    await test_client.start_server()
    try:
        assert (await test_client.get("/dashboard")).status == 401
        assert (await test_client.get("/history")).status == 401
    finally:
        await test_client.close()


@pytest.mark.asyncio
async def test_a_token_gets_the_lan_client_in(tmp_path):
    server = _server(tmp_path)
    server._settings.remote.token = "watch-secret"
    server._is_local = lambda request: False
    test_client = TestClient(TestServer(server.build_app()))
    await test_client.start_server()
    try:
        headers = {"Authorization": "Bearer watch-secret"}
        assert (await test_client.get("/history", headers=headers)).status == 200
    finally:
        await test_client.close()


@pytest.mark.asyncio
async def test_a_cross_origin_page_cannot_read_the_history(tmp_path):
    """A website you happen to visit must not be able to read your transcript."""
    server = _server(tmp_path)
    test_client = TestClient(TestServer(server.build_app()))
    await test_client.start_server()
    try:
        resp = await test_client.get(
            "/history", headers={"Origin": "http://evil.example"})
        assert resp.status == 403
    finally:
        await test_client.close()


def test_the_dashboard_adds_no_new_way_to_act():
    """Re-run posts to /ask; the page itself only ever reads."""
    from core.config import PROJECT_ROOT

    page = (PROJECT_ROOT / "remote" / "web" / "dashboard.html").read_text(
        encoding="utf-8")
    posts = [line for line in page.splitlines() if '"POST"' in line]
    assert len(posts) == 1
    assert "/ask" in page


def test_the_dashboard_script_actually_parses():
    """A syntax error anywhere in the page kills ALL of its JavaScript.

    It happened. The regex that escapes a search term before highlighting had
    an unterminated character class, so the browser threw at parse time and the
    dashboard rendered its empty shell forever — served, token-gated, correct in
    every Python test here, and completely inert in a browser. Nothing in this
    file could see it, because none of it reads the JavaScript. This hands the
    page to a real parser whenever one is on the machine.
    """
    import re
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    from core.config import PROJECT_ROOT

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed, so the page cannot be parsed here")
    page = (PROJECT_ROOT / "remote" / "web" / "dashboard.html").read_text(
        encoding="utf-8")
    blocks = re.findall(r"<script[^>]*>(.*?)</script>", page, re.DOTALL)
    assert blocks, "the dashboard page has no script block"
    with tempfile.TemporaryDirectory() as tmp:
        for index, block in enumerate(blocks):
            script = Path(tmp) / f"block{index}.js"
            script.write_text(block, encoding="utf-8")
            done = subprocess.run([node, "--check", str(script)],
                                  capture_output=True, text=True)
            assert done.returncode == 0, done.stderr or done.stdout


def test_a_search_term_with_regex_characters_is_escaped_not_executed():
    """The highlighter builds a RegExp out of whatever was typed.

    Searching for "(five)" must find the literal text, not compile a capture
    group — and the escape has to actually INSERT a backslash. Written as
    ``"$&"`` it replaced each metacharacter with itself, which escapes nothing
    and throws on the next unbalanced bracket.
    """
    from core.config import PROJECT_ROOT

    page = (PROJECT_ROOT / "remote" / "web" / "dashboard.html").read_text(
        encoding="utf-8")
    line = next(row for row in page.splitlines() if "new RegExp" in row)
    backslash = chr(92)
    # The replacement must be a literal backslash followed by the match.
    assert '"' + backslash * 2 + '$&"' in line
    # ...and the character class it scans for must itself be closed.
    assert "[" + backslash + "]" + backslash * 2 + "]" in line
