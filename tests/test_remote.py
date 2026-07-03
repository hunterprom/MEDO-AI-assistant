"""Companion API tests: /ping liveness and /ask routing over real HTTP."""

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


@pytest_asyncio.fixture
async def client() -> AsyncIterator[TestClient]:
    settings = load_settings()
    registry = SkillRegistry()
    registry.register(DateTimeSkill())
    bus = EventBus()
    router = Router(settings, registry, OllamaClient(settings.llm), bus)
    router.model = None  # no LLM in tests; fast path only
    server = RemoteServer(settings, router, StateMachine(bus))

    test_client = TestClient(TestServer(server.build_app()))
    await test_client.start_server()
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
