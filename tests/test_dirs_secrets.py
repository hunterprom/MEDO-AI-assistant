"""Folder-shortcut whitelist + git-ignored LLM secrets round-trip.

Covers core.dirs (the /open whitelist gate) and core.config's local secrets
helpers, plus the companion API's /open + /dirs endpoints over real HTTP.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from core import config, dirs
from core.config import (
    apply_local_secrets,
    load_llm_secrets,
    load_settings,
    save_llm_secrets,
)
from core.events import EventBus, StateMachine
from remote.server import RemoteServer


# --- core.dirs --------------------------------------------------------------


def test_user_dirs_has_exactly_one_center():
    centers = [d for d in dirs.user_dirs() if d["center"]]
    assert len(centers) == 1


def test_resolve_only_returns_whitelisted_keys():
    assert dirs.resolve("root") is not None
    # Arbitrary / traversal keys are never resolvable — the gate for /open.
    assert dirs.resolve("../../etc/passwd") is None
    assert dirs.resolve("") is None
    assert dirs.resolve("definitely-not-a-key") is None


# --- local secrets ----------------------------------------------------------


def test_secrets_round_trip_and_overlay(tmp_path):
    path = tmp_path / "secrets.local.yaml"
    save_llm_secrets(path, provider="openai", api_key="sk-123", default_model="gpt-4o-mini")
    assert load_llm_secrets(path) == {
        "provider": "openai",
        "api_key": "sk-123",
        "default_model": "gpt-4o-mini",
    }

    settings = load_settings()
    apply_local_secrets(settings, path)
    assert settings.llm.provider == "openai"
    assert settings.llm.api_key == "sk-123"
    assert settings.llm.default_model == "gpt-4o-mini"


def test_secrets_save_merges_and_ignores_unknown_fields(tmp_path):
    path = tmp_path / "secrets.local.yaml"
    save_llm_secrets(path, provider="openai", api_key="sk-keep")
    # Switching provider later must not wipe the stored key (merge semantics).
    save_llm_secrets(path, provider="ollama", nonsense="x")
    stored = load_llm_secrets(path)
    assert stored["provider"] == "ollama"
    assert stored["api_key"] == "sk-keep"
    assert "nonsense" not in stored


def test_load_missing_secrets_is_empty(tmp_path):
    assert load_llm_secrets(tmp_path / "nope.yaml") == {}


# --- /open + /dirs over HTTP ------------------------------------------------


@pytest_asyncio.fixture
async def api() -> AsyncIterator[TestClient]:
    settings = load_settings()
    server = RemoteServer(settings, router=None, sm=StateMachine(EventBus()))  # type: ignore[arg-type]
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    yield client
    await client.close()


@pytest.mark.asyncio
async def test_dirs_endpoint_lists_shortcuts(api):
    body = await (await api.get("/dirs")).json()
    assert any(d["center"] for d in body["dirs"])


@pytest.mark.asyncio
async def test_open_rejects_unknown_key(api):
    resp = await api.post("/open", json={"key": "../secret"})
    assert resp.status == 404


@pytest.mark.asyncio
async def test_open_known_key_launches_explorer(api, monkeypatch):
    opened: dict = {}
    monkeypatch.setattr(dirs, "open_path", lambda p: opened.setdefault("path", str(p)) or True)
    resp = await api.post("/open", json={"key": "root"})
    assert resp.status == 200
    assert (await resp.json())["ok"] is True
    assert "path" in opened
