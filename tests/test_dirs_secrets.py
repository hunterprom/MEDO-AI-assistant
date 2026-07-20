"""Folder-shortcut whitelist + git-ignored LLM secrets round-trip.

Covers core.dirs (the /open whitelist gate) and core.config's local secrets
helpers, plus the companion API's /open + /dirs endpoints over real HTTP.
"""

from __future__ import annotations

import pathlib
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from core import config, dirs
from core.config import (
    apply_local_secrets,
    load_llm_secrets,
    load_settings,
    save_audio_input,
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


def test_sphere_dirs_scatters_many_dots_with_one_center():
    s = dirs.sphere_dirs()
    assert len(s) >= len(dirs.user_dirs())        # at least the curated set
    assert sum(1 for d in s if d["center"]) == 1  # exactly one centre dot
    assert all("primary" in d for d in s)
    # curated folders are the labelled (primary) ones
    assert any(d["primary"] for d in s)


def test_sphere_dirs_respects_limit():
    assert len(dirs.sphere_dirs(limit=5)) <= 5


def test_search_files_finds_matches_and_ignores_empty(tmp_path, monkeypatch):
    (tmp_path / "alpha_report.txt").write_text("x")
    sub = tmp_path / "sub"; sub.mkdir()
    (sub / "beta_report.txt").write_text("y")
    (tmp_path / "unrelated.log").write_text("z")
    monkeypatch.setattr(dirs, "_search_roots", lambda: [tmp_path])
    names = {r["name"] for r in dirs.search_files("report")}
    assert "alpha_report.txt" in names and "beta_report.txt" in names
    assert "unrelated.log" not in names
    assert dirs.search_files("") == []          # empty query -> nothing


def test_resolve_path_allows_home_blocks_outside(tmp_path, monkeypatch):
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    f = tmp_path / "doc.txt"; f.write_text("x")
    assert dirs.resolve_path(str(f)) is not None
    outside = tmp_path.parent / "elsewhere_medo"; outside.mkdir(exist_ok=True)
    assert dirs.resolve_path(str(outside)) is None       # exists but outside home
    assert dirs.resolve_path(str(tmp_path / "missing")) is None


def test_sphere_path_key_resolves_and_traversal_still_blocked():
    s = dirs.sphere_dirs()
    extra = next((d for d in s if not d["primary"]), None)
    if extra is not None:                         # only if this machine has sub-dirs
        assert dirs.resolve(extra["key"]) == pathlib.Path(extra["path"])
    assert dirs.resolve("/etc/shadow") is None
    assert dirs.resolve("../../secret") is None


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


def test_audio_and_llm_overrides_coexist(tmp_path):
    path = tmp_path / "secrets.local.yaml"
    save_llm_secrets(path, provider="openai", api_key="sk-keep")
    save_audio_input(4, path)                     # must not wipe the llm section
    assert load_llm_secrets(path)["api_key"] == "sk-keep"
    save_llm_secrets(path, default_model="gpt-4o")  # must not wipe the audio section
    settings = load_settings()
    apply_local_secrets(settings, path)
    assert settings.audio.input_device == 4
    assert settings.llm.api_key == "sk-keep"
    assert settings.llm.default_model == "gpt-4o"


def test_apply_audio_override_accepts_name_and_null(tmp_path):
    path = tmp_path / "secrets.local.yaml"
    save_audio_input("FHD Webcam", path)
    s = load_settings(); apply_local_secrets(s, path)
    assert s.audio.input_device == "FHD Webcam"
    save_audio_input(None, path)
    s2 = load_settings(); apply_local_secrets(s2, path)
    assert s2.audio.input_device is None


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


@pytest.mark.asyncio
async def test_wake_returns_409_when_voice_not_running(api):
    # The api fixture builds a server with no wake_event (voice off).
    resp = await api.post("/wake")
    assert resp.status == 409


@pytest.mark.asyncio
async def test_wake_sets_event_when_voice_running():
    import threading

    ev = threading.Event()
    server = RemoteServer(
        load_settings(), router=None, sm=StateMachine(EventBus()), wake_event=ev  # type: ignore[arg-type]
    )
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    try:
        resp = await client.post("/wake")
        assert resp.status == 200
        assert ev.is_set()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_interrupt_stops_and_listens():
    import threading

    ev = threading.Event()
    server = RemoteServer(
        load_settings(), router=None, sm=StateMachine(EventBus()), wake_event=ev  # type: ignore[arg-type]
    )
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    try:
        resp = await client.post("/interrupt")            # default listen=true
        assert resp.status == 200
        assert ev.is_set()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_search_files_endpoint_empty_query(api):
    resp = await api.get("/search/files?q=")
    assert resp.status == 200
    assert (await resp.json())["results"] == []


@pytest.mark.asyncio
async def test_search_web_endpoint(api, monkeypatch):
    import remote.server as rs

    monkeypatch.setattr(rs, "_web_search", lambda q, n=8: [{"title": "T", "url": "http://x", "snippet": "s"}])
    body = await (await api.get("/search/web?q=hello")).json()
    assert body["results"][0]["url"] == "http://x"
    assert (await (await api.get("/search/web?q=")).json())["results"] == []


@pytest.mark.asyncio
async def test_open_path_rejects_disallowed(api):
    # A real path outside the user's home must not be openable.
    resp = await api.post("/open", json={"path": "C:/Windows/System32"})
    assert resp.status in (403, 404)


@pytest.mark.asyncio
async def test_audio_devices_endpoint(api):
    resp = await api.get("/audio/devices")
    assert resp.status == 200
    body = await resp.json()
    assert "devices" in body and "current" in body  # devices may be [] without audio


@pytest.mark.asyncio
async def test_set_audio_input_updates_settings_and_validates(monkeypatch):
    import voice.audio as va

    # Indices are converted to stable NAMES before applying/persisting
    # (PortAudio re-numbers devices across reboots).
    monkeypatch.setattr(va, "list_input_devices",
                        lambda: [{"index": 4, "name": "Microphone (FHD Webcam)"}])
    settings = load_settings()
    server = RemoteServer(settings, router=None, sm=StateMachine(EventBus()))  # type: ignore[arg-type]
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    try:
        assert (await client.post("/audio/input", json={"device": 4})).status == 200
        assert settings.audio.input_device == "Microphone (FHD Webcam)"
        # an index with no matching device stays an index (best effort)
        assert (await client.post("/audio/input", json={"device": 99})).status == 200
        assert settings.audio.input_device == 99
        assert (await client.post("/audio/input", json={"device": "FHD Webcam"})).status == 200
        assert settings.audio.input_device == "FHD Webcam"
        assert (await client.post("/audio/input", json={"device": None})).status == 200
        assert settings.audio.input_device is None
        # a non-scalar device is rejected
        assert (await client.post("/audio/input", json={"device": {"x": 1}})).status == 400
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_facts_endpoints_crud(tmp_path):
    from core.facts import FactsStore

    class _R:  # the slice of Router the facts endpoints touch
        facts = FactsStore(tmp_path / "facts.db")

    server = RemoteServer(load_settings(), router=_R(), sm=StateMachine(EventBus()))  # type: ignore[arg-type]
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    try:
        assert (await client.post("/facts", json={"fact": "likes espresso"})).status == 200
        body = await (await client.get("/facts")).json()
        assert [f["fact"] for f in body["facts"]] == ["likes espresso"]
        fid = body["facts"][0]["id"]
        assert (await client.post("/facts/delete", json={"id": fid})).status == 200
        assert (await client.post("/facts/delete", json={"id": fid})).status == 404
        assert (await client.post("/facts", json={"fact": ""})).status == 400
        body = await (await client.get("/facts")).json()
        assert body["facts"] == []
    finally:
        await client.close()


# --- companion-API bearer token (remote.token) ------------------------------


def test_ensure_remote_token_generates_persists_and_reloads(tmp_path: pathlib.Path):
    path = tmp_path / "secrets.local.yaml"
    settings = load_settings()
    settings.remote.token = ""

    token = config.ensure_remote_token(settings, path)
    assert token and settings.remote.token == token

    # Stable across "restarts": a fresh settings object gets the same token.
    again = load_settings()
    again.remote.token = ""
    assert config.ensure_remote_token(again, path) == token

    # apply_local_secrets overlays it at startup too (HUD-only runs).
    fresh = load_settings()
    fresh.remote.token = ""
    apply_local_secrets(fresh, path)
    assert fresh.remote.token == token

    # It coexists with the llm section in the same overrides file.
    save_llm_secrets(path, provider="openai")
    assert load_llm_secrets(path)["provider"] == "openai"
    fresh2 = load_settings()
    fresh2.remote.token = ""
    assert config.ensure_remote_token(fresh2, path) == token
