"""The HUD's controlled-browser picker: chrome | msedge | chromium | opera.

Opera (or any Chromium-based browser) is driven via Playwright ``executable_path``
on MEDO's OWN profile — never the browser's real profile — so the "never touch
Opera browser data" rule holds. Persisted per machine in the git-ignored
secrets file, re-applied at startup.
"""

from __future__ import annotations

import pytest

import core.config as cfg
from core.config import (
    BrowserConfig,
    apply_local_secrets,
    browser_choice,
    load_settings,
    resolve_browser,
    save_browser,
)


# --- resolve/validate --------------------------------------------------------

def test_resolve_channels():
    assert resolve_browser("chrome") == {
        "choice": "chrome", "channel": "chrome", "executable_path": ""}
    assert resolve_browser("msedge")["channel"] == "msedge"
    assert resolve_browser("chromium") == {
        "choice": "chromium", "channel": "", "executable_path": ""}


def test_resolve_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_browser("firefox")            # not a Chromium browser Playwright drives


def test_opera_uses_a_detected_executable(monkeypatch):
    monkeypatch.setattr(cfg, "find_browser_executable",
                        lambda name: r"C:\Opera\opera.exe")
    r = resolve_browser("opera")
    assert r["channel"] == ""                 # executable wins over channel
    assert r["executable_path"] == r"C:\Opera\opera.exe"


def test_opera_not_installed_raises(monkeypatch):
    monkeypatch.setattr(cfg, "find_browser_executable", lambda name: None)
    with pytest.raises(ValueError):
        resolve_browser("opera")


# --- the current-choice label ------------------------------------------------

def test_browser_choice_label():
    assert browser_choice(BrowserConfig(channel="chrome")) == "chrome"
    assert browser_choice(BrowserConfig(channel="")) == "chromium"
    assert browser_choice(
        BrowserConfig(executable_path=r"C:\Users\me\AppData\Local\Programs\Opera\opera.exe")
    ) == "opera"


# --- persistence round-trip --------------------------------------------------

def test_save_and_apply(tmp_path):
    path = tmp_path / "secrets.local.yaml"
    save_browser("msedge", path)
    s = apply_local_secrets(load_settings(), path)
    assert s.browser.channel == "msedge"
    assert s.browser.executable_path == ""


def test_save_opera_persists_the_executable(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "find_browser_executable",
                        lambda name: r"C:\Opera\opera.exe")
    path = tmp_path / "secrets.local.yaml"
    save_browser("opera", path)
    s = apply_local_secrets(load_settings(), path)
    assert s.browser.executable_path == r"C:\Opera\opera.exe"
    assert browser_choice(s.browser) == "opera"


# --- the companion endpoint --------------------------------------------------

@pytest.mark.asyncio
async def test_browser_endpoint(tmp_path):
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
        resp = await client.post("/control/browser", json={"browser": "msedge"})
        assert resp.status == 200
        assert settings.browser.channel == "msedge"
        assert (await (await client.get("/status")).json())["browser"] == "msedge"

        # a browser Playwright can't drive is a 400, settings untouched
        resp = await client.post("/control/browser", json={"browser": "firefox"})
        assert resp.status == 400
        assert settings.browser.channel == "msedge"
    finally:
        await client.close()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
