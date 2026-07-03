"""MediaSkill: fast-path patterns + mocked execution — no real key presses.

Every execute test replaces ``skills.media.subprocess.run`` and/or injects a
fake ``pyautogui`` module, so nothing here ever touches a music player.
"""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from skills.base import SkillRequest
from skills.media import MediaSkill


# --- pattern matching --------------------------------------------------------
@pytest.mark.parametrize("utterance", [
    "play music",
    "play the music",
    "pause music",
    "pause the song",
    "resume music",
    "stop the music",
    "next track",
    "next song",
    "skip this song",
    "skip track",
    "previous track",
    "previous song",
    "last song",
])
def test_patterns_match(utterance):
    assert MediaSkill().match(utterance) is not None


@pytest.mark.parametrize("utterance", [
    "open spotify",                 # launching an app is AppsSkill's job
    "play chess with me",           # "play" without a media noun
    "what's next on my calendar",   # "next" without track/song
    "read my notes",
])
def test_patterns_ignore_non_media(utterance):
    assert MediaSkill().match(utterance) is None


@pytest.mark.parametrize("utterance, action", [
    ("play music", "playpause"),
    ("pause the song", "playpause"),
    ("resume music", "playpause"),
    ("next track", "next"),
    ("skip this song", "next"),
    ("previous song", "previous"),
    ("last track", "previous"),
])
def test_action_parsing(utterance, action):
    assert MediaSkill()._action(utterance) == action


# --- macOS path (osascript mocked) -------------------------------------------
def _fake_run_factory(calls, spotify_running: bool):
    """A subprocess.run stand-in that records commands and fakes osascript."""

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        stdout = "true" if ("System Events" in cmd[-1] and spotify_running) else "false"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    return fake_run


@pytest.mark.asyncio
async def test_macos_playpause_targets_music_by_default(monkeypatch):
    calls: list = []
    monkeypatch.setattr("skills.media.IS_MACOS", True)
    monkeypatch.setattr("skills.media.subprocess.run", _fake_run_factory(calls, False))
    result = await MediaSkill().execute(SkillRequest(text="play music"))
    assert result.success and "Music" in result.speech
    assert any('tell application "Music" to playpause' in c[-1] for c in calls)


@pytest.mark.asyncio
async def test_macos_prefers_spotify_when_running(monkeypatch):
    calls: list = []
    monkeypatch.setattr("skills.media.IS_MACOS", True)
    monkeypatch.setattr("skills.media.subprocess.run", _fake_run_factory(calls, True))
    result = await MediaSkill().execute(SkillRequest(text="next track"))
    assert result.success and result.data["action"] == "next"
    assert any('tell application "Spotify" to next track' in c[-1] for c in calls)


@pytest.mark.asyncio
async def test_macos_osascript_failure_is_graceful(monkeypatch):
    def boom(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr("skills.media.IS_MACOS", True)
    monkeypatch.setattr("skills.media.subprocess.run", boom)
    result = await MediaSkill().execute(SkillRequest(text="pause music"))
    assert "couldn't" in result.speech  # spoken failure, not an exception


# --- Windows/Linux path (pyautogui mocked) ------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("utterance, key", [
    ("play music", "playpause"),
    ("next song", "nexttrack"),
    ("previous track", "prevtrack"),
])
async def test_media_keys_pressed_elsewhere(monkeypatch, utterance, key):
    pressed: list[str] = []
    fake = types.ModuleType("pyautogui")
    fake.press = pressed.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyautogui", fake)
    monkeypatch.setattr("skills.media.IS_MACOS", False)
    result = await MediaSkill().execute(SkillRequest(text=utterance))
    assert result.success and pressed == [key]


@pytest.mark.asyncio
async def test_media_keys_failure_is_graceful(monkeypatch):
    fake = types.ModuleType("pyautogui")

    def raise_press(key: str) -> None:
        raise RuntimeError("no display")

    fake.press = raise_press  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyautogui", fake)
    monkeypatch.setattr("skills.media.IS_MACOS", False)
    result = await MediaSkill().execute(SkillRequest(text="play music"))
    assert "couldn't" in result.speech


# --- LLM path plumbing ---------------------------------------------------------
def test_tool_schema_shape():
    schema = MediaSkill().tool_schema()
    assert schema["function"]["name"] == "media"
    assert schema["function"]["parameters"]["properties"]["action"]["enum"] == [
        "playpause", "next", "previous",
    ]
