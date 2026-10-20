"""Decision logic of the skills that ACT, and the gate that stands in front.

The code that can hurt is the code that runs a command: power, volume, typing,
launching apps, and the router branch that decides whether any of it is allowed
to run at all. That branch — the LLM tool-call path — was the least-covered
safety-critical code in the project despite being the one an invented tool call
arrives through.

Everything here mocks at the OS boundary and asserts on what was *invoked*, so
a regression shows up as a command that ran, not as a changed sentence.
"""

from __future__ import annotations

import re

import pytest

from core.config import load_settings
from core.events import EventBus, RoutePath
from core.router import PC_CONTROL_OFF_REPLY, Router
from llm.client import OllamaClient
from skills.apps import AppsSkill
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult
from skills.timers import parse_duration


# --- the gate on the LLM tool path -------------------------------------------
#
# An invented tool call arrives here, not through a regex. This is the path a
# hallucinated "power" call takes.


class _Actuator(Skill):
    name = "apps"
    description = "test actuation"
    controls_pc = True
    patterns = [re.compile(r"\bnever matches this\b")]

    def __init__(self):
        self.ran = False

    async def execute(self, request: SkillRequest) -> SkillResult:
        self.ran = True
        return SkillResult("Did it.")


class _Destructive(Skill):
    name = "power"
    description = "test destructive"
    controls_pc = True
    patterns = [re.compile(r"\bnever matches this\b")]

    def __init__(self):
        self.ran = False

    async def execute(self, request: SkillRequest) -> SkillResult:
        if not request.context.get("confirmed"):
            return SkillResult("Sure?", needs_confirmation=True)
        self.ran = True
        return SkillResult("Done.")


def _tool_router(monkeypatch, tool_name, *, pc_on=True, lion=False):
    """A router whose model always calls ``tool_name`` once, then answers."""
    settings = load_settings()
    settings.safety.pc_control_enabled = pc_on
    settings.mode.lion = lion
    act, boom = _Actuator(), _Destructive()
    registry = SkillRegistry()
    registry.register(act)
    registry.register(boom)
    router = Router(settings, registry, OllamaClient(settings.llm), EventBus())
    router.model = "test-model"

    calls = {"n": 0}

    async def fake_chat(model, messages, tools=None, temperature=None, on_delta=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"content": "", "tool_calls": [
                {"function": {"name": tool_name, "arguments": {}}}]}
        return {"content": "All done."}

    monkeypatch.setattr(router._llm, "chat", fake_chat)
    return router, act, boom


@pytest.mark.asyncio
async def test_llm_tool_call_is_refused_when_pc_control_is_off(monkeypatch):
    router, act, _ = _tool_router(monkeypatch, "apps", pc_on=False)
    result = await router.route("do the thing")
    assert act.ran is False
    assert PC_CONTROL_OFF_REPLY in result.speech or result.speech == "All done."


@pytest.mark.asyncio
async def test_llm_tool_call_runs_when_pc_control_is_on(monkeypatch):
    router, act, _ = _tool_router(monkeypatch, "apps", pc_on=True)
    await router.route("do the thing")
    assert act.ran is True


@pytest.mark.asyncio
async def test_llm_destructive_tool_call_stops_for_confirmation(monkeypatch):
    router, _, boom = _tool_router(monkeypatch, "power")
    result = await router.route("power down")
    assert boom.ran is False
    assert result.speech == "Sure?" and router.awaiting_confirmation is True


@pytest.mark.asyncio
async def test_lion_mode_does_not_let_a_destructive_tool_call_through(monkeypatch):
    # Lion mode is a presentation profile; the LLM-path gate is unchanged.
    router, _, boom = _tool_router(monkeypatch, "power", lion=True)
    result = await router.route("power down")
    assert boom.ran is False
    assert result.speech == "Sure?" and router.awaiting_confirmation is True


@pytest.mark.asyncio
async def test_lion_mode_does_not_override_pc_control_on_the_llm_path(monkeypatch):
    router, act, _ = _tool_router(monkeypatch, "apps", pc_on=False, lion=True)
    result = await router.route("do the thing")
    assert act.ran is False
    assert PC_CONTROL_OFF_REPLY in result.speech or result.speech == "All done."


@pytest.mark.asyncio
async def test_an_unknown_tool_name_does_not_crash_the_turn(monkeypatch):
    router, act, boom = _tool_router(monkeypatch, "no_such_tool")
    result = await router.route("do something imaginary")
    assert act.ran is False and boom.ran is False
    assert result.path is RoutePath.LLM and result.speech


# --- confirmation resolution --------------------------------------------------


@pytest.mark.asyncio
async def test_saying_no_cancels_the_pending_action(monkeypatch):
    router, _, boom = _tool_router(monkeypatch, "power")
    await router.route("power down")
    result = await router.route("no")
    assert boom.ran is False and router.awaiting_confirmation is False
    assert "cancel" in result.speech.lower()


@pytest.mark.asyncio
async def test_saying_yes_performs_the_pending_action(monkeypatch):
    router, _, boom = _tool_router(monkeypatch, "power")
    await router.route("power down")
    await router.route("yes")
    assert boom.ran is True and router.awaiting_confirmation is False


@pytest.mark.asyncio
async def test_an_ambiguous_answer_cancels_rather_than_guessing(monkeypatch):
    """Never resolve a destructive action from a reply that isn't yes or no."""
    router, _, boom = _tool_router(monkeypatch, "power")
    await router.route("power down")
    await router.route("what time is it")
    assert boom.ran is False and router.awaiting_confirmation is False


# --- apps: closing, and the exe name it kills ---------------------------------


APPS = {
    "chrome": {"windows": "start chrome", "darwin": "open -a 'Google Chrome'",
               "linux": "google-chrome"},
    "browser": {"windows": "start msedge", "darwin": "open -a Safari",
                "linux": "xdg-open https://"},
    "editor": {"windows": r'start "" "C:\Program Files\VS\Code.exe"',
               "darwin": "code", "linux": "code"},
    "terminal": {"windows": "start cmd", "darwin": "open -a Terminal",
                 "linux": "x-terminal-emulator"},
}


@pytest.mark.parametrize("key,expected", [
    ("chrome", "chrome.exe"),
    ("browser", "msedge.exe"),          # the config key is not the exe name
    ("editor", "Code.exe"),             # quoted path -> basename
    ("terminal", "cmd.exe"),
])
def test_windows_image_name_comes_from_the_launch_command(key, expected):
    """Killing <key>.exe only works when key happens to equal the exe."""
    assert AppsSkill(APPS)._win_image_name(key) == expected


def test_unknown_app_image_name_falls_back_to_the_key():
    assert AppsSkill(APPS)._win_image_name("nothing") == "nothing.exe"


@pytest.mark.asyncio
async def test_closing_an_app_invokes_the_platform_killer(monkeypatch):
    killed = []
    monkeypatch.setattr(AppsSkill, "_close", lambda self, key: killed.append(key))
    skill = AppsSkill(APPS)
    phrase = "close chrome"
    result = await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert killed == ["chrome"] and result.data["action"] == "close"


def test_windows_close_is_graceful_not_a_force_kill(monkeypatch):
    """Closing must send WM_CLOSE (no /F), so unsaved work can prompt to save."""
    invoked = []
    monkeypatch.setattr("skills.apps.current_os", lambda: "windows")
    monkeypatch.setattr("skills.apps.subprocess.Popen",
                        lambda cmd, *a, **k: invoked.append(cmd))
    AppsSkill(APPS)._close("chrome")
    assert invoked == [["taskkill", "/IM", "chrome.exe"]]
    assert "/F" not in invoked[0]


@pytest.mark.asyncio
async def test_an_app_missing_for_this_os_is_reported_not_launched(monkeypatch):
    launched = []
    monkeypatch.setattr("skills.apps.run_detached", lambda c: launched.append(c))
    monkeypatch.setattr("skills.apps.pick_for_os", lambda table: "")
    skill = AppsSkill(APPS)
    phrase = "open chrome"
    result = await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert launched == [] and result.success is False


# --- timers: the duration parser ---------------------------------------------


@pytest.mark.parametrize("text,seconds", [
    ("set a timer for 5 minutes", 300),
    ("timer for 90 seconds", 90),
    ("remind me in 2 hours", 7200),
    ("set a timer for one minute", 60),        # spoken number, not a digit
    ("set a timer for half an hour", 1800),
])
def test_parse_duration_understands_spoken_time(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["set a timer", "remind me", "hello"])
def test_parse_duration_returns_nothing_when_no_time_is_named(text):
    assert not parse_duration(text)
