"""Screen agent: confirmation gate, action loop, coord scaling, safety."""

from __future__ import annotations

import json

import pytest

from core.config import load_settings
from skills.base import SkillRequest
from skills.screen_agent import ScreenAgentSkill, parse_action, scale_point


# --- pure helpers ------------------------------------------------------------


def test_scale_point_maps_and_clamps():
    assert scale_point(500, 500, 2560, 1440) == (1280, 720)   # centre
    assert scale_point(0, 0, 2560, 1440) == (0, 0)
    assert scale_point(1000, 1000, 2560, 1440) == (2559, 1439)
    assert scale_point(-50, 5000, 2560, 1440) == (0, 1439)    # clamped


def test_parse_action_tolerates_prose_and_junk():
    assert parse_action('{"action":"done","say":"ok"}')["action"] == "done"
    assert parse_action('Sure! {"action":"click","x":10,"y":20} done')["x"] == 10
    assert parse_action("no json here") is None
    assert parse_action("") is None


# --- the agent loop (all backends mocked) -----------------------------------


class _Img:
    size = (2560, 1440)

    def save(self, buf, format="PNG"):
        buf.write(b"fake")


def _skill(script, acted, *, enabled=True):
    """script = list of action dicts the fake model returns in order."""
    settings = load_settings()
    settings.vision.agent_enabled = enabled
    settings.vision.agent_max_steps = 6
    calls = {"i": 0}

    def capture():
        return _Img(), _Img.size

    async def ask(image, task, history):
        i = calls["i"]
        calls["i"] += 1
        return json.dumps(script[min(i, len(script) - 1)])

    def act(action, w, h):
        acted.append((action["action"], w, h))
        return action["action"]

    return ScreenAgentSkill(settings, capture=capture, ask=ask, act=act), calls


@pytest.mark.asyncio
async def test_first_call_asks_for_confirmation():
    skill, _ = _skill([{"action": "done"}], [])
    r = await skill.execute(SkillRequest(text="do this for me: open notepad"))
    assert r.needs_confirmation is True
    assert "open notepad" in r.speech


@pytest.mark.asyncio
async def test_confirmed_runs_actions_until_done():
    acted = []
    script = [
        {"action": "click", "x": 500, "y": 500},
        {"action": "type", "text": "hello"},
        {"action": "done", "say": "Typed hello."},
    ]
    skill, calls = _skill(script, acted)
    r = await skill.execute(SkillRequest(
        text="do this for me: type hello", context={"confirmed": True}))
    assert r.success and r.speech == "Typed hello."
    assert [a[0] for a in acted] == ["click", "type"]   # stopped at done
    assert r.data["steps"] == 2


@pytest.mark.asyncio
async def test_step_cap_stops_a_runaway():
    acted = []
    skill, _ = _skill([{"action": "scroll", "amount": 1}], acted)  # never says done
    r = await skill.execute(SkillRequest(
        text="operate my screen", context={"confirmed": True}))
    assert len(acted) == 6 and "limit" in r.speech


@pytest.mark.asyncio
async def test_unparseable_action_stops_safely():
    async def ask(image, task, history):
        return "I cannot help with that."

    settings = load_settings()
    skill = ScreenAgentSkill(settings, capture=lambda: (_Img(), _Img.size),
                             ask=ask, act=lambda *a: "x")
    r = await skill.execute(SkillRequest(
        text="do this for me: x", context={"confirmed": True}))
    assert not r.success and "couldn't work out" in r.speech


@pytest.mark.asyncio
async def test_disabled_config_refuses():
    skill, _ = _skill([{"action": "done"}], [], enabled=False)
    r = await skill.execute(SkillRequest(
        text="do this for me: x", context={"confirmed": True}))
    assert not r.success and "disabled" in r.speech


def test_safety_flags_and_patterns():
    assert ScreenAgentSkill.controls_pc is True
    assert ScreenAgentSkill.requires_confirmation is True
    s = _skill([{"action": "done"}], [])[0]
    for phrase in ("do this for me", "operate my screen", "operate my computer"):
        assert s.match(phrase) is not None, phrase
    assert s.match("what's on my screen") is None   # sensing stays see_screen


def test_task_extraction():
    s = _skill([{"action": "done"}], [])[0]
    assert s._task_from(SkillRequest(text="do this for me: open notepad")) == "open notepad"
    assert s._task_from(SkillRequest(text="x", args={"task": "click save"})) == "click save"


def test_default_act_handles_key_list_and_string(monkeypatch):
    import skills.screen_agent as sa

    pressed = []

    class _Gui:
        def press(self, k): pressed.append(("press", k))
        def hotkey(self, *ks): pressed.append(("hotkey", ks))
        def click(self, x, y): pressed.append(("click", x, y))
        def scroll(self, n): pressed.append(("scroll", n))
        def write(self, t, interval=0): pressed.append(("write", t))

    monkeypatch.setattr("skills.desktop._pyautogui", lambda: _Gui())
    act = sa.ScreenAgentSkill._default_act
    act({"action": "key", "keys": ["win"]}, 2560, 1440)          # list form
    act({"action": "key", "keys": "ctrl a"}, 2560, 1440)          # string chord
    assert ("press", "win") in pressed
    assert ("hotkey", ("ctrl", "a")) in pressed
