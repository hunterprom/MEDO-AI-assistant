"""The screen agent's action executor and loop branches.

``_default_act`` is the piece that actually moves the mouse and presses keys,
translating a model's JSON into pyautogui calls. It is worth its own tests
because a wrong branch here is a real click at a real coordinate — and because
the model's output is untrusted input: it can name an action that doesn't
exist, or send keys as a list where a string was expected.
"""

from __future__ import annotations

import pytest

from core.config import load_settings
from skills.base import SkillRequest
from skills.screen_agent import ScreenAgentSkill, parse_action, scale_point


class _FakeGui:
    def __init__(self):
        self.calls: list[tuple] = []

    def click(self, x, y):
        self.calls.append(("click", x, y))

    def doubleClick(self, x, y):                     # noqa: N802 - pyautogui's name
        self.calls.append(("double", x, y))

    def write(self, text, interval=0):
        self.calls.append(("write", text))

    def scroll(self, amount):
        self.calls.append(("scroll", amount))

    def press(self, key):
        self.calls.append(("press", key))

    def hotkey(self, *keys):
        self.calls.append(("hotkey", *keys))


@pytest.fixture
def gui(monkeypatch):
    fake = _FakeGui()
    monkeypatch.setattr("skills.desktop._pyautogui", lambda: fake)
    return fake


def _act(action, w=1920, h=1080):
    return ScreenAgentSkill._default_act(action, w, h)


# --- the executor --------------------------------------------------------------


def test_click_scales_normalised_coordinates(gui):
    _act({"action": "click", "x": 500, "y": 500})
    assert gui.calls == [("click", 960, 540)]


def test_double_click_is_distinct_from_click(gui):
    _act({"action": "double_click", "x": 0, "y": 0})
    assert gui.calls == [("double", 0, 0)]


def test_type_writes_the_text(gui):
    _act({"action": "type", "text": "hello"})
    assert gui.calls == [("write", "hello")]


def test_scroll_inverts_so_positive_means_down(gui):
    _act({"action": "scroll", "amount": 3})
    assert gui.calls == [("scroll", -360)]


def test_a_single_key_is_pressed(gui):
    _act({"action": "key", "keys": "enter"})
    assert gui.calls == [("press", "enter")]


@pytest.mark.parametrize("keys", ["ctrl a", "ctrl+a", ["ctrl", "a"]])
def test_a_chord_is_sent_as_a_hotkey_however_the_model_spells_it(gui, keys):
    """The model returns chords as a string or a list, joined by space or +."""
    _act({"action": "key", "keys": keys})
    assert gui.calls == [("hotkey", "ctrl", "a")]


def test_an_unknown_action_does_nothing(gui):
    """An invented action must be ignored, not guessed at."""
    description = _act({"action": "teleport", "x": 5})
    assert gui.calls == [] and "ignored" in description


def test_empty_keys_press_nothing(gui):
    _act({"action": "key", "keys": "   "})
    assert gui.calls == []


# --- coordinate scaling --------------------------------------------------------


@pytest.mark.parametrize("x,y,expected", [
    (0, 0, (0, 0)),
    (1000, 1000, (1919, 1079)),           # clamped inside the screen
    (500, 500, (960, 540)),
    (-50, 2000, (0, 1079)),               # out-of-range values are clamped
])
def test_scale_point_clamps_into_the_screen(x, y, expected):
    assert scale_point(x, y, 1920, 1080) == expected


# --- parsing the model's reply -------------------------------------------------


def test_parse_action_accepts_bare_json():
    assert parse_action('{"action": "click", "x": 1}')["action"] == "click"


def test_parse_action_digs_json_out_of_prose():
    raw = 'Sure! Here is the step:\n{"action": "done", "say": "finished"}\nHope that helps.'
    assert parse_action(raw)["say"] == "finished"


@pytest.mark.parametrize("raw", ["", "no json here at all", "{not: valid json}", None])
def test_parse_action_returns_none_on_junk(raw):
    assert parse_action(raw) is None


# --- the loop ------------------------------------------------------------------


def _skill(replies, settings=None):
    """A screen agent whose capture and brain are stubbed."""
    settings = settings or load_settings()
    queue = list(replies)
    acted: list[dict] = []

    async def ask(image, task, history):
        return queue.pop(0) if queue else '{"action": "done", "say": "over"}'

    skill = ScreenAgentSkill(
        settings,
        capture=lambda: (object(), (1920, 1080)),
        ask=ask,
        act=lambda action, w, h: acted.append(action) or "did it",
    )
    skill.acted = acted
    return skill


async def _run(skill, text="do this for me: click around", confirmed=True):
    return await skill.execute(SkillRequest(text=text, match=skill.match(text),
                                            context={"confirmed": confirmed}))


@pytest.mark.asyncio
async def test_the_agent_asks_before_touching_the_screen():
    skill = _skill(['{"action": "done"}'])
    result = await _run(skill, confirmed=False)
    assert result.needs_confirmation is True and skill.acted == []


@pytest.mark.asyncio
async def test_the_agent_runs_steps_until_done():
    skill = _skill(['{"action": "click", "x": 1, "y": 2}',
                    '{"action": "done", "say": "Finished."}'])
    result = await _run(skill)
    assert result.success and result.speech == "Finished."
    assert len(skill.acted) == 1


@pytest.mark.asyncio
async def test_the_agent_stops_at_the_step_cap():
    settings = load_settings()
    settings.vision.agent_max_steps = 2
    skill = _skill(['{"action": "scroll"}'] * 5, settings)
    result = await _run(skill)
    assert result.data["steps"] == 2 and len(skill.acted) == 2


@pytest.mark.asyncio
async def test_an_unparseable_reply_stops_the_agent():
    skill = _skill(["I have no idea what to do"])
    result = await _run(skill)
    assert result.success is False and skill.acted == []


@pytest.mark.asyncio
async def test_the_agent_declines_when_disabled():
    settings = load_settings()
    settings.vision.agent_enabled = False
    skill = _skill(['{"action": "done"}'], settings)
    result = await _run(skill)
    assert result.success is False and skill.acted == []
