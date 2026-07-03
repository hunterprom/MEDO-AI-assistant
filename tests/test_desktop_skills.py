"""Desktop skills: key resolution and action wiring with GUI libs stubbed."""

from __future__ import annotations

import sys
import types

import pytest

import skills.desktop as desktop
from skills.base import SkillRequest
from skills.desktop import (
    ClipboardSkill,
    PressKeysSkill,
    TypeTextSkill,
    WindowActionSkill,
    resolve_keys,
)


def test_resolve_keys_names_modifiers_and_unknowns():
    assert resolve_keys("control shift a") == ["ctrl", "shift", "a"]
    assert resolve_keys("windows d") == ["win", "d"]
    assert resolve_keys(["ALT", "F4"]) == ["alt", "f4"]
    assert resolve_keys("page down") == ["pagedown"]  # two-word alias, not ↓
    assert resolve_keys("ctrl page down") == ["ctrl", "pagedown"]
    assert resolve_keys("volume up") == ["volumeup"]
    assert resolve_keys("banana") == []


def _req(skill, text):
    """Build a SkillRequest the way the router does: with the regex match."""
    return SkillRequest(text=text, match=skill.match(text))


class _GuiStub(types.SimpleNamespace):
    def __init__(self):
        super().__init__(FAILSAFE=True, calls=[])

    def write(self, text, interval=0.0):
        self.calls.append(("write", text))

    def hotkey(self, *keys):
        self.calls.append(("hotkey", keys))

    def press(self, key):
        self.calls.append(("press", key))


@pytest.fixture
def gui(monkeypatch):
    stub = _GuiStub()
    monkeypatch.setitem(sys.modules, "pyautogui", stub)
    return stub


@pytest.fixture
def clip(monkeypatch):
    stub = types.SimpleNamespace(buffer="old clipboard")

    def copy(text):
        stub.buffer = text

    def paste():
        return stub.buffer

    stub.copy, stub.paste = copy, paste
    monkeypatch.setitem(sys.modules, "pyperclip", stub)
    return stub


async def test_type_ascii_uses_write(gui):
    skill = TypeTextSkill()
    r = await skill.execute(_req(skill, "type hello world"))
    assert r.success and ("write", "hello world") in gui.calls
    assert gui.FAILSAFE is False  # fail-safe must be off (pointer parks corners)


async def test_type_cyrillic_pastes_and_restores_clipboard(gui, clip):
    skill = TypeTextSkill()
    r = await skill.execute(_req(skill, "type Здраво Ана"))
    assert r.success
    assert ("hotkey", ("ctrl", "v")) in gui.calls
    assert not any(c[0] == "write" for c in gui.calls)
    assert clip.buffer == "old clipboard"  # restored after the paste


async def test_press_single_and_chord(gui):
    skill = PressKeysSkill()
    r = await skill.execute(_req(skill, "press enter"))
    assert r.success and ("press", "enter") in gui.calls
    r = await skill.execute(_req(skill, "press control shift t"))
    assert r.success and ("hotkey", ("ctrl", "shift", "t")) in gui.calls


async def test_press_unknown_keys_graceful(gui):
    skill = PressKeysSkill()
    r = await skill.execute(_req(skill, "press flux capacitor"))
    assert r.success is False


async def test_window_actions_map_to_chords(gui):
    cases = {
        "minimize the window": ("win", "down"),
        "maximize the window": ("win", "up"),
        "close the window": ("alt", "f4"),
        "show the desktop": ("win", "d"),
        "switch apps": ("alt", "tab"),
    }
    for text, chord in cases.items():
        gui.calls.clear()
        r = await WindowActionSkill().execute(SkillRequest(text=text))
        assert r.success and ("hotkey", chord) in gui.calls, text


async def test_clipboard_read_and_write(gui, clip):
    skill = ClipboardSkill()
    r = await skill.execute(_req(skill, "read my clipboard"))
    assert r.success and "old clipboard" in r.speech

    r = await skill.execute(_req(skill, "copy hello there to clipboard"))
    assert r.success and clip.buffer == "hello there"

    clip.copy("")
    r = await skill.execute(_req(skill, "read my clipboard"))
    assert "empty" in r.speech.lower()


async def test_brightness_not_windows_is_graceful(monkeypatch):
    monkeypatch.setattr(desktop, "IS_WINDOWS", False)
    r = await desktop.BrightnessSkill().execute(
        SkillRequest(text="set brightness to 40")
    )
    assert r.success is False and "OS" in r.speech
