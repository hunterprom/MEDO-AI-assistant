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


# The chords are platform-dependent (Bug Log #24: the v1 mapping was
# Windows-only, dead keys on a Mac) — tests assert THIS platform's mapping.
IS_MAC = sys.platform == "darwin"
SUPER = "command" if IS_MAC else "win"       # spoken "windows"/"command"
PASTE = ("command", "v") if IS_MAC else ("ctrl", "v")


def test_resolve_keys_names_modifiers_and_unknowns():
    assert resolve_keys("control shift a") == ["ctrl", "shift", "a"]
    assert resolve_keys("windows d") == [SUPER, "d"]
    assert resolve_keys("command d") == [SUPER, "d"]  # either name, same key
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
    assert ("hotkey", PASTE) in gui.calls  # Cmd+V on mac — Ctrl+V is a no-op there
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
        "minimize the window": ("command", "m") if IS_MAC else ("win", "down"),
        "maximize the window": ("ctrl", "command", "f") if IS_MAC else ("win", "up"),
        "close the window": ("command", "w") if IS_MAC else ("alt", "f4"),
        "show the desktop": ("f11",) if IS_MAC else ("win", "d"),
        "switch apps": ("command", "tab") if IS_MAC else ("alt", "tab"),
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


# -- press_keys: don't silently swallow the rest of the sentence -------------

def test_press_reports_words_it_ignored():
    """"press control s and close the window" saved the file and dropped the
    rest with no signal — it looked like the whole request had run."""
    from skills.desktop import parse_key_request
    keys, ignored, sequential = parse_key_request("control s and close the window")
    assert keys == ["ctrl", "s"] and sequential is False
    assert "close" in ignored and "window" in ignored


def test_then_means_a_sequence_not_a_chord():
    from skills.desktop import parse_key_request
    keys, ignored, sequential = parse_key_request("a and then b")
    assert keys == ["a", "b"] and sequential is True and ignored == []


def test_plain_chord_is_unchanged():
    from skills.desktop import parse_key_request, resolve_keys
    keys, ignored, sequential = parse_key_request("ctrl shift a")
    assert keys == ["ctrl", "shift", "a"] and not ignored and sequential is False
    assert resolve_keys("page down") == ["pagedown"]      # old API still works
