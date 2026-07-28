"""Typing, key chords, window actions and volume — the rest of what acts.

Same principle as test_actuation_coverage: mock at the OS boundary and assert
on what was *invoked*. A key chord resolved wrongly is a keystroke sent to the
user's focused window, so the mapping deserves tests of its own rather than
being exercised incidentally.
"""

from __future__ import annotations

import pytest

from skills.base import SkillRequest
from skills.desktop import (
    PressKeysSkill,
    TypeTextSkill,
    WindowActionSkill,
    resolve_keys,
)
from skills.system import VolumeSkill


# --- key resolution ------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("control s", ["ctrl", "s"]),
    ("ctrl+shift+a", ["ctrl", "shift", "a"]),
    (["ctrl", "a"], ["ctrl", "a"]),
    ("f5", ["f5"]),
    ("escape", ["esc"]),
])
def test_resolve_keys_maps_spoken_chords(raw, expected):
    assert resolve_keys(raw) == expected


def test_two_word_aliases_beat_single_tokens():
    """"page down" is PageDown, not Page then the down arrow."""
    assert resolve_keys("page down") == ["pagedown"]
    assert resolve_keys("down") == ["down"]


def test_unknown_tokens_are_dropped_not_typed():
    """A dropped token is a no-op; a passed-through one is a stray keystroke."""
    assert resolve_keys("control banana s") == ["ctrl", "s"]
    assert resolve_keys("complete gibberish") == []


@pytest.mark.asyncio
async def test_press_keys_refuses_when_nothing_resolves(monkeypatch):
    pressed = []
    monkeypatch.setattr("skills.desktop._pyautogui",
                        lambda: type("G", (), {
                            "press": lambda self, k: pressed.append(k),
                            "hotkey": lambda self, *k: pressed.append(k)})())
    skill = PressKeysSkill()
    phrase = "press complete gibberish"
    result = await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert pressed == [] and result.success is False


# --- typing --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ascii_text_is_typed_directly(monkeypatch):
    typed = []
    monkeypatch.setattr("skills.desktop._pyautogui",
                        lambda: type("G", (), {
                            "write": lambda self, body, interval=0: typed.append(body)})())
    skill = TypeTextSkill()
    phrase = "type hello world"
    result = await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert typed == ["hello world"] and result.success


@pytest.mark.asyncio
async def test_cyrillic_goes_through_the_clipboard(monkeypatch):
    """pyautogui.write silently drops non-ASCII — Cyrillic must be pasted."""
    written, copied, chords = [], [], []
    monkeypatch.setattr("skills.desktop._pyautogui",
                        lambda: type("G", (), {
                            "write": lambda self, b, interval=0: written.append(b),
                            "hotkey": lambda self, *k: chords.append(k)})())
    import sys
    fake = type(sys)("pyperclip")
    fake.paste = lambda: "previous clipboard"
    fake.copy = lambda text: copied.append(text)
    monkeypatch.setitem(sys.modules, "pyperclip", fake)

    skill = TypeTextSkill()
    phrase = "type здраво свету"
    await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert written == [], "Cyrillic must not go through write()"
    assert "здраво свету" in copied
    assert copied[-1] == "previous clipboard", "the clipboard must be restored"
    assert chords, "nothing was pasted"


@pytest.mark.asyncio
async def test_typing_nothing_is_refused():
    skill = TypeTextSkill()
    result = await skill.execute(SkillRequest(text="", args={}))
    assert result.success is False


# --- window actions ------------------------------------------------------------


@pytest.mark.parametrize("phrase,action", [
    ("minimize the window", "minimize"),
    ("maximize the window", "maximize"),
    ("close the window", "close"),
])
@pytest.mark.asyncio
async def test_window_actions_send_the_mapped_chord(monkeypatch, phrase, action):
    chords = []
    monkeypatch.setattr("skills.desktop._pyautogui",
                        lambda: type("G", (), {
                            "hotkey": lambda self, *k: chords.append(k),
                            "press": lambda self, k: chords.append((k,))})())
    skill = WindowActionSkill()
    match = skill.match(phrase)
    assert match is not None, phrase
    result = await skill.execute(SkillRequest(text=phrase, match=match))
    assert result.success and chords, f"{action} sent nothing"
    assert tuple(skill.ACTIONS[action]) == chords[0]


def test_close_the_window_is_not_an_app_launch():
    """WindowActionSkill registers before AppsSkill for exactly this reason."""
    assert WindowActionSkill().match("close the window") is not None


# --- volume --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_absolute_volume_is_clamped(monkeypatch):
    levels = []
    monkeypatch.setattr("skills.system.IS_WINDOWS", True)
    monkeypatch.setattr("skills.system.IS_MACOS", False)
    monkeypatch.setattr("skills.system._win_set_volume",
                        lambda level: levels.append(level) or True)
    skill = VolumeSkill()
    for phrase, expected in (("set volume to 50", 50), ("set volume to 300", 100)):
        await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
        assert levels[-1] == expected, f"{phrase!r} set {levels[-1]}, not {expected}"
    assert levels == [50, 100], "a level above 100 must be clamped, not passed on"


@pytest.mark.asyncio
async def test_volume_up_steps_from_the_current_level(monkeypatch):
    levels = []
    monkeypatch.setattr("skills.system.IS_WINDOWS", True)
    monkeypatch.setattr("skills.system.IS_MACOS", False)
    monkeypatch.setattr("skills.system._current_volume", lambda: 40)
    monkeypatch.setattr("skills.system._win_set_volume",
                        lambda level: levels.append(level) or True)
    skill = VolumeSkill()
    phrase = "volume up"
    await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert levels == [50]


@pytest.mark.asyncio
async def test_mute_and_unmute_are_distinguished(monkeypatch):
    states = []
    monkeypatch.setattr("skills.system.IS_WINDOWS", True)
    monkeypatch.setattr("skills.system.IS_MACOS", False)
    monkeypatch.setattr("skills.system._win_set_muted",
                        lambda muted: states.append(muted) or True)
    skill = VolumeSkill()
    for phrase in ("mute", "unmute"):
        await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert states == [True, False]
