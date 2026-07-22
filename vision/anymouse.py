"""Generic mouse backend via pyautogui — Linux and anything else.

Last-resort implementation of the :mod:`vision.mouse` surface for platforms
without a native ctypes backend. pyautogui's per-call pause and fail-safe
corner trap are disabled (the corner trap would kill pointer mode the moment
the cursor reaches a screen corner — which sensitivity remapping does
constantly). If pyautogui isn't installed in the sidecar venv this module
fails to import, and the engine's existing gate reports pointer mode
unavailable instead of crashing.
"""

from __future__ import annotations

import logging

import pyautogui

logger = logging.getLogger(__name__)

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False


def screen_size() -> tuple[int, int]:
    size = pyautogui.size()
    return int(size.width), int(size.height)


def screen_bounds() -> tuple[int, int, int, int]:
    """Virtual-desktop bounds (x0, y0, w, h). pyautogui only knows the primary
    display, so this is the primary box — multi-monitor spanning needs a native
    backend (win/mac). Kept so the engine can call screen_bounds() uniformly."""
    w, h = screen_size()
    return 0, 0, w, h


def move(x: int, y: int) -> None:
    pyautogui.moveTo(int(x), int(y), _pause=False)


def click_left() -> None:
    pyautogui.click(_pause=False)


def click_right() -> None:
    pyautogui.rightClick(_pause=False)


def press_left() -> None:
    """Hold the left button down (for click-and-drag)."""
    pyautogui.mouseDown(_pause=False)


def release_left() -> None:
    """Release the left button (ends a drag; a quick press+release is a click)."""
    pyautogui.mouseUp(_pause=False)


def scroll(notches: int) -> None:
    if notches:
        pyautogui.scroll(int(notches) * 3, _pause=False)


def zoom(notches: int) -> None:
    """Ctrl + wheel — the near-universal zoom shortcut off-macOS."""
    if not notches:
        return
    with pyautogui.hold("ctrl"):
        pyautogui.scroll(int(notches) * 3, _pause=False)


def volume_up() -> None:
    _media_key("volumeup")


def volume_down() -> None:
    _media_key("volumedown")


def play_pause() -> None:
    _media_key("playpause")


def _media_key(key: str) -> None:
    try:
        pyautogui.press(key, _pause=False)
    except Exception:  # key name unsupported on this platform — non-fatal
        logger.debug("media key %r unavailable", key, exc_info=True)

def next_tab() -> None:
    pyautogui.hotkey("ctrl", "tab")


def prev_tab() -> None:
    pyautogui.hotkey("ctrl", "shift", "tab")


def switch_window() -> None:
    pyautogui.hotkey("alt", "tab")


def taskbar() -> None:
    """No portable "focus the taskbar" chord on Linux desktops; the super key
    opens the activities/app overview on GNOME and KDE, which is the closest
    equivalent and at least reachable."""
    pyautogui.press("super")


def show_desktop() -> None:
    pyautogui.hotkey("super", "d")
