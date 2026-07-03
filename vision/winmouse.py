"""Thin Win32 mouse control via ctypes — no third-party deps in the sidecar.

pyautogui would add pillow plus a default 0.1 s pause per call; pointer mode
moves the cursor every camera frame (~15 fps), so raw user32 it is. The module
imports on any OS; the functions themselves need Windows (callers gate on
pointer mode, which only the Windows sidecar enables).
"""

from __future__ import annotations

import ctypes

_LEFTDOWN, _LEFTUP = 0x0002, 0x0004
_RIGHTDOWN, _RIGHTUP = 0x0008, 0x0010


def _user32():
    return ctypes.windll.user32  # AttributeError off-Windows — callers gate


def screen_size() -> tuple[int, int]:
    u = _user32()
    return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))


def move(x: int, y: int) -> None:
    _user32().SetCursorPos(int(x), int(y))


def click_left() -> None:
    u = _user32()
    u.mouse_event(_LEFTDOWN, 0, 0, 0, 0)
    u.mouse_event(_LEFTUP, 0, 0, 0, 0)


def click_right() -> None:
    u = _user32()
    u.mouse_event(_RIGHTDOWN, 0, 0, 0, 0)
    u.mouse_event(_RIGHTUP, 0, 0, 0, 0)
