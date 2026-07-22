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
_WHEEL = 0x0800                 # MOUSEEVENTF_WHEEL
_WHEEL_DELTA = 120             # one notch
_KEYUP = 0x0002                # KEYEVENTF_KEYUP
_VK_CONTROL = 0x11
_VK_VOLUME_DOWN, _VK_VOLUME_UP = 0xAE, 0xAF
_VK_MEDIA_PLAY_PAUSE = 0xB3
_VK_TAB = 0x09
_VK_SHIFT = 0x10
_VK_MENU = 0x12          # ALT
_VK_LWIN = 0x5B
_VK_T = 0x54
_VK_D = 0x44


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


def press_left() -> None:
    """Hold the left button down (for click-and-drag)."""
    _user32().mouse_event(_LEFTDOWN, 0, 0, 0, 0)


def release_left() -> None:
    """Release the left button (ends a drag; a quick press+release is a click)."""
    _user32().mouse_event(_LEFTUP, 0, 0, 0, 0)


def scroll(notches: int) -> None:
    """Turn the mouse wheel. Positive = up/away from the user, negative = down."""
    if notches:
        # mouse_event's wheel delta is signed; ctypes needs an unsigned c_ulong,
        # so wrap negatives into 32-bit two's complement.
        _user32().mouse_event(_WHEEL, 0, 0, ctypes.c_int(int(notches) * _WHEEL_DELTA).value & 0xFFFFFFFF, 0)


def zoom(notches: int) -> None:
    """Ctrl + wheel — the near-universal zoom shortcut (browsers, editors, maps)."""
    if not notches:
        return
    u = _user32()
    u.keybd_event(_VK_CONTROL, 0, 0, 0)
    try:
        scroll(notches)
    finally:
        u.keybd_event(_VK_CONTROL, 0, _KEYUP, 0)


def _tap(vk: int) -> None:
    u = _user32()
    u.keybd_event(vk, 0, 0, 0)
    u.keybd_event(vk, 0, _KEYUP, 0)


def _chord(*vks: int) -> None:
    """Press a key chord in order, release in reverse (so modifiers wrap)."""
    u = _user32()
    for vk in vks:
        u.keybd_event(vk, 0, 0, 0)
    for vk in reversed(vks):
        u.keybd_event(vk, 0, _KEYUP, 0)


def next_tab() -> None:
    """Ctrl+Tab — next tab in a browser, editor, or terminal."""
    _chord(_VK_CONTROL, _VK_TAB)


def prev_tab() -> None:
    """Ctrl+Shift+Tab — previous tab."""
    _chord(_VK_CONTROL, _VK_SHIFT, _VK_TAB)


def switch_window() -> None:
    """Alt+Tab — next window. A tap switches to the last one you used."""
    _chord(_VK_MENU, _VK_TAB)


def taskbar() -> None:
    """Win+T — move focus onto the taskbar.

    The point of this one for pointer mode: once focus is on the taskbar you
    can walk it with the arrow keys and open with Enter, so the taskbar
    becomes reachable by gesture without having to land the cursor on a 40 px
    icon.
    """
    _chord(_VK_LWIN, _VK_T)


def show_desktop() -> None:
    """Win+D — minimise everything (and restore on a second tap)."""
    _chord(_VK_LWIN, _VK_D)


def volume_up() -> None:
    _tap(_VK_VOLUME_UP)


def volume_down() -> None:
    _tap(_VK_VOLUME_DOWN)


def play_pause() -> None:
    """Media play/pause key — toggles Spotify/YouTube/whatever has media focus."""
    _tap(_VK_MEDIA_PLAY_PAUSE)
