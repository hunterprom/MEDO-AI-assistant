"""Native macOS mouse control via ctypes on CoreGraphics — no third-party deps.

The macOS twin of :mod:`vision.winmouse`, same function surface, same
philosophy: the sidecar moves the cursor 15×/s, so raw OS calls beat pyautogui
(which would add pillow + a 0.1 s pause per call). CGEvents are created,
posted to the HID tap, and released (CFRelease — at 15 fps a leak would add
up fast).

macOS caveat: posting synthetic input needs the **Accessibility** permission
for whatever launched the sidecar (Terminal, run.command, the IDE) — System
Settings → Privacy & Security → Accessibility. Without it macOS silently
drops the events; :func:`screen_size` warns once so the sidecar log says why
"nothing moves" instead of leaving it a mystery.

Volume/media go through ``osascript`` (stdlib subprocess): the media-key HID
usages need AppKit, and AppleScript is rate-limited here to ~5 calls/s by the
gesture debouncers, so the ~60 ms it costs never touches the camera loop.
"""

from __future__ import annotations

import ctypes
import logging
import subprocess

logger = logging.getLogger(__name__)


class _CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


_cg = ctypes.CDLL(
    "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
)
_cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")

_cg.CGEventCreateMouseEvent.restype = ctypes.c_void_p
_cg.CGEventCreateMouseEvent.argtypes = [
    ctypes.c_void_p, ctypes.c_uint32, _CGPoint, ctypes.c_uint32,
]
# Varargs API: fixed prefix declared, per-wheel int32 deltas passed extra.
_cg.CGEventCreateScrollWheelEvent.restype = ctypes.c_void_p
_cg.CGEventCreateScrollWheelEvent.argtypes = [
    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
]
_cg.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p
_cg.CGEventCreateKeyboardEvent.argtypes = [
    ctypes.c_void_p, ctypes.c_uint16, ctypes.c_bool,
]
_cg.CGEventSetFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
_cg.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
_cg.CGEventCreate.restype = ctypes.c_void_p
_cg.CGEventCreate.argtypes = [ctypes.c_void_p]
_cg.CGEventGetLocation.restype = _CGPoint
_cg.CGEventGetLocation.argtypes = [ctypes.c_void_p]
_cg.CGMainDisplayID.restype = ctypes.c_uint32
_cg.CGDisplayPixelsWide.restype = ctypes.c_size_t
_cg.CGDisplayPixelsWide.argtypes = [ctypes.c_uint32]
_cg.CGDisplayPixelsHigh.restype = ctypes.c_size_t
_cg.CGDisplayPixelsHigh.argtypes = [ctypes.c_uint32]
_cg.AXIsProcessTrusted.restype = ctypes.c_bool
_cf.CFRelease.argtypes = [ctypes.c_void_p]

_HID_TAP = 0                       # kCGHIDEventTap
_LDOWN, _LUP, _RDOWN, _RUP = 1, 2, 3, 4
_MOVED, _LDRAG = 5, 6
_BTN_LEFT, _BTN_RIGHT = 0, 1
_UNIT_LINE = 1                     # kCGScrollEventUnitLine
_LINES_PER_NOTCH = 3               # match a Windows wheel notch's feel
_CMD_MASK = 1 << 20                # kCGEventFlagMaskCommand
_KEY_EQUAL, _KEY_MINUS = 0x18, 0x1B  # Cmd+'='/'-' = zoom in/out everywhere

#: Is the left button held (drag)? Drags must post LeftMouseDragged, not
#: MouseMoved, or drops/drags are ignored by most apps.
_left_down = False


def _post(event: int | None) -> None:
    if event:
        _cg.CGEventPost(_HID_TAP, event)
        _cf.CFRelease(event)


def _cursor() -> _CGPoint:
    """Current cursor position (clicks must land where the cursor is)."""
    probe = _cg.CGEventCreate(None)
    try:
        return _cg.CGEventGetLocation(probe)
    finally:
        if probe:
            _cf.CFRelease(probe)


def _mouse(etype: int, x: float, y: float, button: int = _BTN_LEFT) -> None:
    _post(_cg.CGEventCreateMouseEvent(
        None, etype, _CGPoint(float(x), float(y)), button))


def screen_size() -> tuple[int, int]:
    if not _cg.AXIsProcessTrusted():
        logger.warning(
            "macOS Accessibility permission missing — the cursor will NOT move. "
            "Grant it to the app that launches the sidecar (Terminal / "
            "run.command) in System Settings → Privacy & Security → "
            "Accessibility, then restart the sidecar."
        )
    display = _cg.CGMainDisplayID()
    return int(_cg.CGDisplayPixelsWide(display)), int(_cg.CGDisplayPixelsHigh(display))


def move(x: int, y: int) -> None:
    _mouse(_LDRAG if _left_down else _MOVED, x, y)


def click_left() -> None:
    press_left()
    release_left()


def click_right() -> None:
    at = _cursor()
    _mouse(_RDOWN, at.x, at.y, _BTN_RIGHT)
    _mouse(_RUP, at.x, at.y, _BTN_RIGHT)


def press_left() -> None:
    """Hold the left button down (for click-and-drag)."""
    global _left_down
    at = _cursor()
    _mouse(_LDOWN, at.x, at.y)
    _left_down = True


def release_left() -> None:
    """Release the left button (ends a drag; a quick press+release is a click)."""
    global _left_down
    at = _cursor()
    _mouse(_LUP, at.x, at.y)
    _left_down = False


def scroll(notches: int) -> None:
    """Turn the wheel. Positive = up/away from the user (CGEvent's convention too)."""
    if notches:
        _post(_cg.CGEventCreateScrollWheelEvent(
            None, _UNIT_LINE, 1,
            ctypes.c_int32(int(notches) * _LINES_PER_NOTCH)))


def _tap_key(keycode: int, flags: int = 0) -> None:
    for down in (True, False):
        event = _cg.CGEventCreateKeyboardEvent(None, keycode, down)
        if event and flags:
            _cg.CGEventSetFlags(event, flags)
        _post(event)


def zoom(notches: int) -> None:
    """Cmd+'='/'-' taps — macOS's near-universal zoom (browsers, editors, maps)."""
    key = _KEY_EQUAL if notches > 0 else _KEY_MINUS
    for _ in range(min(5, abs(int(notches)))):  # cap runaway accumulator bursts
        _tap_key(key, _CMD_MASK)


def _osascript(script: str) -> None:
    """Fire-and-forget AppleScript; never let it block or kill the camera loop."""
    try:
        subprocess.Popen(
            ["osascript", "-e", script],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        logger.debug("osascript failed", exc_info=True)


def volume_up() -> None:
    # Out-of-range values are pinned to 0..100 by macOS itself.
    _osascript("set volume output volume "
               "((output volume of (get volume settings)) + 6)")


def volume_down() -> None:
    _osascript("set volume output volume "
               "((output volume of (get volume settings)) - 6)")


def play_pause() -> None:
    """Toggle whichever player is running (Spotify first, then Music)."""
    _osascript(
        'if application "Spotify" is running then\n'
        '  tell application "Spotify" to playpause\n'
        'else if application "Music" is running then\n'
        '  tell application "Music" to playpause\n'
        "end if"
    )

def _key_script(keystroke: str) -> None:
    """Send a keystroke through osascript (same route as the media keys)."""
    import subprocess

    subprocess.run(["osascript", "-e",
                    f'tell application "System Events" to {keystroke}'],
                   check=False)


def next_tab() -> None:
    """Ctrl+Tab — next tab. Same chord as Windows in every mac browser."""
    _key_script('key code 48 using control down')


def prev_tab() -> None:
    _key_script('key code 48 using {control down, shift down}')


def switch_window() -> None:
    """Cmd+Tab — the mac app switcher (Alt+Tab's counterpart)."""
    _key_script('key code 48 using command down')


def taskbar() -> None:
    """Ctrl+F3 — move focus to the Dock, the mac's taskbar equivalent."""
    _key_script('key code 99 using control down')


def show_desktop() -> None:
    """F11 — show the desktop."""
    _key_script('key code 103')
