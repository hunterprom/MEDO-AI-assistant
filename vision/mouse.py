"""Cross-platform mouse backend for pointer mode — one surface, three impls.

    Windows  vision/winmouse.py   raw user32 via ctypes
    macOS    vision/macmouse.py   raw CoreGraphics via ctypes (+ osascript)
    other    vision/anymouse.py   pyautogui (pause/fail-safe disabled)

The engine imports THIS module lazily and calls ``screen_size()`` once; any
ImportError/AttributeError surfaces there and cleanly disables pointer mode
with a logged reason, exactly as the old Windows-only path did. Every backend
exposes: screen_size, move, click_left, click_right, press_left,
release_left, scroll, zoom, volume_up, volume_down, play_pause.
"""

from __future__ import annotations

import sys

if sys.platform == "win32":
    from vision.winmouse import *  # noqa: F401,F403
elif sys.platform == "darwin":
    from vision.macmouse import *  # noqa: F401,F403
else:
    from vision.anymouse import *  # noqa: F401,F403
