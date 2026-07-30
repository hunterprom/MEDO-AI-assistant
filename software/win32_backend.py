"""Real Windows OS backend for the connectors (integration — not unit-tested).

Best-effort, lazy-imported, and defensive: every method degrades to a safe
default if its dependency is missing or a call fails, so a connector never
crashes the engine — it just reports it couldn't do the thing. Uses psutil
(processes), pygetwindow (windows), and a keyboard sender (pyautogui/keyboard)
where available.

Reuses MEDO's existing win32 story; on non-Windows it simply reports nothing is
available (connectors then say so honestly).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

# Virtual key codes for the global media keys.
_MEDIA_VK = {
    "play_pause": 0xB3, "next": 0xB0, "previous": 0xB1,
    "volume_up": 0xAF, "volume_down": 0xAE, "mute": 0xAD,
}


class Win32Backend:
    def running_processes(self) -> Sequence[str]:
        try:
            import psutil
            return [p.info["name"].lower() for p in psutil.process_iter(["name"])
                    if p.info.get("name")]
        except Exception:
            return []

    def is_installed(self, exe: str) -> bool:
        if shutil.which(exe):
            return True
        # common install roots
        for root in filter(None, (os.environ.get("ProgramFiles"),
                                   os.environ.get("ProgramFiles(x86)"),
                                   os.environ.get("LOCALAPPDATA"))):
            for dirpath, _dirs, files in os.walk(root):
                if exe.lower() in {f.lower() for f in files}:
                    return True
                # don't descend forever — one or two levels is enough for a probe
                if dirpath.count(os.sep) - root.count(os.sep) >= 3:
                    _dirs[:] = []
        return exe.lower() in {p.lower() for p in self.running_processes()}

    def find_window(self, title_substr: str) -> Optional[object]:
        if not title_substr:
            return None
        try:
            import pygetwindow as gw
            for w in gw.getAllWindows():
                if title_substr.lower() in (w.title or "").lower():
                    return w
        except Exception:
            logger.debug("find_window unavailable", exc_info=True)
        return None

    def active_window(self) -> Optional[object]:
        try:
            import pygetwindow as gw
            return gw.getActiveWindow()
        except Exception:
            return None

    def focus_window(self, handle: object) -> bool:
        try:
            handle.activate()
            return True
        except Exception:
            try:
                handle.minimize(); handle.restore()
                return True
            except Exception:
                return False

    def send_keys(self, keys: str) -> bool:
        try:
            import pyautogui
            pyautogui.hotkey(*[k.strip().lower() for k in keys.split("+")])
            return True
        except Exception:
            logger.debug("send_keys unavailable", exc_info=True)
            return False

    def send_media_key(self, key: str) -> bool:
        vk = _MEDIA_VK.get(key)
        if vk is None:
            return False
        try:
            import ctypes
            KEYEVENTF_KEYUP = 0x0002
            ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
            ctypes.windll.user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
            return True
        except Exception:
            return False

    def launch(self, command: List[str]) -> bool:
        try:
            subprocess.Popen(list(command))
            return True
        except Exception:
            logger.warning("launch failed: %s", command, exc_info=True)
            return False

    def close_window(self, handle: object) -> bool:
        try:
            handle.close()
            return True
        except Exception:
            return False

    def foreground_rect(self) -> Optional[tuple]:
        try:
            import pygetwindow as gw
            w = gw.getActiveWindow()
            if w is None:
                return None
            return (int(w.left), int(w.top), int(w.right), int(w.bottom))
        except Exception:
            return None

    def foreground_title(self) -> Optional[str]:
        try:
            import pygetwindow as gw
            w = gw.getActiveWindow()
            return (w.title or "") if w is not None else None
        except Exception:
            return None

    def click_point(self, x: int, y: int) -> bool:
        try:
            import pyautogui
            pyautogui.click(int(x), int(y))
            return True
        except Exception:
            logger.debug("click_point unavailable", exc_info=True)
            return False
