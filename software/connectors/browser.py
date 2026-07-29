"""Browser connector — tab control via app hotkeys (new/next tab).

Mechanism 2: app-specific hotkeys sent to the FOCUSED browser window (Ctrl+T,
Ctrl+Tab) — the "focus target → act → restore focus" flow. Opening a URL is left
to MEDO's existing web-open skill (which already enforces the user-given-URL
injection guard); this connector demonstrates hotkey control cleanly.
"""

from __future__ import annotations

from typing import List

from security.capabilities import Capability
from software.connector_base import HOTKEY, Action, DetectResult, SoftwareConnector

_CTRL = frozenset({Capability.CONTROL_INPUT})
_BROWSERS = ("chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe")


class BrowserConnector(SoftwareConnector):
    app_id = "browser"
    display_name = "browser"
    executables = _BROWSERS
    window_hint = "- Google Chrome"       # overridden by whichever browser is found

    def detect(self) -> DetectResult:
        installed = self._mech.is_installed(_BROWSERS)
        running = self._mech.is_process_running(_BROWSERS)
        return DetectResult(installed=installed, running=running)

    def _hint(self) -> str:
        # Match any browser window; the mechanism does a substring find.
        return "Mozilla Firefox" if self._mech.is_process_running(("firefox.exe",)) \
            else "Chrome"

    def actions(self) -> List[Action]:
        m = self._mech
        return [
            Action("new_tab", "Open a new browser tab.",
                   (r"\bnew tab\b", r"\bотвори (?:ново )?јазиче\b"),
                   (HOTKEY,), _CTRL,
                   lambda p: m.send_app_hotkey(self._hint(), "ctrl+t")),
            Action("next_tab", "Switch to the next browser tab.",
                   (r"\bnext tab\b", r"\bследно јазиче\b"),
                   (HOTKEY,), _CTRL,
                   lambda p: m.send_app_hotkey(self._hint(), "ctrl+tab")),
        ]
