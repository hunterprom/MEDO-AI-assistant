"""Window-manager connector — focus / minimize / close an app's window.

Mechanism: process + window control (find/focus/close). ``close`` is the one
high-impact action here — it can lose unsaved work — so it REQUIRES confirmation
(bilingual). Focus/minimize are low-impact control_input.
"""

from __future__ import annotations

from typing import List

from security.capabilities import Capability
from software.connector_base import (
    ACCESSIBILITY,
    HOTKEY,
    Action,
    ActionResult,
    DetectResult,
    SoftwareConnector,
)

_CTRL = frozenset({Capability.CONTROL_INPUT})


class WindowConnector(SoftwareConnector):
    app_id = "window"
    display_name = "the current window"

    def detect(self) -> DetectResult:
        return DetectResult(installed=True, running=True, detail="window manager")

    def is_running(self) -> bool:
        return True

    def launch(self) -> bool:
        return True

    def actions(self) -> List[Action]:
        m = self._mech
        return [
            Action("focus", "Bring an app's window to the front.",
                   (r"\b(focus|switch to|bring up|go to)\s+(?P<app>.+)",
                    r"\bпрефрли се на\b"),
                   (HOTKEY,), _CTRL, self._focus,
                   params={"app": {"type": "string",
                                   "description": "which app/window to focus"}}),
            Action("minimize", "Minimize the current window.",
                   (r"\bminimi[sz]e\b", r"\bминимизирај\b"),
                   (HOTKEY,), _CTRL,
                   lambda p: m.send_global_hotkey("win+down")),
            Action("close_window", "Close an app's window.",
                   (r"\bclose\s+(?P<app>.+)", r"\bзатвори\b"),
                   (HOTKEY, ACCESSIBILITY), _CTRL, self._close,
                   requires_confirmation=True,
                   confirm_prompt=("Closing it may lose anything unsaved. "
                                   "Close it? Say yes to confirm."),
                   params={"app": {"type": "string",
                                   "description": "which app/window to close"}}),
        ]

    def _focus(self, params) -> ActionResult:
        hint = (params.get("app") or "").strip()
        if not hint:
            return ActionResult(False, "Which app should I focus?")
        ok = self._mech.focus_app(hint)
        return ActionResult(ok, "Done." if ok else f"I couldn't find {hint}.",
                            verified=ok)

    def _close(self, params) -> ActionResult:
        hint = (params.get("app") or "").strip()
        # Alt+F4 to the focused target window (after focusing it).
        return self._mech.send_app_hotkey(hint, "alt+F4")
