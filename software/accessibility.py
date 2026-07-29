"""UI-automation adapter — the BRITTLE bottom rung of the control ladder.

Reuses the M7 pointer-snap approach (vision/snap.py): query the OS accessibility
tree for a named control and act on it. Uses ``uiautomation`` (win32), the same
library the snap feature relies on, and degrades gracefully everywhere — no
library, no window, no control, or any error returns a clean "can't reach that
app's controls" ActionResult, never an exception. A connector reaches for this
ONLY when it has no native API and no usable hotkey.
"""

from __future__ import annotations

import logging

from software.connector_base import ActionResult

logger = logging.getLogger(__name__)

# Same clickable allowlist M7 uses — acting on a Pane/Text does nothing useful.
CLICKABLE_ROLES = frozenset({
    "Button", "Hyperlink", "CheckBox", "RadioButton", "MenuItem", "TabItem",
    "ListItem", "ComboBox", "SplitButton", "TreeItem",
})


def available() -> bool:
    try:
        import uiautomation  # noqa: F401
        return True
    except Exception:
        return False


def invoke_control(_os_backend, window_hint: str, control_name: str,
                   action: str = "invoke") -> ActionResult:
    """Find ``control_name`` inside the app window and invoke/toggle it. Best
    effort; any failure is reported honestly, not faked."""
    try:
        import uiautomation as auto
    except Exception:
        return ActionResult(False, "I can't reach that app's controls on this "
                                   "computer.")
    try:
        win = auto.WindowControl(searchDepth=1, SubName=window_hint)
        if not win.Exists(maxSearchSeconds=1.0):
            return ActionResult(False, "I couldn't find that app's window.")
        ctrl = win.Control(searchDepth=20, Name=control_name)
        if not ctrl.Exists(maxSearchSeconds=1.0):
            return ActionResult(False, f"I couldn't find '{control_name}' in "
                                       f"that app.")
        pattern = ctrl.GetInvokePattern() if action == "invoke" else None
        if pattern is not None:
            pattern.Invoke()
            return ActionResult(True, "Done.", verified=True)
        # No invoke pattern -> click its center as a fallback.
        ctrl.Click()
        return ActionResult(True, "Done.", verified=False)
    except Exception:
        logger.warning("ui automation failed on %s/%s", window_hint,
                       control_name, exc_info=True)
        return ActionResult(False, "I couldn't do that in the app.")
