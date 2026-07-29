"""Shared control mechanisms (S2) — the ladder, built once, reused by connectors.

The OS layer is INJECTED (``OSBackend``), so the "focus target → act → restore
focus" flow, the CLI/API policy gating, and graceful failure are all unit-tested
with a fake backend — no real windows. Connectors just declare which mechanism to
use; they never touch the OS directly.

Non-negotiable: the CLI adapter and the local-API adapter route through the
central policy engine (run_command / network capabilities). The window + hotkey +
media mechanisms rely on the skill-level ``control_input`` gate the router already
applies before the action runs.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Protocol, Sequence

from security.capabilities import Capability
from security.policy import ActionRequest, Actor, PolicyEngine
from software.connector_base import ActionResult

logger = logging.getLogger(__name__)


class OSBackend(Protocol):
    """The desktop operations mechanisms need. A real Win32 impl lives below;
    tests inject a fake."""

    def find_window(self, title_substr: str) -> Optional[object]: ...
    def active_window(self) -> Optional[object]: ...
    def focus_window(self, handle: object) -> bool: ...
    def send_keys(self, keys: str) -> bool: ...            # to the focused window
    def send_media_key(self, key: str) -> bool: ...        # global media key
    def running_processes(self) -> Sequence[str]: ...      # lowercased exe names
    def is_installed(self, exe: str) -> bool: ...          # on PATH / common dirs
    def launch(self, command: List[str]) -> bool: ...
    def close_window(self, handle: object) -> bool: ...


class Mechanisms:
    def __init__(self, backend: OSBackend, policy: PolicyEngine, *,
                 exe_paths: Optional[Dict[str, list]] = None,
                 actor: Actor = Actor.LOCAL_USER) -> None:
        self._os = backend
        self._policy = policy
        self._exe_paths = exe_paths or {}
        self._actor = actor

    # -- process / window lifecycle -------------------------------------------

    def is_process_running(self, exe_names: Sequence[str]) -> bool:
        procs = {p.lower() for p in self._os.running_processes()}
        return any(e.lower() in procs for e in exe_names)

    def is_installed(self, exe_names: Sequence[str]) -> bool:
        return any(self._os.is_installed(e) for e in exe_names)

    def launch(self, app_id: str) -> bool:
        cmd = self._exe_paths.get(app_id)
        if not cmd:
            return False
        return self._os.launch(cmd if isinstance(cmd, list) else [cmd])

    def focus_app(self, window_hint: str) -> bool:
        win = self._os.find_window(window_hint)
        return bool(win) and self._os.focus_window(win)

    # -- mechanism 2: app-specific hotkey to the FOCUSED window ---------------

    def send_app_hotkey(self, window_hint: str, keys: str) -> ActionResult:
        """Focus the target window, send the combo, then restore the previous
        focus politely. Honest about whether it landed."""
        prev = self._os.active_window()
        win = self._os.find_window(window_hint)
        if win is None:
            return ActionResult(False, "I couldn't find that app's window.")
        if not self._os.focus_window(win):
            return ActionResult(False, "I couldn't bring that app to the front.")
        ok = self._os.send_keys(keys)
        restored = True
        if prev is not None and prev is not win:
            restored = self._os.focus_window(prev)      # give focus back
        return ActionResult(
            ok, "Done." if ok else "That key didn't go through.",
            verified=ok, data={"restored_focus": restored})

    # -- media keys (global; can't verify the effect) -------------------------

    def send_media_key(self, key: str) -> ActionResult:
        ok = self._os.send_media_key(key)
        # A media key is global — we sent it, but can't confirm the app reacted.
        return ActionResult(ok, "Okay." if ok else "That didn't go through.",
                            verified=False)

    # -- mechanism 1a: CLI adapter (ALWAYS through the policy engine) ----------

    def run_cli(self, subject: str, command: List[str],
                declared: frozenset) -> ActionResult:
        decision = self._policy.check(ActionRequest(
            actor=self._actor, capability=Capability.RUN_COMMAND,
            subject=subject, declared=declared))
        if decision.denied():
            return ActionResult(False, decision.reason)
        # NEVER a raw shell — a concrete argv list only.
        ok = self._os.launch(list(command))
        return ActionResult(ok, "Done." if ok else "That command didn't run.",
                            verified=ok)

    # -- mechanism 1b: local API adapter (localhost, policy-gated) -------------

    def call_local_api(self, subject: str, method: str, url: str,
                       declared: frozenset, *, timeout: float = 3.0,
                       json: Optional[dict] = None) -> ActionResult:
        decision = self._policy.check(ActionRequest(
            actor=self._actor, capability=Capability.NETWORK,
            subject=subject, target=url, declared=declared))
        if decision.denied():
            return ActionResult(False, decision.reason)
        if not (url.startswith("http://127.0.0.1") or url.startswith("http://localhost")):
            return ActionResult(False, "That connector may only talk to the app "
                                       "on this machine.")
        try:
            import httpx
            r = httpx.request(method, url, timeout=timeout, json=json)
            ok = r.status_code < 400
            return ActionResult(ok, "Done." if ok else "The app refused that.",
                                verified=ok, data={"status": r.status_code})
        except Exception:
            logger.warning("local API call failed: %s", url, exc_info=True)
            return ActionResult(False, "The app didn't respond.")

    # -- mechanism 3: UI automation (accessibility tree — brittle fallback) ---

    def ui_automation(self, window_hint: str, control_name: str,
                      action: str = "invoke") -> ActionResult:
        """Reuses the M7 pointer-snap accessibility work to find + act on a named
        control. Marked BRITTLE — the last rung of the ladder; a connector uses it
        only when no API/hotkey exists."""
        try:
            from software import accessibility
        except Exception:
            return ActionResult(False, "I can't reach that app's controls on "
                                       "this system.")
        return accessibility.invoke_control(self._os, window_hint, control_name,
                                            action)
