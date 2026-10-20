"""TEMPLATE connector — copy this file to add a new app. Adding an app is a NEW
FILE, never a core edit.

Shows every part: detection, the mechanism ladder, declared capabilities, and —
importantly — the confirm-before-send rule. ANY action that SENDS or POSTS
something on the user's behalf sets ``requires_confirmation=True``; MEDO never
auto-sends without a yes. Not registered by default (it's an example); add your
real connector class to ``software/connectors/__init__.py``'s CONNECTOR_CLASSES.
"""

from __future__ import annotations

from typing import List

from security.capabilities import Capability
from software.connector_base import (
    API,
    CLI,
    HOTKEY,
    Action,
    ActionResult,
    DetectResult,
    SoftwareConnector,
)


class TemplateConnector(SoftwareConnector):
    app_id = "myapp"
    display_name = "MyApp"
    executables = ("myapp.exe",)
    window_hint = "MyApp"

    def detect(self) -> DetectResult:
        return DetectResult(
            installed=self._mech.is_installed(self.executables),
            running=self._mech.is_process_running(self.executables))

    def actions(self) -> List[Action]:
        m = self._mech
        return [
            # Mechanism 1 (native local API) — preferred, survives UI changes.
            Action("do_thing", "Do the main thing in MyApp.",
                   (r"\bin MyApp do the thing\b",), (API, HOTKEY),
                   frozenset({Capability.CONTROL_INPUT}),
                   lambda p: m.send_app_hotkey(self.window_hint, "ctrl+d")),
            # A CLI action — the adapter forces it through the run_command policy.
            Action("run_task", "Run MyApp's export task.",
                   (r"\bexport in MyApp\b",), (CLI,),
                   frozenset({Capability.RUN_COMMAND}),
                   lambda p: m.run_cli("myapp", ["myapp", "--export"],
                                       frozenset({Capability.RUN_COMMAND})),
                   requires_confirmation=True),
            # SEND/POST — ALWAYS confirms first. MEDO never sends on your behalf
            # without a yes.
            Action("send_message", "Send a message in MyApp.",
                   (r"\bsend .* in MyApp\b",), (API,),
                   frozenset({Capability.NETWORK}),
                   self._send, requires_confirmation=True,
                   confirm_prompt="Do you want me to send that message? Say yes.",
                   params={"text": {"type": "string",
                                    "description": "the message to send"}}),
        ]

    def _send(self, params) -> ActionResult:
        text = params.get("text", "")
        return self._mech.call_local_api(
            "myapp", "POST", "http://127.0.0.1:9999/send",
            frozenset({Capability.NETWORK}), json={"text": text})
