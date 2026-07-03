"""Open and close applications using the per-platform table in config.yaml.

The match patterns are built from the *configured* app names (plus a few common
aliases), so "open chrome" is caught but "start a timer" is left for the timer
skill. Adding an app is a config edit, not a code change.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from typing import Any

from core.platform import current_os, pick_for_os, run_detached
from skills.base import Skill, SkillRequest, SkillResult

# Spoken aliases -> config key. Keeps "google chrome" / "vs code" working.
_ALIASES = {
    "google chrome": "chrome",
    "vs code": "editor",
    "vscode": "editor",
    "code": "editor",
    "visual studio code": "editor",
    "web browser": "browser",
}


class AppsSkill(Skill):
    name = "apps"
    description = "Open or close a known application."

    def __init__(self, apps_table: dict[str, dict[str, str]]) -> None:
        self._apps = apps_table
        names = sorted({*apps_table.keys(), *_ALIASES.keys()}, key=len, reverse=True)
        alternation = "|".join(re.escape(n) for n in names) if names else r"(?!x)x"
        self._app_re = re.compile(alternation, re.IGNORECASE)
        self.patterns = [
            re.compile(
                rf"\b(?P<action>open|launch|start|run|close|quit|kill|exit)\s+"
                rf"(?:the\s+|my\s+)?(?P<app>{alternation})\b",
                re.IGNORECASE,
            )
        ]

    def _resolve_key(self, app: str) -> str | None:
        app = app.lower().strip()
        if app in self._apps:
            return app
        return _ALIASES.get(app)

    async def execute(self, request: SkillRequest) -> SkillResult:
        m = request.match
        if m is None:
            return SkillResult("I didn't catch which app.", success=False)
        action = m.group("action").lower()
        key = self._resolve_key(m.group("app"))
        if key is None:
            return SkillResult(f"I don't have {m.group('app')} configured.", success=False)

        if action in ("open", "launch", "start", "run"):
            command = pick_for_os(self._apps[key])
            if not command:
                return SkillResult(f"{key} isn't set up for this OS.", success=False)
            run_detached(command)
            return SkillResult(f"Opening {key}.", data={"app": key, "action": "open"})

        # close / quit / kill — best-effort, cross-platform.
        self._close(key)
        return SkillResult(f"Closing {key}.", data={"app": key, "action": "close"})

    def _win_image_name(self, key: str) -> str:
        """Windows process image for ``taskkill``, derived from the launch command.

        The config key ("browser") is a *logical* name, not the executable
        ("msedge.exe"), so killing by ``<key>.exe`` only works when they happen to
        match (chrome, spotify). Parse the exe out of the configured Windows launch
        command instead — ``"start msedge"`` -> ``msedge.exe`` — falling back to
        ``<key>.exe`` when there's nothing to go on.
        """
        cmd = (self._apps.get(key, {}).get("windows") or "").strip()
        try:
            tokens = shlex.split(cmd, posix=False)  # keep quotes so titles/paths stay whole
        except ValueError:
            tokens = cmd.split()
        if tokens and tokens[0].lower() == "start":
            tokens = tokens[1:]
            # `start "title" prog`: skip a leading quoted token that's a window
            # title, not the executable (no path separator, no .exe suffix).
            if (len(tokens) > 1 and tokens[0][:1] in ('"', "'")
                    and "\\" not in tokens[0]
                    and not tokens[0].strip("\"'").lower().endswith(".exe")):
                tokens = tokens[1:]
        exe = (tokens[0] if tokens else key).strip("\"'")
        exe = exe.replace("/", "\\").split("\\")[-1]  # strip any directory
        return exe if exe.lower().endswith(".exe") else f"{exe}.exe"

    def _close(self, key: str) -> None:
        os_name = current_os()
        if os_name == "windows":
            subprocess.Popen(["taskkill", "/IM", self._win_image_name(key), "/F"])
        elif os_name == "darwin":
            subprocess.Popen(["osascript", "-e", f'tell application "{key}" to quit'])
        else:
            subprocess.Popen(["pkill", "-i", key])

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["open", "close"]},
                        "app": {"type": "string", "enum": sorted(self._apps.keys())},
                    },
                    "required": ["action", "app"],
                },
            },
        }
