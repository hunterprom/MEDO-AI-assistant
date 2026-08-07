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

from core import mk
from core.platform import current_os, pick_for_os, run_detached
from skills.base import Skill, SkillRequest, SkillResult

# Spoken aliases -> config key. Keeps "google chrome" / "vs code" working, and
# gives the Macedonian names for the same apps — Whisper writes "Chrome" as
# "хром" when you're speaking Macedonian, so those are aliases like any other.
_ALIASES = {
    "google chrome": "chrome",
    "vs code": "editor",
    "vscode": "editor",
    "code": "editor",
    "visual studio code": "editor",
    "web browser": "browser",
    # Macedonian
    "хром": "chrome",
    "гугл хром": "chrome",
    "прелистувач": "browser",
    "прелистувачот": "browser",
    "пребарувач": "browser",
    "браузер": "browser",
    "терминал": "terminal",
    "терминалот": "terminal",
    "командна линија": "terminal",
    "уредувач": "editor",
    "едитор": "editor",
    "спотифај": "spotify",
    "спотифи": "spotify",
    "стим": "steam",
    "дискорд": "discord",
    "калкулатор": "calculator",
    "експлорер": "explorer",
    "фајл менаџер": "explorer",
    "нотепад": "notepad",
    "бележник": "notepad",
    "тимс": "teams",
    "ворд": "word",
    "ексел": "excel",
    "телеграм": "telegram",
    "фајрфокс": "firefox",
    "еџ": "edge",
    "обсидијан": "obsidian",
    "опсидијан": "obsidian",
    "фотошоп": "photoshop",
    "капкат": "capcut",
    "блендер": "blender",
}
# Declined forms ("стимот", "стима") are folded back to these stems by
# mk.undeclined() in _resolve_key — don't list them here. An alias whose target
# is missing from the user's config table is filtered out there too, so naming
# an app most people won't have costs nothing.


class AppsSkill(Skill):
    name = "apps"
    controls_pc = True
    description = "Open or close a known application."

    def __init__(self, apps_table: dict[str, dict[str, str]]) -> None:
        self._apps = apps_table
        names = sorted({*apps_table.keys(), *_ALIASES.keys()}, key=len, reverse=True)
        alternation = "|".join(re.escape(n) for n in names) if names else r"(?!x)x"
        self._app_re = re.compile(alternation, re.IGNORECASE)
        self.patterns = [
            re.compile(
                rf"\b(?P<action>open|launch|start|run|close|quit|kill|exit)\s+"
                # "open UP spotify", "start UP chrome" — the launch particle sat
                # between the verb and the app name and broke the match, sending
                # a plain launch to the LLM (which narrates instead of opening).
                rf"(?:up\s+)?(?:the\s+|my\s+)?(?P<app>{alternation})\b",
                re.IGNORECASE,
            ),
            # MK "отвори ми го хром" / "стартувај спотифај". The clitics
            # ("ми го") pile up between the verb and the app; the verb itself
            # tells us whether this is an open or a close.
            # The optional NOUN_ENDING sits OUTSIDE the capture: Macedonian
            # declines borrowed names ("отвори стимА", "затвори хромОТ"), and a
            # bare \b after the alternation matched neither — the launch fell
            # through to the LLM, which then narrated instead of opening.
            re.compile(
                rf"\b(?P<mk_open>{mk.OPEN}){mk.CLITICS}\s+"
                rf"(?P<app>{alternation})(?:{mk.NOUN_ENDING})?\b",
                re.IGNORECASE,
            ),
            re.compile(
                rf"\b(?P<mk_close>{mk.CLOSE}){mk.CLITICS}\s+"
                rf"(?P<app>{alternation})(?:{mk.NOUN_ENDING})?\b",
                re.IGNORECASE,
            ),
        ]

    #: A launch request that ALSO asks for text to be written ("open Notepad and
    #: summarize the Odyssey, type it out") is not a plain launch. Matching it
    #: here answers "Opening notepad." in 9 ms and silently drops the writing —
    #: exactly what a live transcript showed. Decline, so WriteInAppSkill takes
    #: the literal cases and the brain composes the rest and calls it as a tool.
    _ALSO_WRITES = re.compile(
        r"\b(?:write|type|typing|put|jot|summari[sz]e|summary|explain|describe|"
        r"translate|compose|draft|paraphrase"
        r"|напиши|искуцај|запиши|сумирај)\b", re.IGNORECASE)

    #: The launch-and-then-do-something sibling of _ALSO_WRITES. "open spotify
    #: and play some jazz" / "open chrome and search for cats" is not a plain
    #: launch — matching it here answers "Opening spotify." and silently drops
    #: the play/search half. Decline, so PlaySkill / SiteSearchSkill (or the LLM
    #: tool-chaining path) can carry out the whole request.
    _ALSO_ACTS = re.compile(
        r"\band\s+(?:then\s+)?(?:play|search|look\s+up|pull\s+up|find|"
        r"put\s+on|throw\s+on|go\s+to|navigate(?:\s+to)?|visit|bring\s+up)\b",
        re.IGNORECASE)

    def match(self, text: str):
        if self._ALSO_WRITES.search(text) or self._ALSO_ACTS.search(text):
            return None
        return super().match(text)

    def _resolve_key(self, app: str) -> str | None:
        app = app.lower().strip()
        # Try the spoken form first, then its Macedonian stems ("стимот" ->
        # "стим"), so a declined name still finds the table entry.
        for form in mk.undeclined(app) or [app]:
            if form in self._apps:
                return form
            # An alias only counts when its target is actually in the table: the
            # match patterns are built from *every* alias, so on a machine whose
            # config has no "steam" entry, "отвори стим" would otherwise resolve
            # to a key that isn't there and blow up on the lookup in execute().
            key = _ALIASES.get(form)
            if key in self._apps:
                return key
        return None

    async def execute(self, request: SkillRequest) -> SkillResult:
        speak_mk = mk.is_cyrillic(request.text)
        args = request.args or {}
        m = request.match
        # The LLM tool path passes structured args with match=None; the fast
        # path passes a regex match. Read args first so tool calls actuate too.
        if args.get("app"):
            app_name = str(args["app"])
            action = str(args.get("action") or "open").lower()
        elif m is not None:
            gd = m.groupdict()
            if gd.get("mk_close"):
                action = "close"
            elif gd.get("mk_open"):
                action = "open"
            else:
                action = m.group("action").lower()
            app_name = m.group("app")
        else:
            return SkillResult("I didn't catch which app.", success=False)
        key = self._resolve_key(app_name)
        if key is None:
            return SkillResult(
                f"Немам конфигурирано {app_name}." if speak_mk
                else f"I don't have {app_name} configured.", success=False)

        if action in ("open", "launch", "start", "run"):
            command = pick_for_os(self._apps[key])
            if not command:
                return SkillResult(
                    f"{key} не е поставен за овој систем." if speak_mk
                    else f"{key} isn't set up for this OS.", success=False)
            run_detached(command)
            return SkillResult(
                f"Отворам {key}." if speak_mk else f"Opening {key}.",
                data={"app": key, "action": "open"})

        # close / quit / kill — best-effort, cross-platform. Graceful, never a
        # force-kill: an app with unsaved work must get its "save changes?"
        # prompt, the same courtesy the other destructive ops here extend.
        self._close(key)
        return SkillResult(
            f"Затворам {key}." if speak_mk else f"Closing {key}.",
            data={"app": key, "action": "close"})

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
            # No /F: bare taskkill sends WM_CLOSE, so a doc with unsaved edits
            # still gets to prompt. /F would force-terminate and lose the work.
            subprocess.Popen(["taskkill", "/IM", self._win_image_name(key)])
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
