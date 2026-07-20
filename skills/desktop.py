"""Desktop control ported from v1 jarvis-web's nut-js tool catalog.

Typing, key chords, window management, clipboard, and display brightness —
one Skill per capability so both routing paths (regex fast path + LLM tools)
get them. pyautogui/pyperclip are imported lazily inside ``execute`` so the
registry still imports on a headless machine, and tests can monkeypatch the
modules without a display.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from core.platform import IS_MACOS, IS_WINDOWS
from skills.base import Skill, SkillRequest, SkillResult

#: v1 jarvis-web's KEY_MAP, translated to pyautogui key names.
KEY_MAP: dict[str, str] = {
    "enter": "enter", "return": "enter",
    "esc": "esc", "escape": "esc",
    "tab": "tab", "space": "space", "spacebar": "space",
    "backspace": "backspace", "delete": "delete", "del": "delete",
    "up": "up", "down": "down", "left": "left", "right": "right",
    "home": "home", "end": "end",
    "pageup": "pageup", "page up": "pageup",
    "pagedown": "pagedown", "page down": "pagedown",
    "control": "ctrl", "ctrl": "ctrl", "alt": "alt", "shift": "shift",
    # The OS "super" key: the Windows key there, ⌘ on a Mac — saying either
    # name lands on whatever this machine actually has.
    **dict.fromkeys(("windows", "win", "command", "cmd"),
                    "command" if IS_MACOS else "win"),
    "play": "playpause", "pause": "playpause", "play pause": "playpause",
    "next": "nexttrack", "previous": "prevtrack", "prev": "prevtrack",
    "mute": "volumemute", "volume up": "volumeup", "volume down": "volumedown",
    **{f"f{i}": f"f{i}" for i in range(1, 13)},
    **{ch: ch for ch in "abcdefghijklmnopqrstuvwxyz0123456789"},
}


def resolve_keys(raw: str | list[str]) -> list[str]:
    """'control shift a' or ['ctrl','a'] -> pyautogui names; unknowns dropped.

    Two-word aliases ("page down", "volume up") are matched greedily before
    single tokens, so "press page down" means PageDown, not the down arrow.
    """
    if isinstance(raw, str):
        parts = re.split(r"[\s+,]+", raw.strip().lower())
    else:
        parts = [str(p).strip().lower() for p in raw]
    out: list[str] = []
    i = 0
    while i < len(parts):
        pair = " ".join(parts[i : i + 2])
        if i + 1 < len(parts) and pair in KEY_MAP:
            out.append(KEY_MAP[pair])
            i += 2
        elif parts[i] in KEY_MAP:
            out.append(KEY_MAP[parts[i]])
            i += 1
        else:
            i += 1
    return out


def _pyautogui():
    """Lazy pyautogui with the fail-safe off.

    Pointer mode may park the cursor in a corner; the default FAILSAFE would
    then make every desktop action raise instead of run.
    """
    import pyautogui

    pyautogui.FAILSAFE = False
    return pyautogui


class TypeTextSkill(Skill):
    name = "type_text"
    controls_pc = True
    description = "Type text into the currently focused window."

    patterns = [
        re.compile(r"^\s*(?:medo[,!\s]+)?type\s+(?:out\s+)?(?P<body>.+?)\s*$", re.IGNORECASE),
    ]

    async def execute(self, request: SkillRequest) -> SkillResult:
        body = str(request.args.get("text") or "").strip()
        if not body and request.match:
            body = (request.match.groupdict().get("body") or "").strip()
        if not body:
            return SkillResult("What should I type?", success=False)
        try:
            gui = _pyautogui()
            if body.isascii():
                await asyncio.to_thread(gui.write, body, 0.02)
            else:
                # pyautogui.write silently drops non-ASCII (Cyrillic!) —
                # paste through the clipboard instead, then restore it.
                import pyperclip

                try:
                    previous = pyperclip.paste()
                except Exception:
                    previous = ""
                pyperclip.copy(body)
                # Paste is Cmd+V on macOS; Ctrl+V there does nothing, which
                # made "type <Cyrillic>" silently type nothing on a Mac.
                from core.platform import IS_MACOS

                await asyncio.to_thread(
                    gui.hotkey, "command" if IS_MACOS else "ctrl", "v"
                )
                await asyncio.sleep(0.15)  # let the paste land before restoring
                pyperclip.copy(previous)
        except Exception as exc:
            return SkillResult(f"I couldn't type that: {exc}", success=False)
        return SkillResult(f"Typed {len(body)} characters.")

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Exact text to type."}
                    },
                    "required": ["text"],
                },
            },
        }


class PressKeysSkill(Skill):
    name = "press_keys"
    controls_pc = True
    description = "Press a key or keyboard shortcut (e.g. control s, alt tab, f5)."

    patterns = [
        re.compile(r"^\s*(?:medo[,!\s]+)?(?:press|hit)\s+(?P<keys>.+?)\s*$", re.IGNORECASE),
    ]

    async def execute(self, request: SkillRequest) -> SkillResult:
        raw = request.args.get("keys") or (
            request.match.groupdict().get("keys", "") if request.match else ""
        )
        keys = resolve_keys(raw)
        if not keys:
            return SkillResult(
                "I don't know those keys — try something like 'press control s'.",
                success=False,
            )
        try:
            gui = _pyautogui()
            if len(keys) == 1:
                await asyncio.to_thread(gui.press, keys[0])
            else:
                await asyncio.to_thread(gui.hotkey, *keys)
        except Exception as exc:
            return SkillResult(f"I couldn't press that: {exc}", success=False)
        return SkillResult(f"Pressed {' '.join(keys)}.")

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keys": {
                            "type": "string",
                            "description": "Space-separated keys, modifiers first (e.g. 'ctrl shift t').",
                        }
                    },
                    "required": ["keys"],
                },
            },
        }


class WindowActionSkill(Skill):
    name = "window_action"
    controls_pc = True
    description = "Minimize, maximize, or close the focused window; show desktop; switch apps."

    #: action -> key chord. The v1 jarvis-web mapping was Windows-only —
    #: Alt+F4/Win+Down are dead keys on a Mac, so macOS gets its own chords
    #: (Cmd+M / full-screen toggle / Cmd+W / F11 / Cmd+Tab).
    ACTIONS: dict[str, tuple[str, ...]] = (
        {
            "minimize": ("command", "m"),
            "maximize": ("ctrl", "command", "f"),  # full-screen toggle — the mac "maximize"
            "close": ("command", "w"),
            "show_desktop": ("f11",),  # needs "use F-keys as standard" — best effort
            "switch": ("command", "tab"),
        }
        if IS_MACOS
        else {
            "minimize": ("win", "down"),
            "maximize": ("win", "up"),
            "close": ("alt", "f4"),
            "show_desktop": ("win", "d"),
            "switch": ("alt", "tab"),
        }
    )

    patterns = [
        re.compile(r"\b(?P<mm>minimi[sz]e|maximi[sz]e)\s+(?:the\s+|this\s+)?window\b", re.IGNORECASE),
        re.compile(r"\bclose\s+(?:the\s+|this\s+)?window\b", re.IGNORECASE),
        re.compile(r"\bshow\s+(?:the\s+)?desktop\b", re.IGNORECASE),
        re.compile(r"\bswitch\s+(?:windows?|apps?)\b", re.IGNORECASE),
    ]

    def _action_from(self, text: str) -> str:
        t = text.lower()
        if "minimi" in t:
            return "minimize"
        if "maximi" in t:
            return "maximize"
        if "close" in t:
            return "close"
        if "desktop" in t:
            return "show_desktop"
        return "switch"

    async def execute(self, request: SkillRequest) -> SkillResult:
        action = str(request.args.get("action") or self._action_from(request.text))
        chord = self.ACTIONS.get(action)
        if chord is None:
            return SkillResult(f"I can't do '{action}' with windows.", success=False)
        try:
            gui = _pyautogui()
            await asyncio.to_thread(gui.hotkey, *chord)
        except Exception as exc:
            return SkillResult(f"I couldn't do that: {exc}", success=False)
        spoken = {
            "minimize": "Minimized.",
            "maximize": "Maximized.",
            "close": "Closed the window.",
            "show_desktop": "Showing the desktop.",
            "switch": "Switched.",
        }[action]
        return SkillResult(spoken, data={"action": action})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": sorted(self.ACTIONS),
                        }
                    },
                    "required": ["action"],
                },
            },
        }


class ClipboardSkill(Skill):
    name = "clipboard"
    controls_pc = True
    description = "Read the clipboard aloud, or copy given text to it."

    patterns = [
        re.compile(r"\b(?:read|what(?:'?s| is)\s+(?:in|on))\s+(?:the\s+|my\s+)?clipboard\b", re.IGNORECASE),
        re.compile(r"\bcopy\s+(?P<body>.+?)\s+to\s+(?:the\s+|my\s+)?clipboard\b", re.IGNORECASE),
    ]

    async def execute(self, request: SkillRequest) -> SkillResult:
        body = str(request.args.get("text") or "").strip()
        if not body and request.match:
            body = (request.match.groupdict().get("body") or "").strip()
        try:
            import pyperclip

            if body or request.args.get("action") == "write":
                if not body:
                    return SkillResult("What should I copy?", success=False)
                pyperclip.copy(body)
                return SkillResult("Copied to the clipboard.")
            content = (pyperclip.paste() or "").strip()
        except Exception as exc:
            return SkillResult(f"I couldn't reach the clipboard: {exc}", success=False)
        if not content:
            return SkillResult("The clipboard is empty.")
        clipped = content if len(content) <= 220 else content[:220] + "… and it goes on"
        return SkillResult(f"The clipboard says: {clipped}", data={"length": len(content)})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["read", "write"]},
                        "text": {"type": "string", "description": "Text to copy when writing."},
                    },
                    "required": ["action"],
                },
            },
        }


class BrightnessSkill(Skill):
    name = "brightness"
    controls_pc = True
    description = "Set the display brightness (0-100), where supported."

    patterns = [
        re.compile(r"\b(?:set\s+)?brightness\s+(?:to\s+)?(?P<level>\d{1,3})\b", re.IGNORECASE),
        re.compile(r"\b(?:screen\s+|display\s+)?brightness\s+(?P<ud>up|down)\b", re.IGNORECASE),
    ]

    _UNSUPPORTED = (
        "This display doesn't support software brightness control — external "
        "monitors usually don't."
    )

    async def _powershell(self, command: str) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-NonInteractive", "-Command", command,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        return proc.returncode or 0, out.decode(errors="replace").strip()

    async def _current(self) -> int | None:
        code, out = await self._powershell(
            "(Get-WmiObject -Namespace root/wmi -Class WmiMonitorBrightness).CurrentBrightness"
        )
        try:
            return int(out.splitlines()[0]) if code == 0 and out else None
        except ValueError:
            return None

    async def execute(self, request: SkillRequest) -> SkillResult:
        if not IS_WINDOWS:
            return SkillResult("Brightness control isn't wired up on this OS.", success=False)

        level = request.args.get("level")
        ud = None
        if request.match:
            gd = request.match.groupdict()
            level = level if level is not None else gd.get("level")
            ud = gd.get("ud")
        if level is None and ud:
            current = await self._current()
            if current is None:
                return SkillResult(self._UNSUPPORTED, success=False)
            level = current + (15 if ud.lower() == "up" else -15)
        if level is None:
            return SkillResult("Tell me a brightness from 0 to 100.", success=False)

        level = max(0, min(100, int(level)))
        code, _ = await self._powershell(
            "(Get-WmiObject -Namespace root/wmi -Class WmiMonitorBrightnessMethods)"
            f".WmiSetBrightness(1,{level})"
        )
        if code != 0:
            return SkillResult(self._UNSUPPORTED, success=False)
        return SkillResult(f"Brightness set to {level} percent.", data={"level": level})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "integer", "minimum": 0, "maximum": 100}
                    },
                    "required": ["level"],
                },
            },
        }
