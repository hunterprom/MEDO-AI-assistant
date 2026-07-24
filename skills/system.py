"""System control & telemetry: volume, resource info, screenshots, power.

Read-only skills (battery, CPU/RAM) run freely. Power actions that are hard to
undo (shutdown, restart) set ``needs_confirmation`` so the router's safety gate
demands a spoken "yes" first. Volume/lock are cross-platform where cheap and
degrade with a clear message where an OS isn't wired up yet.
"""

from __future__ import annotations

import asyncio

import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from core.platform import IS_MACOS, IS_WINDOWS, current_os
from skills.base import Skill, SkillRequest, SkillResult


# --------------------------------------------------------------------------- #
# Volume — absolute control on macOS (osascript) and Windows (pycaw / Core
# Audio); a keyboard media-key nudge is the fallback everywhere else (and on
# Windows if pycaw isn't installed). Absolute control lets "set volume to 50"
# and "what's the volume" work exactly; the fallback can only nudge up/down.
# --------------------------------------------------------------------------- #
def _mac_get_volume() -> int:
    out = subprocess.run(
        ["osascript", "-e", "output volume of (get volume settings)"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out or 0)


def _mac_set_volume(level: int) -> None:
    subprocess.run(["osascript", "-e", f"set volume output volume {level}"], check=True)


def _mac_set_muted(muted: bool) -> None:
    subprocess.run(
        ["osascript", "-e", f"set volume output muted {str(muted).lower()}"], check=True
    )


def _win_endpoint():
    """Return the default speaker's ``IAudioEndpointVolume``, or None if pycaw
    (and its ``comtypes`` backend) aren't available or there's no audio device.

    pycaw's API has drifted across versions: older ``GetSpeakers()`` returned a
    raw ``IMMDevice`` you called ``.Activate()`` on; newer releases return an
    ``AudioDevice`` wrapper that exposes an ``EndpointVolume`` property instead.
    Try the wrapper first, then the classic call, then a raw device-enumerator
    fallback, so we work regardless of the installed pycaw.
    """
    try:
        import comtypes
        try:
            comtypes.CoInitialize()  # idempotent; needed off the COM main thread
        except Exception:
            pass

        from ctypes import POINTER, cast

        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        speakers = AudioUtilities.GetSpeakers()

        # 1) Newer pycaw: AudioDevice wrapper with a ready-made EndpointVolume.
        endpoint = getattr(speakers, "EndpointVolume", None)
        if endpoint is not None:
            return endpoint

        # 2) Older pycaw: GetSpeakers() was the raw IMMDevice itself.
        if hasattr(speakers, "Activate"):
            interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            return cast(interface, POINTER(IAudioEndpointVolume))

        # 3) Version-stable fallback: enumerate the default render endpoint directly.
        from pycaw.api.mmdeviceapi import IMMDeviceEnumerator
        from pycaw.constants import CLSID_MMDeviceEnumerator

        enumerator = comtypes.CoCreateInstance(
            CLSID_MMDeviceEnumerator, IMMDeviceEnumerator, comtypes.CLSCTX_INPROC_SERVER
        )
        device = enumerator.GetDefaultAudioEndpoint(0, 1)  # eRender, eMultimedia
        interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))
    except Exception:
        return None


def _win_get_volume() -> int | None:
    ep = _win_endpoint()
    return None if ep is None else round(ep.GetMasterVolumeLevelScalar() * 100)


def _win_set_volume(level: int) -> bool:
    ep = _win_endpoint()
    if ep is None:
        return False
    ep.SetMasterVolumeLevelScalar(max(0, min(100, level)) / 100.0, None)
    return True


def _win_set_muted(muted: bool) -> bool:
    ep = _win_endpoint()
    if ep is None:
        return False
    ep.SetMute(1 if muted else 0, None)
    return True


def _nudge_media_key(up: bool) -> str:
    """Bump the volume one step with the keyboard media keys (best-effort)."""
    try:
        import pyautogui

        pyautogui.press("volumeup" if up else "volumedown")
        return "Turned the volume up." if up else "Turned the volume down."
    except Exception:
        return "I couldn't reach the volume controls on this system."


def _current_volume() -> int | None:
    """Current output volume 0–100, or None if this OS can't report it exactly."""
    if IS_MACOS:
        return _mac_get_volume()
    if IS_WINDOWS:
        return _win_get_volume()
    return None  # Linux: no absolute query without pulling in an extra mixer dep


class VolumeSkill(Skill):
    name = "volume"
    controls_pc = True
    description = "Adjust or query the system output volume."

    patterns = [
        re.compile(r"\b(?:set\s+)?volume\s+(?:to\s+)?(?P<level>\d{1,3})\b", re.IGNORECASE),
        # "turn up/down" is generic — decline it when the object is the SCREEN,
        # so "turn down the brightness" reaches BrightnessSkill instead of
        # silently changing the volume.
        re.compile(r"\b(?:turn\s+(?:it\s+)?(?P<updown>up|down)"
                   r"(?!\s+(?:the\s+|my\s+)?(?:screen|display|brightness|monitor))"
                   r"|volume\s+(?P<ud2>up|down))\b", re.IGNORECASE),
        re.compile(r"\b(?P<louder>louder|quieter)\b", re.IGNORECASE),
        re.compile(r"\b(?P<mute>mute|unmute)\b", re.IGNORECASE),
        # Everyday volume phrasings.
        re.compile(r"\btoo\s+(?:loud|quiet|low|soft)\b", re.IGNORECASE),
        re.compile(r"\b(?:can'?t|cannot)\s+hear\b|\bspeak\s+up\b", re.IGNORECASE),
        re.compile(r"\b(?:max|maximum|full)\s+volume\b|\bvolume\s+all\s+the\s+way\b", re.IGNORECASE),
        re.compile(r"\bwhat(?:'?s| is)?\s+the\s+volume\b", re.IGNORECASE),
    ]

    def _set_absolute(self, level: int) -> str:
        level = max(0, min(100, level))
        if IS_MACOS:
            _mac_set_volume(level)
            return f"Volume set to {level} percent."
        if IS_WINDOWS and _win_set_volume(level):
            return f"Volume set to {level} percent."
        # No absolute control here — nudge toward the target as best we can.
        return _nudge_media_key(level >= (_current_volume() or 0))

    def _set_muted(self, muted: bool) -> str:
        if IS_MACOS:
            _mac_set_muted(muted)
        elif IS_WINDOWS and _win_set_muted(muted):
            pass
        else:
            try:
                import pyautogui

                pyautogui.press("volumemute")  # a toggle; can't force a state
                return "Toggled mute."
            except Exception:
                return "I couldn't reach the mute control on this system."
        return "Muted." if muted else "Unmuted."

    def _step(self, up: bool) -> str:
        cur = _current_volume()
        if cur is not None:
            return self._set_absolute(cur + (10 if up else -10))
        return _nudge_media_key(up)

    async def execute(self, request: SkillRequest) -> SkillResult:
        text = request.text.lower()
        m = request.match
        # LLM tool path carries the intent in args {action, level}, with
        # match=None — the fast-path text logic below can't see it, so a tool
        # call to "set"/"mute"/"up"/"down" would otherwise fall through to the
        # query branch and just READ the volume. Honour args first.
        args = request.args or {}
        action = str(args.get("action") or "").strip().lower()
        alvl = args.get("level")
        if alvl is not None and str(alvl).strip() != "":
            try:
                return SkillResult(self._set_absolute(int(alvl)))
            except (TypeError, ValueError):
                pass
        if action in ("mute", "unmute"):
            return SkillResult(self._set_muted(action == "mute"))
        if action in ("up", "louder"):
            return SkillResult(self._step(True))
        if action in ("down", "quieter"):
            return SkillResult(self._step(False))
        if action == "set":                 # set with no level -> ask
            return SkillResult("What volume — 0 to 100 percent?", success=False)
        if m and m.groupdict().get("level"):
            return SkillResult(self._set_absolute(int(m.group("level"))))
        if any(p in text for p in ("max volume", "maximum volume", "full volume",
                                   "all the way up")):
            return SkillResult(self._set_absolute(100))
        if "mute" in text:
            return SkillResult(self._set_muted("unmute" not in text))
        # "too loud" => quieter; "too quiet/low/soft", "can't hear", "speak up" => louder.
        if "too loud" in text or "all the way down" in text:
            return SkillResult(self._step(False))
        if any(p in text for p in ("too quiet", "too low", "too soft",
                                   "can't hear", "cant hear", "cannot hear", "speak up")):
            return SkillResult(self._step(True))
        if any(w in text for w in ("up", "louder")):
            return SkillResult(self._step(True))
        if any(w in text for w in ("down", "quieter")):
            return SkillResult(self._step(False))
        # bare "what's the volume"
        cur = _current_volume()
        if cur is not None:
            return SkillResult(f"Volume is at {cur} percent.")
        return SkillResult("I can't read the exact volume on this system.")

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["set", "up", "down", "mute", "unmute", "query"]},
                        "level": {"type": "integer", "minimum": 0, "maximum": 100},
                    },
                    "required": ["action"],
                },
            },
        }


# --------------------------------------------------------------------------- #
# System info (read-only)
# --------------------------------------------------------------------------- #
class SystemInfoSkill(Skill):
    name = "system_info"
    description = "Report battery, CPU, memory, and free disk space."
    routing_phrases = [
        "how much battery do I have", "what's my cpu usage",
        "how much disk space is left", "how much memory am I using",
        "how's my computer doing",
    ]

    patterns = [
        re.compile(r"\bbattery\b", re.IGNORECASE),
        re.compile(r"\b(?:cpu|processor)\s*(?:usage|load)?\b", re.IGNORECASE),
        re.compile(r"\b(?:ram|memory)\s*(?:usage)?\b", re.IGNORECASE),
        re.compile(r"\bsystem\s+(?:status|stats|info)\b", re.IGNORECASE),
        # Disk / storage. Deliberately broad on the words people actually use
        # for it ("how much space do I have", "free space", "storage left").
        re.compile(r"\b(?:disk|drive|storage)\s*(?:space|usage)?\b", re.IGNORECASE),
        re.compile(r"\b(?:free|available|much|enough)\s+(?:space|storage|room)\b",
                   re.IGNORECASE),
        re.compile(r"\bspace\s+(?:left|remaining|free|do\s+i\s+have|on\s+(?:my|the)"
                   r"\s+(?:pc|computer|drive|disk|machine))\b", re.IGNORECASE),
        # MK: "колку простор/место имам", "слободен простор"
        re.compile(r"\b(?:простор|место)\b", re.IGNORECASE),
    ]

    async def execute(self, request: SkillRequest) -> SkillResult:
        text = request.text.lower()
        want_batt = "battery" in text or "батериј" in text
        want_cpu = "cpu" in text or "processor" in text or "процесор" in text
        want_mem = "ram" in text or "memory" in text or "меморија" in text
        want_disk = any(w in text for w in (
            "disk", "drive", "storage", "space", "room", "простор", "место"))
        if not (want_batt or want_cpu or want_mem or want_disk):  # "system status"
            want_batt = want_cpu = want_mem = want_disk = True

        parts: list[str] = []
        data: dict[str, Any] = {}
        if want_cpu:
            cpu = psutil.cpu_percent(interval=0.2)
            parts.append(f"CPU at {cpu:.0f} percent")
            data["cpu_percent"] = cpu
        if want_mem:
            mem = psutil.virtual_memory()
            parts.append(f"memory at {mem.percent:.0f} percent")
            data["mem_percent"] = mem.percent
        if want_disk:
            # The system drive: where "how much space do I have" is really asked.
            path = "C:\\" if sys.platform == "win32" else "/"
            try:
                du = psutil.disk_usage(path)
                free_gb, total_gb = du.free / 1e9, du.total / 1e9
                parts.append(
                    f"{free_gb:.0f} GB free of {total_gb:.0f} GB on the system drive")
                data["disk"] = {"free_gb": round(free_gb, 1),
                                "total_gb": round(total_gb, 1),
                                "percent_used": du.percent}
            except OSError:
                parts.append("I couldn't read the disk")
        if want_batt:
            batt = psutil.sensors_battery()
            if batt is None:
                parts.append("no battery detected")
            else:
                state = "charging" if batt.power_plugged else "on battery"
                parts.append(f"battery at {batt.percent:.0f} percent, {state}")
                data["battery_percent"] = batt.percent
        return SkillResult("Right now: " + ", ".join(parts) + ".", data=data)

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string", "enum": ["battery", "cpu", "memory", "all"]}
                    },
                    "required": [],
                },
            },
        }


# --------------------------------------------------------------------------- #
# Screenshot
# --------------------------------------------------------------------------- #
class ScreenshotSkill(Skill):
    name = "screenshot"
    controls_pc = True
    description = "Capture the screen to an image file."

    patterns = [
        re.compile(r"\b(?:take\s+(?:a\s+)?)?screenshot\b", re.IGNORECASE),
        re.compile(r"\bcapture\s+(?:the\s+)?screen\b", re.IGNORECASE),
    ]

    def __init__(self, save_dir: Path) -> None:
        self._save_dir = save_dir

    async def execute(self, request: SkillRequest) -> SkillResult:
        self._save_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
        target = self._save_dir / f"screenshot-{stamp}.png"
        try:
            def _capture() -> None:
                # Grabbing and encoding a full screen takes long enough to be
                # audible as a gap if it runs on the event loop.
                if IS_MACOS:
                    subprocess.run(["screencapture", "-x", str(target)], check=True)
                else:
                    import pyautogui

                    pyautogui.screenshot().save(str(target))

            await asyncio.to_thread(_capture)
        except Exception as exc:  # permission denied, headless, etc.
            return SkillResult(f"I couldn't take a screenshot: {exc}", success=False)
        return SkillResult(f"Screenshot saved to {target.name}.", data={"path": str(target)})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }


# --------------------------------------------------------------------------- #
# Power (lock / sleep / shutdown* / restart*)
# --------------------------------------------------------------------------- #
#: The only actions PowerSkill will ever take. Anything else — including
#: nothing at all — is a question, not an action.
_POWER_ACTIONS = ("lock", "sleep", "shutdown", "restart")


class PowerSkill(Skill):
    name = "power"
    controls_pc = True
    description = "Lock, sleep, shut down, or restart the machine."

    patterns = [
        re.compile(r"\block\s+(?:the\s+)?(?:screen|computer|pc|mac)\b", re.IGNORECASE),
        re.compile(r"\b(?:go\s+to\s+)?sleep\b", re.IGNORECASE),
        re.compile(r"\bshut\s*(?:down|off)\b", re.IGNORECASE),
        re.compile(r"\b(?:restart|reboot)\b", re.IGNORECASE),
    ]

    def _action(self, text: str) -> str | None:
        """Which power action the words name, or None if they name none.

        There is deliberately NO default. This used to fall through to
        "sleep", which meant any call that reached the skill without matching
        text — an LLM tool call with empty arguments, a test harness invoking
        execute() directly — suspended the machine with no confirmation. A
        power skill must never infer an action from silence.
        """
        if "restart" in text or "reboot" in text:
            return "restart"
        if "shut" in text:
            return "shutdown"
        if "lock" in text:
            return "lock"
        if "sleep" in text or "suspend" in text:
            return "sleep"
        return None

    def _run(self, action: str) -> str:
        os_name = current_os()
        cmds = {
            "lock": {
                "darwin": ["pmset", "displaysleepnow"],
                "windows": ["rundll32.exe", "user32.dll,LockWorkStation"],
                "linux": ["loginctl", "lock-session"],
            },
            "sleep": {
                "darwin": ["pmset", "sleepnow"],
                "windows": ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
                "linux": ["systemctl", "suspend"],
            },
            "shutdown": {
                "darwin": ["osascript", "-e", 'tell app "System Events" to shut down'],
                "windows": ["shutdown", "/s", "/t", "0"],
                "linux": ["shutdown", "-h", "now"],
            },
            "restart": {
                "darwin": ["osascript", "-e", 'tell app "System Events" to restart'],
                "windows": ["shutdown", "/r", "/t", "0"],
                "linux": ["shutdown", "-r", "now"],
            },
        }
        cmd = cmds[action].get(os_name)
        if cmd is None:
            return f"I can't {action} on this system."
        subprocess.Popen(cmd)
        verbs = {"lock": "Locking", "sleep": "Sleeping", "shutdown": "Shutting down", "restart": "Restarting"}
        return f"{verbs[action]} now."

    async def execute(self, request: SkillRequest) -> SkillResult:
        # The LLM path can call this with an explicit action; the fast path
        # infers it from the words. Either way an unnamed action is a question,
        # never an assumption.
        requested = str(request.args.get("action") or "").strip().lower()
        action = requested if requested in _POWER_ACTIONS else \
            self._action(request.text.lower())
        if action is None:
            return SkillResult(
                "Do you want me to lock, sleep, restart, or shut down?",
                success=False)
        # Everything except locking the screen interrupts what you were doing,
        # so everything except locking asks first. Sleep used to be silent —
        # and a stray sleep costs you your session just as surely as a reboot.
        if action != "lock" and not request.context.get("confirmed"):
            phrase = {"shutdown": "shut down", "restart": "restart",
                      "sleep": "put the machine to sleep"}[action]
            return SkillResult(
                f"Are you sure you want to {phrase}? Say yes to confirm.",
                needs_confirmation=True,
            )
        return SkillResult(self._run(action), data={"action": action})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["lock", "sleep", "shutdown", "restart"]}
                    },
                    "required": ["action"],
                },
            },
        }


# --------------------------------------------------------------------------- #
# Pointer mode (gesture mouse control) — toggles the vision sidecar
# --------------------------------------------------------------------------- #
class PointerControlSkill(Skill):
    name = "pointer_control"
    controls_pc = True
    description = "Turn gesture mouse control (pointer mode) on or off."

    patterns = [
        re.compile(
            r"\b(?:pointer|mouse|cursor)(?:\s+(?:control|mode))?\s+(?P<state>on|off)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:control|take)\s+(?:the\s+|my\s+)?(?:mouse|cursor)\b", re.IGNORECASE),
        re.compile(r"\bstop\s+controlling\s+(?:the\s+|my\s+)?(?:mouse|cursor)\b", re.IGNORECASE),
    ]

    def __init__(self, stream_port: int) -> None:
        self._port = stream_port

    async def execute(self, request: SkillRequest) -> SkillResult:
        text = request.text.lower()
        state = request.args.get("state")
        if state is None and request.match:
            state = request.match.groupdict().get("state")
        if state is None:
            state = "off" if "stop" in text else "on"
        want = str(state).lower() != "off"

        import httpx

        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.post(
                    f"http://127.0.0.1:{self._port}/pointer", json={"on": want}
                )
                data = resp.json()
        except Exception:
            return SkillResult(
                "The vision sidecar isn't running, so I can't control the mouse.",
                success=False,
            )
        if want and not data.get("on"):
            return SkillResult(
                data.get("error") or "Pointer mode is unavailable.", success=False
            )
        if want:
            return SkillResult(
                "Pointer mode on — point with your index finger; pinch to click, "
                "three fingers for right-click, make a fist to stop."
            )
        return SkillResult("Pointer mode off.")

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "state": {"type": "string", "enum": ["on", "off"]}
                    },
                    "required": ["state"],
                },
            },
        }
