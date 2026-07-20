"""System control & telemetry: volume, resource info, screenshots, power.

Read-only skills (battery, CPU/RAM) run freely. Power actions that are hard to
undo (shutdown, restart) set ``needs_confirmation`` so the router's safety gate
demands a spoken "yes" first. Volume/lock are cross-platform where cheap and
degrade with a clear message where an OS isn't wired up yet.
"""

from __future__ import annotations

import re
import subprocess
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
        re.compile(r"\b(?:turn\s+(?:it\s+)?(?P<updown>up|down)|volume\s+(?P<ud2>up|down))\b", re.IGNORECASE),
        re.compile(r"\b(?P<louder>louder|quieter)\b", re.IGNORECASE),
        re.compile(r"\b(?P<mute>mute|unmute)\b", re.IGNORECASE),
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
        if m and m.groupdict().get("level"):
            return SkillResult(self._set_absolute(int(m.group("level"))))
        if "mute" in text:
            return SkillResult(self._set_muted("unmute" not in text))
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
    description = "Report battery, CPU, and memory usage."

    patterns = [
        re.compile(r"\bbattery\b", re.IGNORECASE),
        re.compile(r"\b(?:cpu|processor)\s*(?:usage|load)?\b", re.IGNORECASE),
        re.compile(r"\b(?:ram|memory)\s*(?:usage)?\b", re.IGNORECASE),
        re.compile(r"\bsystem\s+(?:status|stats|info)\b", re.IGNORECASE),
    ]

    async def execute(self, request: SkillRequest) -> SkillResult:
        text = request.text.lower()
        want_batt = "battery" in text
        want_cpu = "cpu" in text or "processor" in text
        want_mem = "ram" in text or "memory" in text
        if not (want_batt or want_cpu or want_mem):  # "system status" -> all
            want_batt = want_cpu = want_mem = True

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
            if IS_MACOS:
                subprocess.run(["screencapture", "-x", str(target)], check=True)
            else:
                import pyautogui

                pyautogui.screenshot().save(str(target))
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

    def _action(self, text: str) -> str:
        if "restart" in text or "reboot" in text:
            return "restart"
        if "shut" in text:
            return "shutdown"
        if "lock" in text:
            return "lock"
        return "sleep"

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
        action = self._action(request.text.lower())
        destructive = action in ("shutdown", "restart")
        if destructive and not request.context.get("confirmed"):
            phrase = {"shutdown": "shut down", "restart": "restart"}[action]
            return SkillResult(
                f"Are you sure you want to {phrase} the machine? Say yes to confirm.",
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
