"""Media playback control: play/pause and track skipping.

macOS drives the running player through AppleScript — Spotify when it's open,
Apple Music otherwise (both understand ``playpause`` / ``next track`` /
``previous track``). Windows/Linux fall back to the keyboard media keys via
pyautogui. Every path degrades to a clear spoken message instead of raising:
a missing player must never take the router down. No confirmation gate — the
worst case is a song you didn't ask for.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any

from core.platform import IS_MACOS
from skills.base import Skill, SkillRequest, SkillResult

# Canonical action -> what each backend runs / presses.
_MAC_COMMANDS = {"playpause": "playpause", "next": "next track", "previous": "previous track"}
_MEDIA_KEYS = {"playpause": "playpause", "next": "nexttrack", "previous": "prevtrack"}
_SPOKEN = {
    "playpause": "Toggled playback.",
    "next": "Skipping to the next track.",
    "previous": "Going back a track.",
}


def _mac_player() -> str:
    """Prefer Spotify when it's running; otherwise fall back to Apple Music."""
    try:
        out = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to (name of processes) contains "Spotify"'],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if out == "true":
            return "Spotify"
    except Exception:
        pass  # System Events unreachable -> Music is the safe default
    return "Music"


class MediaSkill(Skill):
    name = "media"
    controls_pc = True
    description = "Control music playback: play, pause, resume, next or previous track."

    patterns = [
        # "play SOME music" / "pause MY song" — a determiner between the verb
        # and the noun is the common spoken form and used to match nothing.
        re.compile(r"\b(?:play|pause|resume|stop)\s+(?:the|some|my|a)?\s*"
                   r"(?:music|song|track|playback)\b", re.IGNORECASE),
        # "put on / turn on some music", bare "play/pause the music".
        re.compile(r"\b(?:put|turn)\s+on\s+(?:some\s+|my\s+|the\s+)?music\b", re.IGNORECASE),
        re.compile(r"\b(?:next|skip(?:\s+(?:this|the))?)\s+(?:track|song)\b", re.IGNORECASE),
        re.compile(r"\b(?:previous|last)\s+(?:track|song)\b", re.IGNORECASE),
        # MK: "пушти музика", "паузирај ја песната", "следна песна"
        re.compile(r"\b(?:пушти|пуштиј|паузирај|запри|стопирај|продолжи)\s+"
                   r"(?:ја\s+|го\s+)?(?:музика(?:та)?|песна(?:та)?|нумера(?:та)?)\b",
                   re.IGNORECASE),
        re.compile(r"\bследна\s+(?:песна|нумера)\b|\bпретходна\s+(?:песна|нумера)\b",
                   re.IGNORECASE),
    ]

    def _action(self, text: str) -> str | None:
        """Map the utterance to a canonical action name, or None for neither.

        No default: this used to fall through to "playpause", so a call
        carrying no recognisable words toggled whatever had media focus — an
        action taken from no request at all.
        """
        if re.search(r"\b(?:next|skip|следна)\b", text):
            return "next"
        if re.search(r"\b(?:previous|last|back|претходна)\b", text):
            return "previous"
        if re.search(r"\b(?:play|pause|resume|stop|пушти|пуштиј|паузирај|"
                     r"запри|стопирај|продолжи)\b", text) or \
           re.search(r"\b(?:put|turn)\s+on\b.*\bmusic\b", text):
            return "playpause"   # play / pause / resume / stop / put on all toggle
        return None

    def _run(self, action: str) -> str:
        """Perform ``action`` on the current OS; always returns speech."""
        if IS_MACOS:
            player = _mac_player()
            try:
                subprocess.run(
                    ["osascript", "-e", f'tell application "{player}" to {_MAC_COMMANDS[action]}'],
                    capture_output=True, text=True, check=True,
                )
            except Exception:
                return f"I couldn't control {player}."
            return _SPOKEN[action].replace("playback.", f"playback in {player}.")
        # Windows/Linux: media-key fallback (same idiom as VolumeSkill).
        try:
            import pyautogui

            pyautogui.press(_MEDIA_KEYS[action])
        except Exception:
            return "I couldn't reach the media keys on this system."
        return _SPOKEN[action]

    async def execute(self, request: SkillRequest) -> SkillResult:
        action = self._action(request.text.lower())
        if action is None:
            # No default: a bare call used to toggle play/pause on whatever
            # had media focus, which is a real action taken from no request.
            return SkillResult("Play, pause, next or previous?", success=False)
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
                        "action": {"type": "string", "enum": ["playpause", "next", "previous"]}
                    },
                    "required": ["action"],
                },
            },
        }
