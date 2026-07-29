"""Media/player connector — play/pause/skip/volume via global media keys.

Mechanism: the OS media keys, which the focused media app (Spotify, a browser
tab, a player) responds to. Universal, so it reports installed+running always.
Low-impact (control_input), no confirmation. We can't VERIFY the app reacted to a
global key, so results are marked unverified — honest, not faked.
"""

from __future__ import annotations

from typing import List

from security.capabilities import Capability
from software.connector_base import (
    HOTKEY,
    Action,
    DetectResult,
    SoftwareConnector,
)

_CTRL = frozenset({Capability.CONTROL_INPUT})


class MediaConnector(SoftwareConnector):
    app_id = "media"
    display_name = "media"

    def detect(self) -> DetectResult:
        # Media keys work regardless of which player is focused.
        return DetectResult(installed=True, running=True,
                            detail="OS media keys")

    def is_running(self) -> bool:
        return True

    def launch(self) -> bool:
        return True

    def actions(self) -> List[Action]:
        m = self._mech
        return [
            Action("play_pause", "Play or pause whatever is playing.",
                   (r"\b(play|pause|resume)\s+(?:the\s+)?(music|media|song|video|it)\b",
                    r"\b(?:play|pause)\b", r"\bпушти\b|\bпаузирај\b"),
                   (HOTKEY,), _CTRL,
                   lambda p: m.send_media_key("play_pause")),
            Action("next_track", "Skip to the next track.",
                   (r"\b(next|skip)\s+(?:track|song|this)\b", r"\bследна песна\b"),
                   (HOTKEY,), _CTRL, lambda p: m.send_media_key("next")),
            Action("previous_track", "Go back to the previous track.",
                   (r"\b(previous|last)\s+(?:track|song)\b", r"\bпретходна песна\b"),
                   (HOTKEY,), _CTRL, lambda p: m.send_media_key("previous")),
            Action("volume_up", "Turn the media volume up.",
                   (r"\b(volume up|louder|turn it up)\b", r"\bпогласно\b"),
                   (HOTKEY,), _CTRL, lambda p: m.send_media_key("volume_up")),
            Action("volume_down", "Turn the media volume down.",
                   (r"\b(volume down|quieter|turn it down)\b", r"\bпотивко\b"),
                   (HOTKEY,), _CTRL, lambda p: m.send_media_key("volume_down")),
        ]
