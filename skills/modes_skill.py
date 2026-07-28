"""Toggle MEDO's session modes by voice, text, or LLM tool.

Flips the shared :class:`~core.modes.SessionModes` the voice loop reads each
turn — continuous conversation (keep listening between turns) and interpreter
mode (speak each utterance back in the other language). These change MEDO's own
behaviour, not the computer, so they are never gated by the PC-control switch.
"""

from __future__ import annotations

import re
from typing import Any

from skills.base import Skill, SkillRequest, SkillResult

_LANG_NAME = {"en": "English", "mk": "Macedonian"}


class ModesSkill(Skill):
    name = "session_mode"
    description = (
        "Turn continuous conversation mode or live interpreter (translation) "
        "mode on or off."
    )

    patterns = [
        re.compile(r"\b(?:continuous|conversation)\s+(?:conversation\s+|chat\s+)?mode\b",
                   re.IGNORECASE),
        re.compile(r"\b(?:keep|stop)\s+listening\b", re.IGNORECASE),
        re.compile(r"\b(?:interpreter|translation|translator)\s+mode\b", re.IGNORECASE),
        re.compile(r"\b(?:start|stop|exit|end)\s+(?:interpreting|translating)\b",
                   re.IGNORECASE),
        re.compile(r"\bbe\s+my\s+(?:interpreter|translator)\b", re.IGNORECASE),
    ]

    def __init__(self, modes) -> None:
        self._modes = modes

    @staticmethod
    def _wants_off(text: str) -> bool:
        return bool(re.search(r"\b(?:off|stop|exit|end|disable|no)\b", text, re.IGNORECASE))

    async def execute(self, request: SkillRequest) -> SkillResult:
        text = request.text.lower()
        # LLM-tool path may pass an explicit mode + state.
        mode = (request.args.get("mode") or "").lower()
        interpreter = "interpret" in text or "translat" in text or mode == "interpreter"
        if not interpreter and mode not in ("continuous", ""):
            interpreter = mode == "interpreter"

        off = self._wants_off(text)
        if "state" in request.args:
            off = str(request.args["state"]).lower() in ("off", "false", "0", "stop")

        if interpreter:
            self._modes.interpreter = not off
            if self._modes.interpreter:
                a, b = self._modes.interpreter_langs
                return SkillResult(
                    f"Interpreter mode on. I'll translate between "
                    f"{_LANG_NAME.get(a, a)} and {_LANG_NAME.get(b, b)}. "
                    f"Say 'stop interpreting' to end.")
            return SkillResult("Interpreter mode off.")

        # Continuous conversation (also the "keep/stop listening" phrases).
        self._modes.continuous = not off
        if self._modes.continuous:
            return SkillResult("Continuous conversation on. I'll keep listening "
                               "after each reply.")
        return SkillResult("Continuous conversation off. Say the wake word "
                           "each time now.")

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string",
                                 "enum": ["continuous", "interpreter"]},
                        "state": {"type": "string", "enum": ["on", "off"]},
                    },
                    "required": ["mode", "state"],
                },
            },
        }
