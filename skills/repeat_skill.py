"""Say that again — verbatim.

Ask a small local model to "repeat that" and it RE-GENERATES: often a slightly
different answer — a different time, a reworded step — which is the worst kind of
fabrication because the user trusts a repeat as an exact replay. The router
already keeps the last spoken reply in ConversationMemory and hands it to every
skill as ``context['last_reply']``; this replays it word for word, on the fast
path, with no model in the loop, so a repeat can never drift.
"""

from __future__ import annotations

import re
from typing import Any

from core import mk
from skills.base import Skill, SkillRequest, SkillResult

NOTHING = "I haven't said anything yet."
NOTHING_MK = "Сè уште не сум кажал ништо."


class RepeatSkill(Skill):
    name = "repeat"
    controls_pc = False
    description = "Repeat MEDO's last spoken reply, word for word."

    patterns = [
        re.compile(r"\b(?:say|repeat)\s+(?:that|it)\s+"
                   r"(?:again|once\s+more|one\s+more\s+time)\b", re.IGNORECASE),
        # "what did you (just) say" — anchored so "what did you say ABOUT X"
        # (a follow-up, not a replay) still reaches the LLM.
        re.compile(r"\bwhat\s+did\s+you\s+(?:just\s+)?say"
                   r"(?:\s+(?:just\s+now|to\s+me|again))?\s*[?.!]*$", re.IGNORECASE),
        re.compile(r"^\s*(?:come\s+again|one\s+more\s+time|again\s+please)\s*[?.!]*$",
                   re.IGNORECASE),
        # Object REQUIRED: "can you repeat the address / the number" is a
        # different ask, not a verbatim replay.
        re.compile(r"\bcan\s+you\s+repeat\s+(?:that|it|yourself)\b", re.IGNORECASE),
        re.compile(r"\brepeat\s+(?:that|it|yourself)\b", re.IGNORECASE),
        # MK: "повтори", "повтори го/ја/тоа", "што рече", "уште еднаш". Anchored
        # so "повтори ја лекцијата" / "што рече тој за времето" (a real object /
        # a follow-up) fall through, mirroring the English guards.
        re.compile(r"\bповтори(?:\s+(?:го|ја|тоа))?\s*[?.!]*$"
                   r"|\bшто\s+рече\s*[?.!]*$"
                   r"|\bуште\s+еднаш\b", re.IGNORECASE),
    ]

    async def execute(self, request: SkillRequest) -> SkillResult:
        speak_mk = mk.is_cyrillic(request.text)
        last = request.context.get("last_reply")
        if not last or not str(last).strip():
            return SkillResult(NOTHING_MK if speak_mk else NOTHING, success=False)
        return SkillResult(str(last), data={"repeated": True})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
