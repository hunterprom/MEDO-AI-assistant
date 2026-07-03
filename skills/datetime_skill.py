"""Date & time — the M0 fast-path skill that proves routing end to end.

"what time is it" / "what's the date" never touch the LLM: they match a regex and
return in well under a millisecond. Also exposed as an LLM tool for M3.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from skills.base import Skill, SkillRequest, SkillResult


class DateTimeSkill(Skill):
    name = "datetime"
    description = "Report the current local time or date."

    # Note the ``'?s`` — STT and casual typing produce "whats"/"whens" without an
    # apostrophe, and these deterministic answers must never fall through to the LLM
    # (which will happily hallucinate a date).
    patterns = [
        re.compile(r"\bwhat(?:'?s| is)?\s+the\s+time\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+time\s+is\s+it\b", re.IGNORECASE),
        re.compile(r"\b(?:current\s+)?time\b", re.IGNORECASE),
        re.compile(r"\bwhat(?:'?s| is)?\s+(?:today'?s\s+)?(?:the\s+)?date\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+day\s+is\s+it\b", re.IGNORECASE),
        re.compile(r"\btoday'?s\s+date\b", re.IGNORECASE),
    ]

    def _wants_date(self, text: str) -> bool:
        return bool(re.search(r"\b(date|day)\b", text, re.IGNORECASE))

    async def execute(self, request: SkillRequest) -> SkillResult:
        now = datetime.now()
        # LLM tool path may pass an explicit kind; otherwise infer from the text.
        kind = request.args.get("kind")
        if kind is None:
            kind = "date" if self._wants_date(request.text) else "time"

        if kind == "date":
            speech = now.strftime("It's %A, %B %d, %Y.")
        else:
            speech = now.strftime("It's %I:%M %p.").lstrip("0")
        return SkillResult(speech=speech, data={"iso": now.isoformat(), "kind": kind})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["time", "date"],
                            "description": "Whether to report the time or the date.",
                        }
                    },
                    "required": [],
                },
            },
        }
