"""Date & time — the M0 fast-path skill that proves routing end to end.

"what time is it" / "what's the date" never touch the LLM: they match a regex and
return in well under a millisecond. Also exposed as an LLM tool for M3.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from core import mk
from skills.base import Skill, SkillRequest, SkillResult

_DAYS_MK = ("понеделник", "вторник", "среда", "четврток",
            "петок", "сабота", "недела")
_MONTHS_MK = ("јануари", "февруари", "март", "април", "мај", "јуни", "јули",
              "август", "септември", "октомври", "ноември", "декември")


class DateTimeSkill(Skill):
    name = "datetime"
    description = "Report the current local time or date."
    routing_phrases = [
        "what time is it", "tell me the time", "what's the date today",
        "what day is it", "how late is it",
    ]

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
        # MK: "колку е часот", "кој датум е денес", "кој ден е денес".
        re.compile(r"\bколку\s+е\s+часот\b", re.IGNORECASE),
        re.compile(r"\bкое\s+време\s+е\b", re.IGNORECASE),
        re.compile(r"\bкој\s+(?:датум|ден)\s+е(?:\s+денес)?\b", re.IGNORECASE),
        re.compile(r"\bкој\s+е\s+денешниот\s+датум\b", re.IGNORECASE),
    ]

    def _wants_date(self, text: str) -> bool:
        return bool(re.search(r"\b(date|day|датум|ден|денешниот)\b", text, re.IGNORECASE))

    async def execute(self, request: SkillRequest) -> SkillResult:
        now = datetime.now()
        # LLM tool path may pass an explicit kind; otherwise infer from the text.
        kind = request.args.get("kind")
        if kind is None:
            kind = "date" if self._wants_date(request.text) else "time"

        if mk.is_cyrillic(request.text):
            # 24-hour and Macedonian month/day names — "%A %B" would emit
            # English ones whatever the locale, which is exactly the mismatch
            # that makes a bilingual assistant feel broken.
            speech = (f"Денес е {_DAYS_MK[now.weekday()]}, {now.day} "
                      f"{_MONTHS_MK[now.month - 1]} {now.year}." if kind == "date"
                      else f"Часот е {now:%H:%M}.")
        elif kind == "date":
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
