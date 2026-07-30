"""Instant replies to greetings + pleasantries — no LLM (latency).

"How are you", "hello", "thanks", "good night" used to take the full LLM path
(15 s on a partially-offloaded 30B). They carry no real request, so a fast skill
answers them in <1 ms with a varied, in-character line. Registered LATE so any
real skill (a morning briefing on "good morning") still wins first.

The patterns are deliberately TIGHT and mostly END-ANCHORED so they catch the
pleasantry but never a real question that merely contains the words — "how are
you going to fix this?" is NOT small talk.
"""

from __future__ import annotations

import random
import re
from typing import Any

from core import mk
from skills.base import Skill, SkillRequest, SkillResult


class SmallTalkSkill(Skill):
    name = "smalltalk"
    description = (
        "Reply to greetings and pleasantries — hello, how are you, thanks, good "
        "night. Use for social small talk, not for questions or commands.")

    # category -> patterns. Checked most-specific first (how-are-you before the
    # bare greeting) so "how are you" doesn't read as a plain "hi".
    _CATEGORIES: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
        ("how_are_you", (
            re.compile(r"\bhow(?:\s+are\s+you|\s+are\s+things|\s+are\s+ya|"
                       r"\s*'re\s+you)\b"
                       r"(?:\s+(?:doing|today|feeling|now|holding\s+up|"
                       r"these\s+days))*\s*[?.!]*$", re.IGNORECASE),
            re.compile(r"\bhow'?s\s+it\s+going\b\s*[?.!]*$", re.IGNORECASE),
            re.compile(r"\bкако\s+си\b", re.IGNORECASE),
        )),
        ("thanks", (
            re.compile(r"^\s*(?:thanks|thank\s+you|thanks\s+a\s+lot|thank\s+you\s+"
                       r"so\s+much|cheers|much\s+appreciated|appreciate\s+it)\b"
                       r"[\s,.!]*(?:medo|so\s+much|a\s+lot|mate)?\s*[.!]*$",
                       re.IGNORECASE),
            re.compile(r"^\s*(?:фала|благодарам)\b", re.IGNORECASE),
        )),
        ("night", (
            re.compile(r"^\s*(?:good\s*night|goodnight|nighty\s*night)\b"
                       r"(?:\s+(?:medo|everyone|all|then|now))?\s*[.!]*$",
                       re.IGNORECASE),
            re.compile(r"\bдобра\s+ноќ\b", re.IGNORECASE),
        )),
        ("greet", (
            re.compile(r"^\s*(?:hi|hey|hello|yo|hiya|heya|howdy)\b"
                       r"(?:\s+(?:there|medo|again|friend))?\s*[?.!]*$",
                       re.IGNORECASE),
            re.compile(r"^\s*здраво\b", re.IGNORECASE),
        )),
    )

    patterns = [p for _cat, pats in _CATEGORIES for p in pats]

    _REPLIES: dict[str, dict[str, list[str]]] = {
        "how_are_you": {
            "en": ["Doing well — everything's running as it should. What can I do "
                   "for you?",
                   "All good here. How can I help?",
                   "Running smoothly, thanks. What do you need?"],
            "mk": ["Добро сум — сè работи како што треба. Со што да помогнам?",
                   "Сè е во ред. Како можам да помогнам?"],
        },
        "greet": {
            "en": ["Hello.", "Hi there.", "At your service.",
                   "Ready when you are."],
            "mk": ["Здраво.", "На располагање.", "Спремен сум."],
        },
        "thanks": {
            "en": ["Anytime.", "My pleasure.", "Happy to help.", "Of course."],
            "mk": ["Нема на што.", "Со задоволство.", "Секогаш."],
        },
        "night": {
            "en": ["Good night.", "Sleep well."],
            "mk": ["Добра ноќ.", "Пријатно спиење."],
        },
    }

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def _category(self, text: str) -> str:
        for cat, pats in self._CATEGORIES:
            if any(p.search(text) for p in pats):
                return cat
        return "greet"

    async def execute(self, request: SkillRequest) -> SkillResult:
        cat = self._category(request.text)
        lang = "mk" if mk.is_cyrillic(request.text) else "en"
        options = self._REPLIES[cat].get(lang) or self._REPLIES[cat]["en"]
        return SkillResult(self._rng.choice(options), data={"smalltalk": cat})

    def tool_schema(self) -> dict[str, Any]:
        # Not offered to the LLM — it's a fast-path shortcut, and the model
        # answering a greeting is exactly the latency we're avoiding.
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
