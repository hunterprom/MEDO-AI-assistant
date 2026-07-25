"""Morning briefing (M8): one command chains existing skills into a summary.

Triggers on the fast path ("morning briefing", "brief me", "добро утро медо")
and via the routine scheduler (config ``routines:`` asks "morning briefing").

Content pipeline — every section is optional (config ``briefing.sections``)
and degrades gracefully: a failing skill is skipped, never fatal. Sections:
greeting (date/time) → weather → news → active reminders → "upcoming" facts
(semantic recall, dentist-appointment style).

Assembly: the raw sections go through ONE LLM pass that rewrites them into a
flowing 30–60 s spoken briefing in the user's language, persona-aware (M7).
If the LLM is down the raw sections are spoken joined — the briefing never
fails entirely.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from datetime import datetime

from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")

#: Rewrites (macedonian: bool, raw_sections: str) -> composed briefing ("" = no LLM).
Rewriter = Callable[[bool, str], Awaitable[str]]

_DAYS_MK = ("понеделник", "вторник", "среда", "четврток",
            "петок", "сабота", "недела")


class BriefingSkill(Skill):
    name = "briefing"
    description = (
        "Deliver the morning briefing: date and time, weather, top headlines, "
        "active reminders, and upcoming plans."
    )
    routing_phrases = [
        "give me the rundown for today", "what's on for today",
        "catch me up on my day", "start my day", "what do I need to know today",
        "fill me in on the morning",
    ]
    patterns = [
        # "brief" and "briefing" are both natural ("morning brief", "morning
        # briefing", "daily brief"). Requiring the full word "briefing" here is
        # what sent a plain "morning brief" to the LLM path, where the brain
        # hallucinated calendar/mail "connectors" MEDO doesn't have.
        re.compile(r"\b(?:morning|daily)\s+brief(?:ing)?\b", re.IGNORECASE),
        re.compile(r"\bbrief\s+me\b", re.IGNORECASE),
        re.compile(r"\bbriefing\b", re.IGNORECASE),          # standalone
        re.compile(r"добро\s+утро", re.IGNORECASE),  # "добро утро медо" included
    ]

    def __init__(self, sections: list[str], weather: Skill | None,
                 news: Skill | None, reminders=None, facts=None,
                 rewrite: Rewriter | None = None, now=datetime.now) -> None:
        self._sections = [s.strip().lower() for s in sections]
        self._weather = weather
        self._news = news
        self._reminders = reminders
        self._facts = facts
        self._rewrite = rewrite
        self._now = now  # injectable for tests

    # -- sections (each returns text or "", never raises out) -----------------

    def _greeting(self, mk: bool) -> str:
        now = self._now()
        if mk:
            return (f"Добро утро. Денес е {_DAYS_MK[now.weekday()]}, "
                    f"{now.day}.{now.month}., {now:%H:%M} часот.")
        return f"Good morning. It's {now:%A, %B %d}, {now:%H:%M}."

    async def _from_skill(self, skill: Skill | None, ask: str) -> str:
        if skill is None:
            return ""
        result = await skill.execute(SkillRequest(text=ask))
        return result.speech if result.success else ""

    def _reminders_text(self, mk: bool) -> str:
        if self._reminders is None:
            return ""
        pending = list(self._reminders.all())[:3]
        if not pending:
            return ""
        items = []
        for r in pending:
            due = str(getattr(r, "due_at", "")).replace("T", " ")[:16]
            items.append(f"{getattr(r, 'label', r)} ({due})")
        joined = "; ".join(items)
        return (f"Потсетници: {joined}." if mk else f"Reminders: {joined}.")

    def _upcoming_text(self, mk: bool) -> str:
        if self._facts is None:
            return ""
        hits = self._facts.relevant(
            "appointment, plan, or schedule for today or tomorrow", 3)
        if not hits:
            return ""
        joined = "; ".join(hits)
        return (f"Вреди да се запомни: {joined}." if mk
                else f"Worth remembering: {joined}.")

    # -- assembly -------------------------------------------------------------

    async def execute(self, request: SkillRequest) -> SkillResult:
        mk = bool(_CYRILLIC.search(request.text))
        parts = [self._greeting(mk)]
        gather = {
            "weather": lambda: self._from_skill(self._weather, "what's the weather"),
            "news": lambda: self._from_skill(self._news, "what's the news"),
        }
        for section in self._sections:
            try:
                if section in gather:
                    text = await gather[section]()
                elif section == "reminders":
                    text = self._reminders_text(mk)
                elif section == "upcoming":
                    text = self._upcoming_text(mk)
                else:
                    continue
                if text:
                    parts.append(text)
            except Exception:  # one bad section must not kill the briefing
                logger.exception("briefing section %r failed", section)
        raw = " ".join(parts)

        if self._rewrite is not None:
            try:
                composed = (await self._rewrite(mk, raw)).strip()
                if composed:
                    return SkillResult(composed, data={"sections": len(parts)})
            except Exception:  # LLM down -> speak the raw sections instead
                logger.exception("briefing rewrite failed; speaking raw sections")
        return SkillResult(raw, data={"sections": len(parts)})
