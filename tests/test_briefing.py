"""M8 morning briefing: assembly, degradation, LLM fallback, bilingual."""

from __future__ import annotations

import re
from datetime import datetime

import pytest

from skills.base import Skill, SkillRequest, SkillResult
from skills.briefing import BriefingSkill

NOW = lambda: datetime(2026, 7, 20, 8, 0)  # noqa: E731 - fixed clock for tests


class _CannedSkill(Skill):
    name = "canned"
    description = "test stub"

    def __init__(self, speech: str, success: bool = True):
        self._speech, self._success = speech, success

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult(self._speech, success=self._success)


class _BoomSkill(_CannedSkill):
    async def execute(self, request: SkillRequest) -> SkillResult:
        raise RuntimeError("feed exploded")


class _Reminder:
    def __init__(self, label: str, due_at: str):
        self.label, self.due_at = label, due_at


class _Reminders:
    def __init__(self, items):
        self._items = items

    def all(self):
        return list(self._items)


class _Facts:
    def __init__(self, hits):
        self._hits = hits

    def relevant(self, query: str, limit: int):
        return self._hits[:limit]


def _skill(**overrides) -> BriefingSkill:
    kw = dict(
        sections=["weather", "news", "reminders", "upcoming"],
        weather=_CannedSkill("It's 26 degrees and clear in Skopje."),
        news=_CannedSkill("Top headline: local assistant becomes world-class."),
        reminders=_Reminders([_Reminder("dentist", "2026-07-20T15:00")]),
        facts=_Facts(["car service tomorrow at 3 pm"]),
        rewrite=None,
        now=NOW,
    )
    kw.update(overrides)
    return BriefingSkill(**kw)


@pytest.mark.asyncio
async def test_assembles_all_sections_in_order():
    r = await _skill().execute(SkillRequest(text="morning briefing"))
    assert r.success
    assert r.speech.startswith("Good morning. It's Monday, July 20, 08:00.")
    for chunk in ("26 degrees", "Top headline", "dentist", "car service"):
        assert chunk in r.speech
    assert r.data["sections"] == 5


@pytest.mark.asyncio
async def test_failing_section_is_skipped_not_fatal():
    r = await _skill(news=_BoomSkill("")).execute(SkillRequest(text="brief me"))
    assert r.success
    assert "26 degrees" in r.speech and "Top headline" not in r.speech


@pytest.mark.asyncio
async def test_sections_are_config_toggled():
    r = await _skill(sections=["weather"]).execute(SkillRequest(text="brief me"))
    assert "26 degrees" in r.speech
    for dropped in ("Top headline", "dentist", "car service"):
        assert dropped not in r.speech


@pytest.mark.asyncio
async def test_rewrite_composes_and_failures_fall_back_to_raw():
    async def compose(mk: bool, raw: str) -> str:
        assert "26 degrees" in raw
        return "Here is your morning, sir."

    r = await _skill(rewrite=compose).execute(SkillRequest(text="brief me"))
    assert r.speech == "Here is your morning, sir."

    async def broken(mk: bool, raw: str) -> str:
        raise RuntimeError("LLM down")

    r2 = await _skill(rewrite=broken).execute(SkillRequest(text="brief me"))
    assert "26 degrees" in r2.speech  # raw sections spoken — never fails

    async def empty(mk: bool, raw: str) -> str:
        return ""  # no model available

    r3 = await _skill(rewrite=empty).execute(SkillRequest(text="brief me"))
    assert "26 degrees" in r3.speech


@pytest.mark.asyncio
async def test_macedonian_trigger_gets_macedonian_greeting_and_flag():
    seen = {}

    async def compose(mk: bool, raw: str) -> str:
        seen["mk"] = mk
        return ""

    r = await _skill(rewrite=compose).execute(SkillRequest(text="добро утро медо"))
    assert seen["mk"] is True
    assert r.speech.startswith("Добро утро.")
    assert "понеделник" in r.speech  # Monday, in Macedonian
    assert "Потсетници" in r.speech


def test_fast_path_patterns_cover_the_spec_triggers():
    s = _skill()
    for phrase in ("morning briefing", "give me the daily briefing",
                   "brief me", "добро утро медо"):
        assert s.match(phrase) is not None, phrase
    assert s.match("what's the weather") is None


def test_short_brief_phrasings_route_to_the_skill_not_the_llm():
    # "morning brief" (no -ing) took the LLM path, where the Claude-backed brain
    # invented calendar/mail "connectors" MEDO doesn't have. These must all be
    # caught on the fast path so news comes from RSS, not an ungated web search.
    s = _skill()
    for phrase in ("Morning brief", "morning brief", "daily brief",
                   "what's my morning brief", "give me the briefing"):
        assert s.match(phrase) is not None, phrase
    # ...without swallowing ordinary uses of the word "brief".
    for phrase in ("be brief please", "a brief summary of the meeting",
                   "keep it brief"):
        assert s.match(phrase) is None, phrase
