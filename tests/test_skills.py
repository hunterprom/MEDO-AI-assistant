"""M2 skills: fast-path routing precedence + storage/parsing logic.

Routing is checked via ``SkillRegistry.find_match`` (pure matching, no side
effects) so we can assert the "10 scripted commands" reach the right skill
without launching apps, changing volume, or powering anything off.
"""

from __future__ import annotations

import pytest

from core.config import load_settings
from core.memory import NoteStore
from skills.base import SkillRequest
from skills.notes import NotesSkill
from skills.system import SystemInfoSkill
from skills.timers import parse_duration
from main import Announcer, build_registry


@pytest.fixture
def registry():
    settings = load_settings()
    return build_registry(settings, Announcer())


# The 10 scripted commands -> the skill that must own them.
@pytest.mark.parametrize(
    "utterance, expected",
    [
        ("what time is it", "datetime"),
        ("what's the date", "datetime"),
        ("set a timer for 5 minutes", "timers"),
        ("remind me in 10 seconds to stretch", "timers"),
        ("take a note buy milk", "notes"),
        ("read my notes", "notes"),
        ("open chrome", "apps"),
        ("find file report", "files"),
        ("volume up", "volume"),
        ("what's my battery", "system_info"),
        ("take a screenshot", "screenshot"),
        ("shut down the pc", "power"),
    ],
)
def test_routing_precedence(registry, utterance, expected):
    match = registry.find_match(utterance)
    assert match is not None, f"no skill matched {utterance!r}"
    assert match[0].name == expected


def test_reminder_to_sleep_is_a_timer_not_power(registry):
    # "sleep" appears but this is a reminder — timers must win over power.
    match = registry.find_match("remind me in 5 minutes to sleep")
    assert match is not None and match[0].name == "timers"


def test_start_spotify_is_apps_not_timer(registry):
    match = registry.find_match("start spotify")
    assert match is not None and match[0].name == "apps"


# --- storage / parsing -------------------------------------------------------
def test_parse_duration():
    assert parse_duration("5 minutes") == 300
    assert parse_duration("1 hour and 30 minutes") == 5400
    assert parse_duration("45 seconds") == 45
    assert parse_duration("no numbers here") == 0


def test_note_store_crud(tmp_path):
    store = NoteStore(tmp_path / "t.db")
    n = store.add("buy milk")
    assert n.id == 1
    assert [x.text for x in store.list()] == ["buy milk"]
    assert store.search("milk") and not store.search("eggs")
    assert store.delete(n.id) is True
    assert store.list() == []
    store.close()


@pytest.mark.asyncio
async def test_notes_skill_add_and_read(tmp_path):
    store = NoteStore(tmp_path / "t.db")
    skill = NotesSkill(store)
    add = await skill.execute(SkillRequest(text="take a note call the dentist",
                                           match=skill.patterns[0].search("take a note call the dentist")))
    assert add.success and "note" in add.speech.lower()
    read = await skill.execute(SkillRequest(text="read my notes",
                                            match=skill.patterns[2].search("read my notes")))
    assert "dentist" in read.speech
    store.close()


@pytest.mark.asyncio
async def test_system_info_runs_readonly():
    # psutil read-only call; safe to actually execute.
    skill = SystemInfoSkill()
    result = await skill.execute(SkillRequest(text="system status"))
    assert result.success and "%" not in result.speech  # spoken form uses "percent"
    assert "percent" in result.speech
