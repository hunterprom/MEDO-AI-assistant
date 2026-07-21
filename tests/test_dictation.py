"""Dictation mode: MEDO writes down what you say instead of answering it."""

from __future__ import annotations

import pytest

from core.config import load_settings
from core.modes import SessionModes
from skills.base import SkillRequest
from skills.dictate import STOP_DICTATION, DictateSkill, target_path


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    s.conversation.dictation_dir = str(tmp_path)
    s.conversation.dictation_file = "dictation.md"
    return s


@pytest.fixture
def modes():
    return SessionModes()


@pytest.mark.parametrize("phrase", [
    "take dictation", "start dictating", "write this down for me",
    "take this down for me", "take dictation into ideas.md",
    "земи диктат", "запишувај", "пиши што ќе кажам",
])
def test_start_patterns(settings, modes, phrase):
    assert DictateSkill(settings, modes).match(phrase) is not None, phrase


@pytest.mark.parametrize("phrase", [
    "stop dictation", "stop dictating", "dictation off", "end the dictation",
    "стоп диктат", "прекини со диктирање", "доста пишување",
])
def test_stop_phrases(phrase):
    assert STOP_DICTATION.search(phrase) is not None, phrase


def test_stop_does_not_fire_on_ordinary_speech():
    for line in ("the meeting is at four", "write the report tomorrow",
                 "запиши дека треба леб"):
        assert STOP_DICTATION.search(line) is None, line


def test_target_path_defaults_and_names(settings, tmp_path):
    assert target_path(settings) == tmp_path / "dictation.md"
    assert target_path(settings, "ideas") == tmp_path / "ideas.md"
    assert target_path(settings, "notes dot md") == tmp_path / "notes.md"


def test_target_path_cannot_escape_the_dictation_dir(settings, tmp_path):
    # A spoken name is never a path: separators are stripped, not followed.
    for spoken in ("../../etc/passwd", r"..\..\secrets.yaml", "/tmp/evil.md"):
        assert target_path(settings, spoken).parent == tmp_path


@pytest.mark.asyncio
async def test_starting_flips_the_mode_and_creates_the_file(settings, modes, tmp_path):
    skill = DictateSkill(settings, modes)
    r = await skill.execute(SkillRequest(text="take dictation",
                                         match=skill.match("take dictation")))
    assert r.success and modes.dictating is True
    assert modes.dictation_path == str(tmp_path / "dictation.md")
    assert (tmp_path / "dictation.md").exists()


@pytest.mark.asyncio
async def test_starting_with_a_named_file(settings, modes, tmp_path):
    skill = DictateSkill(settings, modes)
    phrase = "take dictation into ideas.md"
    await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert modes.dictation_path == str(tmp_path / "ideas.md")


@pytest.mark.asyncio
async def test_macedonian_start_answers_in_macedonian(settings, modes):
    skill = DictateSkill(settings, modes)
    r = await skill.execute(SkillRequest(text="земи диктат",
                                         match=skill.match("земи диктат")))
    assert r.success and modes.dictating is True
    assert "запишувам" in r.speech.lower()


def test_is_gated_by_the_pc_control_switch(settings, modes):
    assert DictateSkill(settings, modes).controls_pc is True
