"""Editing files by voice: dictate into them, replace text, open them to edit.

Writes are the only thing MEDO does that destroys information, so the tests
lean on the guards rather than the happy path: the whitelist, the text-only
rule, the confirmation before a replace, and the .bak that makes any edit
undoable.
"""

from __future__ import annotations

import pytest

from core.safety import PathWhitelist
from skills.base import SkillRequest
from skills.file_edit import (
    FileEditSkill,
    OpenInEditorSkill,
    apply_replace,
    backup,
    is_text_file,
)


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "shopping.txt").write_text("bread\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        "port: 8710\nhost: localhost\nport2: 8710\n", encoding="utf-8")
    (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff\x00binary")
    (tmp_path / "notes.md").write_text("# Notes\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def skill(workspace):
    return FileEditSkill(PathWhitelist([str(workspace)]))


async def _say(skill, text, confirmed=False):
    m = skill.match(text)
    assert m is not None, f"no pattern matched: {text!r}"
    return await skill.execute(
        SkillRequest(text=text, match=m, context={"confirmed": confirmed}))


# --- pure helpers -------------------------------------------------------------


def test_is_text_file_allows_text_and_refuses_binary(workspace):
    assert is_text_file(workspace / "shopping.txt")
    assert is_text_file(workspace / "config.yaml")
    assert not is_text_file(workspace / "photo.jpg")


def test_is_text_file_refuses_binary_wearing_a_text_suffix(workspace):
    sneaky = workspace / "corrupt.txt"
    sneaky.write_bytes(b"text\x00\x00more")     # NUL bytes => not really text
    assert not is_text_file(sneaky)


def test_apply_replace_prefers_an_exact_match():
    content = "Port and port and PORT"
    updated, count = apply_replace(content, "port", "slot")
    assert count == 1 and updated == "Port and slot and PORT"


def test_apply_replace_falls_back_to_case_insensitive():
    updated, count = apply_replace("Hello World", "hello", "Goodbye")
    assert count == 1 and updated == "Goodbye World"


def test_apply_replace_reports_no_match():
    updated, count = apply_replace("abc", "xyz", "1")
    assert count == 0 and updated == "abc"


def test_backup_copies_the_original(workspace):
    saved = backup(workspace / "shopping.txt")
    assert saved is not None and saved.name == "shopping.txt.bak"
    assert saved.read_text(encoding="utf-8") == "bread\n"


# --- patterns -----------------------------------------------------------------


@pytest.mark.parametrize("phrase", [
    "add to shopping.txt: milk and eggs",
    "append butter to shopping.txt",
    "write the new plan to notes.md",
    "add HEADER to the top of shopping.txt",
    "in config.yaml replace 8710 with 9000",
    "replace 8710 with 9000 in config.yaml",
    "додај во shopping.txt: сирење",
    "запиши млеко во shopping.txt",
    "во config.yaml замени 8710 со 9000",
])
def test_matches_edit_requests(skill, phrase):
    assert skill.match(phrase) is not None, phrase


@pytest.mark.parametrize("phrase", [
    "take a note buy milk",      # NotesSkill — the notes DB, not a file
    "note that the door is open",
    "read my notes",
    "find the file report",      # FilesSkill
    "what time is it",
    "play some music",
])
def test_does_not_steal_other_skills(skill, phrase):
    assert skill.match(phrase) is None, phrase


# --- dictating ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_append_with_the_colon_form(skill, workspace):
    r = await _say(skill, "add to shopping.txt: milk and eggs")
    assert r.success
    assert workspace.joinpath("shopping.txt").read_text(encoding="utf-8") == \
        "bread\nmilk and eggs\n"


@pytest.mark.asyncio
async def test_append_with_the_to_form(skill, workspace):
    await _say(skill, "append butter to shopping.txt")
    assert "butter" in workspace.joinpath("shopping.txt").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_append_adds_the_missing_newline(skill, workspace):
    workspace.joinpath("shopping.txt").write_text("bread", encoding="utf-8")  # no \n
    await _say(skill, "append butter to shopping.txt")
    assert workspace.joinpath("shopping.txt").read_text(encoding="utf-8") == \
        "bread\nbutter\n"


@pytest.mark.asyncio
async def test_prepend_goes_to_the_top(skill, workspace):
    await _say(skill, "add HEADER to the top of shopping.txt")
    assert workspace.joinpath("shopping.txt").read_text(
        encoding="utf-8").startswith("HEADER\nbread")


@pytest.mark.asyncio
async def test_macedonian_dictation(skill, workspace):
    r = await _say(skill, "додај во shopping.txt: сирење")
    assert r.speech.startswith("Запишав")
    assert "сирење" in workspace.joinpath("shopping.txt").read_text(encoding="utf-8")


# --- replacing (the destructive one) ------------------------------------------


@pytest.mark.asyncio
async def test_replace_asks_first_and_changes_nothing(skill, workspace):
    before = workspace.joinpath("config.yaml").read_text(encoding="utf-8")
    r = await _say(skill, "in config.yaml replace 8710 with 9000")
    assert r.needs_confirmation is True
    assert "2 times" in r.speech            # says exactly how much will change
    assert workspace.joinpath("config.yaml").read_text(encoding="utf-8") == before


@pytest.mark.asyncio
async def test_replace_after_confirmation_writes_and_backs_up(skill, workspace):
    r = await _say(skill, "in config.yaml replace 8710 with 9000", confirmed=True)
    assert r.success and r.data["count"] == 2
    assert workspace.joinpath("config.yaml").read_text(encoding="utf-8") == \
        "port: 9000\nhost: localhost\nport2: 9000\n"
    # The original survives, so the edit is undoable.
    assert workspace.joinpath("config.yaml.bak").read_text(encoding="utf-8") == \
        "port: 8710\nhost: localhost\nport2: 8710\n"


@pytest.mark.asyncio
async def test_replace_reports_a_string_that_is_not_there(skill, workspace):
    r = await _say(skill, "in config.yaml replace nothinghere with x")
    assert r.success is False and r.needs_confirmation is False


@pytest.mark.asyncio
async def test_macedonian_replace_asks_in_macedonian(skill):
    r = await _say(skill, "во config.yaml замени 8710 со 9000")
    assert r.needs_confirmation is True and r.speech.startswith("Ќе заменам")


# --- the guards ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_refuses_a_binary_file(skill, workspace):
    before = workspace.joinpath("photo.jpg").read_bytes()
    r = await _say(skill, "add caption to photo.jpg")
    assert r.success is False and "isn't a text file" in r.speech
    assert workspace.joinpath("photo.jpg").read_bytes() == before


@pytest.mark.asyncio
async def test_refuses_a_file_it_cannot_find(skill):
    r = await _say(skill, "add x to nonexistent.txt")
    assert r.success is False and "couldn't find" in r.speech


@pytest.mark.asyncio
async def test_cannot_touch_a_file_outside_the_whitelist(tmp_path, workspace):
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("classified\n", encoding="utf-8")
    try:
        skill = FileEditSkill(PathWhitelist([str(workspace)]))
        r = await _say(skill, "add oops to outside-secret.txt")
        assert r.success is False
        assert outside.read_text(encoding="utf-8") == "classified\n"
    finally:
        outside.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_no_whitelist_means_no_editing(tmp_path):
    skill = FileEditSkill(PathWhitelist([]))
    r = await _say(skill, "add x to shopping.txt")
    assert r.success is False and "whitelisted" in r.speech


def test_is_gated_by_the_pc_control_switch(skill):
    assert skill.controls_pc is True


# --- opening in an editor -----------------------------------------------------


@pytest.mark.parametrize("phrase", [
    "edit notes.md",
    "open notes.md in the editor",
    "уреди notes.md",
    "отвори notes.md во уредувач",
])
def test_editor_patterns(workspace, phrase):
    skill = OpenInEditorSkill(PathWhitelist([str(workspace)]), {})
    assert skill.match(phrase) is not None, phrase


@pytest.mark.asyncio
async def test_editor_launches_the_configured_command(workspace, monkeypatch):
    launched = []
    monkeypatch.setattr("skills.file_edit.run_detached", lambda c: launched.append(c))
    skill = OpenInEditorSkill(PathWhitelist([str(workspace)]),
                              {"editor": {"windows": "code", "darwin": "code",
                                          "linux": "code"}})
    r = await _say(skill, "edit notes.md")
    assert r.success and len(launched) == 1
    assert "notes.md" in launched[0] and launched[0].startswith("code ")


@pytest.mark.asyncio
async def test_editor_without_config_says_so(workspace):
    skill = OpenInEditorSkill(PathWhitelist([str(workspace)]), {})
    r = await _say(skill, "edit notes.md")
    assert r.success is False and "editor" in r.speech.lower()
