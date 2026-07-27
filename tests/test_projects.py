"""Project organization: the sqlite store and the voice/text skill.

The store is exercised directly (pure sqlite in a tmp db); the skill drives a
real store through natural phrasings, including the "remembered project" so a
follow-up "add a task: X" lands without naming the project again.
"""

from __future__ import annotations

import pytest

from core.projects import ProjectStore
from skills.base import SkillRequest
from skills.projects_skill import ProjectsSkill


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "medo.sqlite")


# --- the store ----------------------------------------------------------------

def test_create_project_and_reject_duplicates(store):
    assert store.create_project("Kitchen Remodel") is True
    assert store.create_project("kitchen remodel") is False   # case-insensitive dup
    assert store.project_exists("KITCHEN REMODEL") is True


def test_add_task_autocreates_project_and_lists(store):
    assert store.add_task("Garden", "buy seeds") is True
    assert store.add_task("Garden", "till the beds") is True
    tasks = store.tasks("Garden")
    assert [t["text"] for t in tasks] == ["buy seeds", "till the beds"]
    assert all(t["done"] is False for t in tasks)
    assert store.tasks("nope") is None                        # missing project


def test_complete_task_marks_first_open_match(store):
    store.add_task("Garden", "buy seeds")
    store.add_task("Garden", "buy tools")
    assert store.complete_task("Garden", "seeds") == "buy seeds"
    assert store.complete_task("Garden", "seeds") is None     # already done
    open_now = store.tasks("Garden", include_done=False)
    assert [t["text"] for t in open_now] == ["buy tools"]


def test_list_projects_counts_open_and_done(store):
    store.add_task("A", "x")
    store.add_task("A", "y")
    store.complete_task("A", "x")
    store.create_project("B")
    projects = {p["name"]: p for p in store.list_projects()}
    assert projects["A"]["open"] == 1 and projects["A"]["done"] == 1
    assert projects["B"]["open"] == 0


def test_summary(store):
    store.add_task("A", "one")
    store.add_task("A", "two")
    store.complete_task("A", "one")
    s = store.summary("A")
    assert s["total"] == 2 and s["done"] == 1 and s["open"] == 1
    assert s["next_tasks"] == ["two"]
    assert store.summary("missing") is None


# --- the skill ----------------------------------------------------------------

def _skill(tmp_path):
    return ProjectsSkill(ProjectStore(tmp_path / "medo.sqlite"))


async def _run(skill, text):
    return await skill.execute(SkillRequest(text=text, match=skill.match(text)))


@pytest.mark.asyncio
async def test_skill_full_flow(tmp_path):
    skill = _skill(tmp_path)

    r = await _run(skill, "start a project called kitchen remodel")
    assert r.success and r.data["project"] == "kitchen remodel"

    r = await _run(skill, "add a task to kitchen remodel: get three quotes")
    assert r.success and r.data["task"] == "get three quotes"

    # follow-up with no project named uses the remembered one
    r = await _run(skill, "add a task: pick the tiles")
    assert r.success and r.data["project"] == "kitchen remodel"

    r = await _run(skill, "what's left on kitchen remodel")
    assert r.success and r.data["open"] == 2

    r = await _run(skill, "mark get three quotes done")
    assert r.success and "get three quotes" in r.speech

    r = await _run(skill, "list my projects")
    assert "kitchen remodel" in r.speech.lower()


@pytest.mark.asyncio
async def test_skill_complete_with_no_match(tmp_path):
    skill = _skill(tmp_path)
    await _run(skill, "start a project called garden")
    await _run(skill, "add a task to garden: water the plants")
    r = await _run(skill, "mark mow the lawn done in garden")
    assert r.success is False and "couldn't find" in r.speech.lower()


def test_status_how_needs_a_project_marker(tmp_path):
    skill = _skill(tmp_path)
    # the greeting must NOT be read as a status query for a project called "it"
    assert skill.match("how's it going") is None
    assert skill.match("how is it going today") is None
    # ...but an explicit "project" marker does route here
    assert skill.match("how's the garden project going") is not None
    assert skill.match("how is project taxes coming") is not None


@pytest.mark.asyncio
async def test_skill_tool_path(tmp_path):
    skill = _skill(tmp_path)
    r = await skill.execute(SkillRequest(
        text="", args={"action": "add_task", "project": "Taxes", "task": "gather receipts"}))
    assert r.success and r.data["project"] == "Taxes"
    r = await skill.execute(SkillRequest(
        text="", args={"action": "list_tasks", "project": "Taxes"}))
    assert r.data["open"] == 1


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
