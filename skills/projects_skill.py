"""Project organization by voice or text, over core/projects.py.

"Start a project called kitchen remodel" · "add a task to it: get three quotes" ·
"what's left on the kitchen remodel" · "mark get quotes done" · "list my projects".
A lightweight tracker so MEDO can hold the shape of ongoing work between turns.

The most recently touched project is remembered, so "add a task: X" / "what's
left" without naming one fall back to it — the way people actually talk.
"""

from __future__ import annotations

import asyncio
import re

from core.projects import ProjectStore
from skills.base import Skill, SkillRequest, SkillResult

_CREATE = re.compile(
    r"\b(?:start|create|begin|open|set\s+up)\s+(?:a\s+)?(?:new\s+)?project\s+"
    r"(?:called\s+|named\s+|titled\s+)?(?P<name>.+)", re.IGNORECASE)
_LIST = re.compile(
    r"\b(?:list|show(?:\s+me)?)\s+(?:my\s+|the\s+|all\s+)?projects\b"
    r"|\bmy\s+projects\b|\bwhat\s+projects\b", re.IGNORECASE)
_ADD = re.compile(
    r"\badd\s+(?:a\s+)?task\s+to\s+(?:the\s+)?(?P<proj>.+?)\s*[:\-]\s*(?P<task>.+)",
    re.IGNORECASE)
_ADD_HERE = re.compile(
    r"\badd\s+(?:a\s+)?task\s*[:\-]\s*(?P<task>.+)"
    r"|\badd\s+(?P<task2>.+?)\s+to\s+(?:the\s+)?(?P<proj>.+?)\s+project\b",
    re.IGNORECASE)
_FOR = re.compile(
    r"\bfor\s+(?:project\s+)?(?P<proj>.+?)\s*,\s*add\s+(?:a\s+)?(?:task\s+)?(?P<task>.+)",
    re.IGNORECASE)
_COMPLETE = re.compile(
    r"\b(?:mark|set)\s+(?P<task>.+?)\s+(?:as\s+)?(?:done|complete|completed|finished)\b"
    r"(?:\s+(?:in|on|for)\s+(?:project\s+)?(?P<proj>.+))?", re.IGNORECASE)
_COMPLETE2 = re.compile(
    r"\b(?:complete|finish)\s+(?P<task>.+?)\s+(?:in|on|for)\s+(?:project\s+)?(?P<proj>.+)",
    re.IGNORECASE)
_STATUS = re.compile(
    r"\b(?:what(?:'s| is)?\s+(?:on|in|left\s+(?:on|in|for)?)"
    r"|(?:the\s+)?status\s+of|show\s+(?:me\s+)?(?:the\s+)?tasks?\s+(?:in|on|for)"
    r"|list\s+(?:the\s+)?tasks?\s+(?:in|on|for))\s+"
    r"(?:project\s+|the\s+)?(?P<proj>.+?)\s*[?.!]*$", re.IGNORECASE)
# "how's X going" needs an explicit "project" marker, so the greeting "how's it
# going" isn't read as a status query for a project called "it".
_STATUS_HOW = re.compile(
    r"\bhow(?:'s| is)\s+(?:the\s+)?(?P<proj>.+?)\s+project\b(?:\s+(?:going|doing|coming))?"
    r"|\bhow(?:'s| is)\s+project\s+(?P<proj2>.+?)(?:\s+(?:going|doing|coming))?\s*[?.!]*$",
    re.IGNORECASE)


def _clean(text: str) -> str:
    return " ".join((text or "").split()).strip(" .?!,")


class ProjectsSkill(Skill):
    name = "projects"
    description = (
        "Organize work into projects and tasks — create a project, add tasks, "
        "list what's left, and mark tasks done.")
    patterns = [_CREATE, _LIST, _ADD, _ADD_HERE, _FOR, _COMPLETE, _COMPLETE2,
                _STATUS, _STATUS_HOW]

    def __init__(self, store: ProjectStore) -> None:
        self._store = store
        self._last: str | None = None          # most recently touched project

    def tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string",
                                   "enum": ["create_project", "add_task",
                                            "list_projects", "list_tasks",
                                            "complete_task"]},
                        "project": {"type": "string", "description": "the project name"},
                        "task": {"type": "string",
                                 "description": "task text (add) or a phrase to match (complete)"},
                    },
                    "required": ["action"],
                },
            },
        }

    async def execute(self, request: SkillRequest) -> SkillResult:
        args = request.args or {}
        if args.get("action"):
            return await self._dispatch_tool(args)
        text = request.text
        # Order matters: add/complete before the broad status/create.
        if (m := _ADD.search(text)):
            return await self._add(m.group("proj"), m.group("task"))
        if (m := _FOR.search(text)):
            return await self._add(m.group("proj"), m.group("task"))
        if (m := _ADD_HERE.search(text)):
            return await self._add(m.groupdict().get("proj"),
                                   m.group("task") or m.group("task2"))
        if (m := _COMPLETE2.search(text)):
            return await self._complete(m.group("proj"), m.group("task"))
        if (m := _COMPLETE.search(text)):
            return await self._complete(m.groupdict().get("proj"), m.group("task"))
        if (m := _CREATE.search(text)):
            return await self._create(m.group("name"))
        if _LIST.search(text):
            return await self._list_projects()
        if (m := _STATUS.search(text)):
            return await self._status(m.group("proj"))
        if (m := _STATUS_HOW.search(text)):
            return await self._status(m.group("proj") or m.group("proj2"))
        return SkillResult("I didn't catch what to do with that project.",
                           success=False)

    # -- actions ------------------------------------------------------------

    async def _create(self, name: str) -> SkillResult:
        name = _clean(name)
        if not name:
            return SkillResult("What should the project be called?", success=False)
        made = await asyncio.to_thread(self._store.create_project, name)
        self._last = name
        if not made:
            return SkillResult(f"You already have a project called {name}.",
                               success=False, data={"project": name})
        return SkillResult(f"Started the project {name}.", data={"project": name})

    async def _add(self, project: str | None, task: str) -> SkillResult:
        task = _clean(task)
        project = _clean(project) if project else (self._last or "")
        if not project:
            return SkillResult(
                "Which project should I add that to?", success=False)
        if not task:
            return SkillResult("What's the task?", success=False)
        await asyncio.to_thread(self._store.add_task, project, task)
        self._last = project
        return SkillResult(f"Added to {project}: {task}.",
                           data={"project": project, "task": task})

    async def _complete(self, project: str | None, task: str) -> SkillResult:
        task = _clean(task)
        project = _clean(project) if project else (self._last or "")
        if not project:
            return SkillResult("Which project is that task in?", success=False)
        done = await asyncio.to_thread(self._store.complete_task, project, task)
        self._last = project
        if done is None:
            return SkillResult(
                f"I couldn't find an open task matching '{task}' in {project}.",
                success=False)
        return SkillResult(f"Marked done in {project}: {done}.",
                           data={"project": project, "task": done})

    async def _list_projects(self) -> SkillResult:
        projects = await asyncio.to_thread(self._store.list_projects)
        if not projects:
            return SkillResult("You don't have any projects yet.",
                               data={"projects": []})
        parts = [f"{p['name']} ({p['open']} open)" for p in projects]
        return SkillResult("Your projects: " + ", ".join(parts) + ".",
                           data={"projects": projects})

    async def _status(self, project: str) -> SkillResult:
        project = _clean(project)
        summary = await asyncio.to_thread(self._store.summary, project)
        if summary is None:
            return SkillResult(f"I don't have a project called {project}.",
                               success=False)
        self._last = project
        if summary["total"] == 0:
            return SkillResult(f"{project} has no tasks yet.", data=summary)
        if summary["open"] == 0:
            return SkillResult(
                f"{project} is all done — {summary['done']} task(s) complete.",
                data=summary)
        nxt = "; ".join(summary["next_tasks"])
        return SkillResult(
            f"{project}: {summary['open']} open, {summary['done']} done. "
            f"Next up: {nxt}.", data=summary)

    async def _dispatch_tool(self, args: dict) -> SkillResult:
        action = str(args.get("action") or "")
        project = str(args.get("project") or "")
        task = str(args.get("task") or "")
        if action == "create_project":
            return await self._create(project)
        if action == "add_task":
            return await self._add(project or None, task)
        if action == "complete_task":
            return await self._complete(project or None, task)
        if action == "list_tasks":
            return await self._status(project)
        if action == "list_projects":
            return await self._list_projects()
        return SkillResult("I didn't understand that project action.", success=False)
