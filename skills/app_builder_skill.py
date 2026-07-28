"""Build a standalone app by voice/text, over core/app_builder.py.

"Make an app that tracks my water intake" / "build a tool called budgeter that
logs expenses" → MEDO scaffolds a self-contained project in its own folder under
apps_dir IN THE BACKGROUND (the coding agent takes minutes), then announces where
it landed and opens the folder. "make an app" with no description asks what it
should do and captures the answer (router reply-capture).

Registered only when app_builder.enabled — its presence is the opt-in.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re

from core.app_builder import AppBuildError, AppBuilder
from core.platform import open_path
from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

_MAKE_APP = re.compile(
    r"\b(?:make|build|create|write|scaffold|code)\s+(?:me\s+)?(?:a|an)\s+"
    r"(?:app|application|program|tool|script|website|web\s*app|web\s*site|game|"
    r"utility|dashboard)\b"
    r"(?:\s+called\s+(?P<name>[\w -]+?))?"
    r"(?:\s+(?:that|which|to|for|which\s+can)\s+(?P<spec>.+))?",
    re.IGNORECASE)
_CANCEL = re.compile(
    r"^\s*(?:never\s*mind|nevermind|cancel|forget\s+it|stop|nothing|no)\s*[.!]*$",
    re.IGNORECASE)


def _clean(text: str) -> str:
    return " ".join((text or "").split()).strip(" .")


class MakeAppSkill(Skill):
    name = "make_app"
    description = (
        "Build a standalone app or program from a description — MEDO scaffolds a "
        "self-contained project in its own folder.")
    patterns = [_MAKE_APP]

    def __init__(self, builder: AppBuilder, announcer=None) -> None:
        self._builder = builder
        self._announce = announcer
        self._pending_name = ""
        self._task: asyncio.Task | None = None
        self._busy = False

    def tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "spec": {"type": "string",
                                 "description": "what the app should do"},
                        "name": {"type": "string",
                                 "description": "optional name for the app"},
                    },
                    "required": ["spec"],
                },
            },
        }

    async def execute(self, request: SkillRequest) -> SkillResult:
        args = request.args or {}
        if request.context.get("captured_reply"):
            return self._begin(self._pending_name, request.text)
        gd = request.match.groupdict() if request.match else {}
        spec = str(args.get("spec") or gd.get("spec") or "").strip()
        name = str(args.get("name") or gd.get("name") or "").strip()
        if not spec:
            # bare "make an app" — ask, and capture the answer as the spec.
            self._pending_name = name
            return SkillResult("What should the app do?", await_reply=True)
        return self._begin(name, spec)

    def _begin(self, name: str, spec: str) -> SkillResult:
        spec = _clean(spec)
        if _CANCEL.match(spec):
            return SkillResult("Okay, never mind.")
        if len(spec) < 4:
            self._pending_name = name
            return SkillResult(
                "Tell me what the app should do — for example, 'make an app that "
                "tracks my daily water intake'.", await_reply=True)
        if self._busy:
            return SkillResult(
                "I'm already building one — let me finish that first.", success=False)
        self._busy = True
        self._task = asyncio.ensure_future(self._work(name, spec))
        label = _clean(name) or "it"
        return SkillResult(
            f"Okay — I'll build {label} in the background and let you know when "
            "it's ready.", data={"make_app": "started", "spec": spec})

    async def _work(self, name: str, spec: str) -> None:
        try:
            build = await self._builder.build(name, spec)
            if build.ok:
                with contextlib.suppress(Exception):
                    open_path(str(build.path))
                msg = (f"I've built {build.name} — {len(build.files)} file(s) in "
                       f"{build.path}. I've opened the folder for you.")
            else:
                msg = f"I tried to build that but got nothing usable — {build.error}."
        except AppBuildError as exc:
            msg = f"I couldn't build that: {exc}"
        finally:
            self._busy = False
        logger.info("make_app: %s", msg)
        if self._announce is not None:
            with contextlib.suppress(Exception):
                await self._announce(msg)
