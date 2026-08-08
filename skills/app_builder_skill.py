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

from core import mk
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

#: The Macedonian twin. Without it "направи апликација за прогноза" reached no
#: builder at all: the weather skill claimed it on a bare "прогноза" and read
#: out the temperature, and a plain "направи апликација" fell through to the
#: LLM, which invented a half-translated excuse about winget. The noun stems
#: end in \w* so the definite article ("апликацијаТА") comes along for free.
_MAKE_APP_MK = re.compile(
    rf"\b(?:{mk.MAKE}){mk.CLITICS}\s+(?:ед[нeaо]\w*\s+)?"
    r"(?:апликациј\w*|аплкациј\w*|програм\w*|алатк\w*|скрипт\w*|"
    r"веб\s*страниц\w*|веб\s*сајт\w*|веб\s*апликациј\w*|сајт\w*|игр[аиen]\w*)"
    r"(?:\s+(?:по\s+име|со\s+име|наречен\w*)\s+(?P<name>[\w -]+?))?"
    r"(?:\s+(?:за|што|шо|кој[аао]?|коишто|да)\s+(?P<spec>.+))?",
    re.IGNORECASE)

_CANCEL = re.compile(
    r"^\s*(?:never\s*mind|nevermind|cancel|forget\s+it|stop|nothing|no"
    r"|нема\s+врска|откажи|заборави|ништо|не)\s*[.!]*$",
    re.IGNORECASE)


def _clean(text: str) -> str:
    return " ".join((text or "").split()).strip(" .")


class MakeAppSkill(Skill):
    name = "make_app"
    description = (
        "Build a standalone app or program from a description — MEDO scaffolds a "
        "self-contained project in its own folder.")
    patterns = [_MAKE_APP, _MAKE_APP_MK]

    def __init__(self, builder: AppBuilder, announcer=None) -> None:
        self._builder = builder
        self._announce = announcer
        self._pending_name = ""
        self._pending_mk = False
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
            # The spec arrives in whatever language the follow-up was given in,
            # but the question was asked in the language of the ORIGINAL
            # request — keep answering in that one.
            return self._begin(self._pending_name, request.text, self._pending_mk)
        speak_mk = mk.is_cyrillic(request.text)
        gd = request.match.groupdict() if request.match else {}
        spec = str(args.get("spec") or gd.get("spec") or "").strip()
        name = str(args.get("name") or gd.get("name") or "").strip()
        if not spec:
            # bare "make an app" — ask, and capture the answer as the spec.
            self._pending_name = name
            self._pending_mk = speak_mk
            return SkillResult(
                "Што треба да прави апликацијата?" if speak_mk
                else "What should the app do?",
                await_reply=True, reply_is_open=True)
        return self._begin(name, spec, speak_mk)

    def _begin(self, name: str, spec: str, speak_mk: bool = False) -> SkillResult:
        spec = _clean(spec)
        if _CANCEL.match(spec):
            return SkillResult("Добро, нема врска." if speak_mk else "Okay, never mind.")
        if len(spec) < 4:
            self._pending_name = name
            self._pending_mk = speak_mk
            return SkillResult(
                "Кажи ми што треба да прави апликацијата — на пример, „направи "
                "апликација што ми го следи внесот на вода“." if speak_mk else
                "Tell me what the app should do — for example, 'make an app that "
                "tracks my daily water intake'.",
                await_reply=True, reply_is_open=True)
        if self._busy:
            return SkillResult(
                "Веќе градам една — да ја завршам прво неа." if speak_mk else
                "I'm already building one — let me finish that first.", success=False)
        self._busy = True
        self._task = asyncio.ensure_future(self._work(name, spec))
        label = _clean(name) or ("неа" if speak_mk else "it")
        return SkillResult(
            f"Добро — ќе ја изградам {label} во позадина и ќе ти јавам кога е "
            "готова." if speak_mk else
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
