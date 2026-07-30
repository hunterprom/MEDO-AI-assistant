"""Maker Studio skills — turn "design a circuit for …" / "model a …" into a REAL
artifact by driving the shared engine (studio.engine): the brain writes library
code, MEDO executes it in the sandbox into a schematic / STL+STEP, checks it, and
saves a versioned project. Off unless ``studio.enabled``.

Generation runs the LLM + a sandboxed execution, so it isn't instant; it's a
deliberate build action, gated by the policy engine like any PC control.
"""

from __future__ import annotations

import logging
import re

from security.capabilities import Capability
from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).strip(" .?!,")


class _StudioSkill(Skill):
    """Shared drive-the-engine flow for a domain."""

    controls_pc = True
    capabilities = frozenset({Capability.RUN_COMMAND, Capability.WRITE_FILES})
    _verb = "build"
    _noun = "the design"
    _ask = "What should I make?"

    def __init__(self, settings, domain, *, engine=None) -> None:
        self._settings = settings
        self._domain = domain
        self._engine = engine

    def _get_engine(self):
        if self._engine is None:
            from studio.engine import StudioEngine

            self._engine = StudioEngine(self._settings)
        return self._engine

    def _description(self, request: SkillRequest) -> str:
        gd = request.match.groupdict() if request.match else {}
        return _clean(gd.get("desc") or str((request.args or {}).get("description")
                                             or ""))

    async def execute(self, request: SkillRequest) -> SkillResult:
        if not self._settings.studio.enabled:
            return SkillResult(
                "Maker Studio is off — enable studio.enabled in config so I can "
                "build that.", success=False, data={"reason": "disabled"})
        desc = self._description(request)
        if not desc:
            return SkillResult(self._ask, success=False)
        try:
            result = await self._get_engine().create(self._domain, desc)
        except Exception as exc:                # never crash the turn
            logger.warning("studio %s failed", self._domain.name, exc_info=True)
            return SkillResult(f"I couldn't {self._verb} that: {exc}",
                               success=False)
        if result.question:
            return SkillResult(result.question, await_reply=True)
        return self._report(result)

    def _report(self, result) -> SkillResult:
        fails = "; ".join(a.message for a in result.advisories
                          if a.level == "fail")
        if not result.ok:
            why = fails or result.error or "it didn't produce a usable artifact"
            return SkillResult(f"I tried to {self._verb} that but couldn't — {why}.",
                               success=False,
                               data={"error": result.error,
                                     "attempts": result.attempts})
        path = result.project.root if result.project is not None else None
        if path is not None:
            from core.platform import open_path
            try:
                open_path(str(path))
            except Exception:
                pass
        notes = "; ".join(a.message for a in result.advisories
                          if a.level in ("warn", "fail", "info"))
        extra = f" ({notes})" if notes else ""
        arts = ", ".join(result.artifacts) or "the files"
        return SkillResult(
            f"Done — {self._noun} is in {path} as {arts}.{extra}",
            data={"path": str(path) if path else "",
                  "artifacts": result.artifacts, "version": result.version})

    def tool_schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": {
                "description": {"type": "string",
                                "description": "what to make, in plain words"}},
                "required": ["description"]}}}


class Model3DSkill(_StudioSkill):
    name = "model_3d"
    description = ("Create a real, printable 3D model (STL + STEP) of a part from "
                   "a description — a bracket, enclosure, mount, screw, gear, etc. "
                   "Use when the user asks to model / 3D-print something.")
    _verb = "model"
    _noun = "the 3D model"
    _ask = "What should I model?"
    routing_phrases = ["3d print a phone stand", "model a bracket for a motor",
                       "design an enclosure for an ESP32", "make me a gear"]

    patterns = [
        re.compile(r"\b(?:3-?d|three-?d)\s*(?:model|print)\s+(?:of|for|me)?\s*(?:a\s+|an\s+)?(?P<desc>.+)", re.IGNORECASE),
        re.compile(r"\bmodel\s+(?:me\s+)?(?:a\s+|an\s+)?(?P<desc>.+?)\s+in\s+3-?d\b", re.IGNORECASE),
        re.compile(r"\b(?:design|make|create|print|model)\s+(?:me\s+)?(?:a\s+|an\s+)?(?P<desc>(?:bracket|enclosure|mount|case|holder|clip|adapter|stand|spacer|gear|knob|box|tray|hook|handle|grommet|screw|bolt|nut|washer)\b.*)", re.IGNORECASE),
    ]

    def __init__(self, settings, *, engine=None) -> None:
        from studio.domains.model3d import Model3DDomain

        super().__init__(settings, Model3DDomain(), engine=engine)


class DesignCircuitSkill(_StudioSkill):
    name = "design_circuit"
    description = ("Design a real electrical schematic / wiring diagram (SVG) from "
                   "a description — components, a microcontroller, sensors, power. "
                   "Use when the user asks to design a circuit / schematic / wiring.")
    _verb = "design"
    _noun = "the schematic"
    _ask = "What circuit should I design?"
    routing_phrases = ["design a circuit to blink an LED with an ESP32",
                       "draw a schematic for a 3.3V regulator",
                       "wire up a temperature sensor to an Arduino"]

    patterns = [
        re.compile(r"\b(?:design|draw|make|create|build|sketch)\s+(?:me\s+)?(?:a\s+|an\s+)?(?:circuit|schematic|wiring(?:\s+diagram)?)\s+(?:for\s+|of\s+|to\s+|that\s+)?(?P<desc>.+)", re.IGNORECASE),
        re.compile(r"\b(?:schematic|circuit|wiring)\s+(?:for|of)\s+(?P<desc>.+)", re.IGNORECASE),
        re.compile(r"\bwire\s+up\s+(?P<desc>.+)", re.IGNORECASE),
    ]

    def __init__(self, settings, *, engine=None) -> None:
        from studio.domains.schematic import SchematicDomain

        super().__init__(settings, SchematicDomain(), engine=engine)


class CodeBuildSkill(_StudioSkill):
    name = "build_code"
    description = ("Build a small self-contained script/tool from a description, "
                   "then RUN it in the sandbox to verify it works (its own "
                   "self-tests must pass). Use for 'write me a script/tool that…'.")
    _verb = "build"
    _noun = "the script"
    _ask = "What should the script do?"
    routing_phrases = ["write me a script that renames files by date",
                       "build a tool to convert csv to json",
                       "make a python function that validates an IBAN"]

    # NOTE: narrower than app_builder's MakeAppSkill (script/tool/utility, not
    # app/website/game); both are off by default and are alternative approaches.
    patterns = [
        re.compile(r"\b(?:build|write|make|create|code|scaffold|generate)\s+(?:me\s+)?(?:a\s+|an\s+)?(?:python\s+)?(?:script|tool|utility|cli|function|parser|converter)\b(?:\s+(?:that|to|which|for)\s+)?(?P<desc>.+)", re.IGNORECASE),
    ]

    def __init__(self, settings, *, engine=None) -> None:
        from studio.domains.code import CodeDomain

        super().__init__(settings, CodeDomain(), engine=engine)


class StudioProjectsSkill(Skill):
    """S5 — 'show my studio projects' / 'open my last model/schematic'."""

    name = "studio_projects"
    controls_pc = True                     # opening a folder acts on the PC
    description = ("List your Maker Studio projects, or open the most recent one "
                   "(schematic / 3D model / script).")
    routing_phrases = ["show my studio projects", "open my last 3d model",
                       "open my last schematic", "what have I made in studio"]
    patterns = [
        re.compile(r"\b(?:show|list|what are|what)\s+(?:my\s+)?(?:maker\s+)?studio\s+projects\b", re.IGNORECASE),
        re.compile(r"\b(?:open|show)\s+my\s+(?:last|latest|recent)\s+(?:studio\s+)?(?P<kind>schematic|circuit|3-?d\s*model|model|script|project)\b", re.IGNORECASE),
    ]

    _KIND = {"schematic": "schematic", "circuit": "schematic", "model": "model3d",
             "3d model": "model3d", "3dmodel": "model3d", "script": "code"}

    def __init__(self, settings) -> None:
        self._settings = settings

    async def execute(self, request: SkillRequest) -> SkillResult:
        from studio.projects import list_projects

        projects = list_projects(self._settings.studio.projects_dir)
        gd = request.match.groupdict() if request.match else {}
        kind = (gd.get("kind") or "").lower().replace(" ", "")
        want = self._KIND.get(kind, self._KIND.get(kind.replace("3-d", "3d"), ""))
        if want:
            projects = [p for p in projects if p.get("domain") == want] or projects
        if not projects:
            return SkillResult("You don't have any Maker Studio projects yet.",
                               success=False, data={"count": 0})
        if request.match is not None and "open" in request.text.lower():
            newest = projects[0]
            from core.platform import open_path
            try:
                open_path(newest["path"])
            except Exception:
                pass
            return SkillResult(
                f"Opening your last one — {newest.get('description') or newest['path']}.",
                data={"path": newest["path"]})
        names = ", ".join(p.get("description") or p["domain"] for p in projects[:6])
        return SkillResult(
            f"You have {len(projects)} Studio project"
            f"{'s' if len(projects) != 1 else ''}: {names}.",
            data={"count": len(projects)})

    def tool_schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": {}, "required": []}}}
