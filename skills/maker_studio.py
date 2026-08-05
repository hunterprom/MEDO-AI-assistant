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


#: "never mind" — drops a pending clarifying question instead of building.
_CANCEL = re.compile(
    r"^\s*(?:never\s*mind|nevermind|cancel|forget\s+it|stop|nothing|no)\s*[.!]*$",
    re.IGNORECASE)


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
        # What we were asked to make when we posed a clarifying question, so the
        # captured reply extends the brief instead of replacing it.
        self._pending_desc = ""

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
        # The answer to our OWN clarifying question arrives as free text — without
        # this the reply was parsed as a fresh request, found no description, and
        # the exchange dead-ended.
        if (request.context or {}).get("captured_reply"):
            answer = _clean(request.text)
            pending, self._pending_desc = self._pending_desc, ""
            if not answer or _CANCEL.match(answer):
                return SkillResult("Okay, never mind.")
            desc = f"{pending} — {answer}" if pending else answer
        else:
            desc = self._description(request)
        if not desc:
            self._pending_desc = ""
            return SkillResult(self._ask, await_reply=True)
        try:
            result = await self._get_engine().create(self._domain, desc)
        except Exception as exc:                # never crash the turn
            logger.warning("studio %s failed", self._domain.name, exc_info=True)
            return SkillResult(f"I couldn't {self._verb} that: {exc}",
                               success=False)
        if result.question:
            self._pending_desc = desc          # keep the brief for the reply
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


#: Nouns that essentially only ever name a physical part — safe with any making
#: verb ("design a bracket", "make me a gear").
_PART_NOUNS = (r"bracket|enclosure|mount|holder|clip|adapter|spacer|gear|knob|"
               r"tray|grommet|screw|bolt|nut|washer")
#: Idiom-prone words — "make a CASE for hiring", "make a STAND against X", "get a
#: HANDLE on it", "box him in". These need a fabrication verb (print/model) or a
#: physical modifier ("phone case", "project box") before the fast path claims
#: them; the LLM tool path still covers every other phrasing.
_AMBIG_NOUNS = r"case|stand|box|hook|handle"


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
        # unambiguous physical parts — any making verb
        re.compile(rf"\b(?:design|make|create|print|model)\s+(?:me\s+)?(?:a\s+|an\s+)?(?P<desc>(?:{_PART_NOUNS})\b.*)", re.IGNORECASE),
        # idiom-prone nouns: only with a fabrication verb…
        re.compile(rf"\b(?:print|model)\s+(?:me\s+)?(?:a\s+|an\s+)?(?P<desc>(?:{_AMBIG_NOUNS})\b.*)", re.IGNORECASE),
        # …or with a physical modifier in front ("a phone case", "a project box");
        # the lookahead keeps a bare article out of the modifier slot, so
        # "make a case for hiring" is left to the LLM.
        re.compile(rf"\b(?:design|make|create|print|model)\s+(?:me\s+)?(?:a\s+|an\s+)?(?P<desc>(?!(?:a|an|the)\s)\w+\s+(?:{_AMBIG_NOUNS})\b.*)", re.IGNORECASE),
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
    # Listing is pure sensing, so the skill itself isn't PC control; the ONE
    # actuating branch (opening the folder) checks the switch itself below.
    controls_pc = False
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
        # The LLM tool path has no regex match, so it needs its own way to ask
        # for "open" — otherwise the model could only ever list.
        wants_open = bool((request.args or {}).get("open"))
        if request.match is not None and "open" in request.text.lower():
            wants_open = True
        if wants_open:
            newest = projects[0]
            label = newest.get("description") or newest["path"]
            if not self._settings.safety.pc_control_enabled:
                return SkillResult(
                    f"Your last one is {label}, at {newest['path']} — turn PC "
                    f"control on and I'll open it.",
                    data={"path": newest["path"], "opened": False})
            from core.platform import open_path
            try:
                open_path(newest["path"])
            except Exception:
                logger.warning("could not open studio project %s",
                               newest["path"], exc_info=True)
                return SkillResult(
                    f"Your last one is at {newest['path']}, but I couldn't open "
                    f"the folder.", success=False,
                    data={"path": newest["path"], "opened": False})
            return SkillResult(f"Opening your last one — {label}.",
                               data={"path": newest["path"], "opened": True})
        names = ", ".join(p.get("description") or p["domain"] for p in projects[:6])
        return SkillResult(
            f"You have {len(projects)} Studio project"
            f"{'s' if len(projects) != 1 else ''}: {names}.",
            data={"count": len(projects)})

    def tool_schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": {
                "open": {"type": "boolean",
                         "description": "true to OPEN the most recent project's "
                                        "folder; omit/false to just list them"}},
                "required": []}}}
