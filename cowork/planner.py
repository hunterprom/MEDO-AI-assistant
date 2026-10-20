"""S1 — turn a delegated task into a plan of things MEDO can actually do.

The planner asks the brain to decompose the task, then **resolves every step
against the live registry**. That order matters: the model proposes, the
registry disposes. A step naming a capability that does not exist does not
become a step MEDO pretends to do — it stays in the plan carrying the reason it
cannot be done, and gets said out loud.

Everything that decides *risk* is read from the skill, never from the model:
``side_effect`` from the declared security capabilities, ``requires_confirmation``
from the skill's own flag, ``high_impact`` from HIGH_IMPACT_CAPS. A model that
writes ``"side_effect": false`` next to a shutdown changes nothing.

Bounded on purpose (``cowork.max_steps``, default 6). MEDO's brain is local and
modest; a fifteen-step chain on it is a demo that fails in front of people. Past
the cap the honest answer is "that's too big — split it", and that is what this
returns.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from core import mk
from cowork.plan import Plan, Step
from security.capabilities import (
    HIGH_IMPACT_CAPS,
    SIDE_EFFECT_CAPS,
    effective_capabilities,
)

logger = logging.getLogger(__name__)

#: Skills that must never appear in a plan. Cowork itself, or a plan can plan a
#: plan; ``quit`` because ending MEDO's own session mid-task takes the executor
#: with it, and the user asking MEDO to close down does not need an orchestrator.
NEVER_PLANNED = frozenset({"cowork", "quit"})

#: Characters of a capability's description kept in the planner prompt.
_DESCRIPTION_CHARS = 110
#: A markdown fence around the JSON. The prompt forbids them; models add them.
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$", re.MULTILINE)
#: Named in the prompt, because "in the user's language" is not an instruction a
#: model can follow — it has no idea which language that is until it is told.
#: Asked in Macedonian, qwen3 read the plan back as "Report today's date".
LANGUAGE_NAMES = {"en": "English", "mk": "Macedonian"}
#: A brain that answered NOTHING after this long didn't refuse — it ran out of
#: time. Distinguishing the two changes what MEDO tells the user to do about it.
SLOW_PLAN_S = 60.0


def _now() -> float:
    """Indirection so a test can control the clock without patching ``time``
    itself — asyncio reads ``time.monotonic`` too, and a fake one breaks it."""
    return time.monotonic()

PLANNER_PROMPT = (
    "You are the planner for a voice assistant. Break the user's task into an "
    "ORDERED list of steps, where every step is performed by ONE capability "
    "from the list below. Rules:\n"
    "- Use at most {max_steps} steps. Fewer is better.\n"
    "- capability MUST be a name from the list, copied exactly. If a step "
    "cannot be done by any of them, still include the step and set capability "
    "to null — do NOT substitute a different capability.\n"
    "- args must only use the argument names given for that capability.\n"
    "- description is what the step does, in one short phrase, written in "
    "{language}.\n"
    "- If a critical value is missing from the task (which file, which city, "
    "which device), put ONE short question in \"question\" and still give your "
    "best plan.\n"
    "Reply with ONLY a JSON object, no prose, no markdown:\n"
    '{{"steps": [{{"description": "...", "capability": "name", '
    '"args": {{}}}}], "question": ""}}\n\n'
    "Capabilities:\n{catalogue}"
)


@dataclass(frozen=True)
class Ability:
    """One capability a plan may use, as the planner sees it.

    Built from a registered skill — which is also what a software connector
    action and a MEDO Link device capability already are, so connectors and
    devices arrive here for free rather than through a second code path.
    """

    name: str
    description: str
    params: dict[str, str] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    side_effect: bool = False
    requires_confirmation: bool = False
    high_impact: bool = False

    def prompt_line(self) -> str:
        """One catalogue line. Trimmed: sixty-odd skills at full description is
        ten kilobytes of prompt before the task is even mentioned, and the local
        brain's attention is the scarce thing here."""
        args = ", ".join(sorted(self.params)) or "no arguments"
        risk = " [side effect]" if self.side_effect else ""
        text = " ".join(self.description.split())
        if len(text) > _DESCRIPTION_CHARS:
            text = text[:_DESCRIPTION_CHARS].rsplit(" ", 1)[0] + "…"
        return f"- {self.name}({args}): {text}{risk}"


def _schema_params(skill: Any) -> tuple[dict[str, str], tuple[str, ...]]:
    """A skill's tool arguments, or empty when its schema is unusable."""
    try:
        params = (skill.tool_schema().get("function", {})
                  .get("parameters", {}) or {})
        props = params.get("properties") or {}
        described = {
            str(key): str((value or {}).get("description")
                          or (value or {}).get("type") or "value")
            for key, value in props.items()
        }
        return described, tuple(str(r) for r in (params.get("required") or []))
    except Exception:  # noqa: BLE001 - one broken schema must not kill planning
        logger.warning("cowork: unusable tool schema on %r",
                       getattr(skill, "name", skill), exc_info=True)
        return {}, ()


def ability_for(skill: Any) -> Ability:
    """Describe one skill as a plannable ability. Risk comes from the SKILL."""
    caps = effective_capabilities(skill)
    params, required = _schema_params(skill)
    return Ability(
        name=str(getattr(skill, "name", "")),
        description=str(getattr(skill, "description", "") or ""),
        params=params,
        required=required,
        side_effect=bool(caps & SIDE_EFFECT_CAPS)
        or bool(getattr(skill, "controls_pc", False))
        or bool(getattr(skill, "requires_confirmation", False)),
        requires_confirmation=bool(getattr(skill, "requires_confirmation", False)),
        high_impact=bool(caps & HIGH_IMPACT_CAPS),
    )


def catalogue(registry: Any, exclude: frozenset[str] = NEVER_PLANNED) -> list[Ability]:
    """Every capability a plan may use, read from the LIVE registry.

    Not a hardcoded list: whatever is registered right now — built-ins, plugins,
    MCP tools, connector actions, device capabilities, routines — is what MEDO
    can plan with, which is the same thing as what it can do.
    """
    out = []
    for skill in registry.all():
        name = str(getattr(skill, "name", ""))
        if not name or name in exclude:
            continue
        out.append(ability_for(skill))
    return out


def _json_objects(text: str):
    """Every BALANCED ``{...}`` in ``text``, in order, ignoring braces in strings.

    A greedy ``\\{.*\\}`` was the obvious version and the wrong one: a real
    model wraps its answer in prose, fences it, or (being a reasoning model)
    drafts a plan, changes its mind, and writes a second one. First-brace to
    last-brace then spans two different objects and parses as neither.
    """
    depth = start = 0
    in_string = escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0:
                    yield text[start:index + 1]


def _parse_plan(raw: str) -> dict | None:
    """The planner's JSON, tolerating the prose and fences a model wraps it in.

    Prefers the LAST object that actually carries steps: a model that reasons
    in the open writes its final answer last, and an early draft must not win
    over the plan it settled on.
    """
    if not raw:
        return None
    text = _FENCE.sub("", raw)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (ValueError, TypeError):
        pass
    best: dict | None = None
    for candidate in _json_objects(text):
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        if isinstance(data.get("steps"), list) and data["steps"]:
            best = data                       # keep looking; the last one wins
        elif best is None:
            best = data
    return best


class Planner:
    """Decompose a task into a plan bound to real capabilities.

    ``think`` is an async ``(prompt) -> str`` onto whichever brain is configured
    to plan; None means no brain, and the planner says so rather than guessing.
    ``policy`` is the live :class:`security.policy.PolicyEngine`: a step whose
    skill is denied RIGHT NOW (PC control switched off, say) is marked
    impossible at planning time instead of failing halfway through the task.
    """

    def __init__(self, registry: Any, think: Any = None, *, max_steps: int = 6,
                 policy: Any = None) -> None:
        self._registry = registry
        self._think = think
        self._max_steps = max(1, int(max_steps))
        self._policy = policy

    # -- the one entry point --------------------------------------------------

    async def plan(self, task: str, language: str = "en") -> Plan:
        speak_mk = language == "mk"
        task = (task or "").strip()
        plan = Plan(task=task, language=language)
        if not task:
            plan.refusal = ("Која задача да ја преземам?" if speak_mk
                            else "What would you like me to do?")
            return plan
        if self._think is None:
            plan.refusal = (
                "Мозокот ми е офлајн, па не можам да направам план."
                if speak_mk else
                "My brain is offline, so I can't plan that.")
            return plan

        abilities = {a.name: a for a in catalogue(self._registry)}
        prompt = PLANNER_PROMPT.format(
            max_steps=self._max_steps,
            language=LANGUAGE_NAMES.get(language, "the user's language"),
            catalogue="\n".join(a.prompt_line() for a in abilities.values()),
        ) + f"\n\nTask: {task}"
        started = _now()
        try:
            raw = await self._think(prompt)
        except Exception:  # noqa: BLE001 - a brain that errors is not a crash
            logger.warning("cowork: planner brain failed", exc_info=True)
            raw = ""
        elapsed = _now() - started
        data = _parse_plan(raw or "")
        if not data and not raw and elapsed >= SLOW_PLAN_S:
            # Measured: the 30B plans a two-step task in ~60-120 s and never
            # finished a nine-part one inside 300. "I couldn't work out a plan"
            # is true and useless — the user's actual remedy is to split the
            # task, and nothing in that sentence suggests it.
            logger.info("cowork: planning gave up after %.0fs", elapsed)
            plan.refusal = (
                "Ми треба предолго за тоа — веројатно е преголема задача. "
                "Подели ја на помали."
                if speak_mk else
                "That took longer than I can wait for — it's probably too big. "
                "Try splitting it into smaller tasks.")
            return plan
        if not data:
            # Log what it actually said. "I couldn't work out a plan" is the
            # right thing to SAY and a useless thing to debug from — the
            # difference between a timeout, a refusal and malformed JSON is
            # invisible without the reply itself.
            logger.info("cowork: unusable planner reply (%d chars): %.400r",
                        len(raw or ""), raw or "")
            plan.refusal = ("Не можев да составам план за тоа."
                            if speak_mk else
                            "I couldn't work out a plan for that.")
            return plan
        return self._resolve(plan, data, abilities)

    # -- resolution: the model proposes, the registry disposes ----------------

    def _resolve(self, plan: Plan, data: dict,
                 abilities: dict[str, Ability]) -> Plan:
        speak_mk = plan.language == "mk"
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            plan.refusal = ("Не можев да составам план за тоа."
                            if speak_mk else
                            "I couldn't work out a plan for that.")
            return plan
        if len(raw_steps) > self._max_steps:
            # Refuse rather than truncate. A plan cut off at step six does the
            # first six things of a task that needed nine — which is worse than
            # doing none of them, because it looks like it worked.
            plan.refusal = (
                f"Тоа е преголема задача — бара {len(raw_steps)} чекори, а јас "
                f"правам најмногу {self._max_steps}. Подели ја на помали."
                if speak_mk else
                f"That's too big for one go — it needs {len(raw_steps)} steps "
                f"and I cap myself at {self._max_steps}. Split it up for me.")
            return plan

        questions: list[str] = []
        for raw in raw_steps:
            if not isinstance(raw, dict):
                continue
            step = self._resolve_step(raw, abilities, plan.language, questions)
            if step is not None:
                plan.add_step(step)

        if not plan.steps:
            plan.refusal = ("Не можев да составам план за тоа."
                            if speak_mk else
                            "I couldn't work out a plan for that.")
            return plan

        # ONE question: the model's own, else the first missing argument.
        asked = str(data.get("question") or "").strip()
        plan.question = asked or (questions[0] if questions else "")
        return plan

    def _resolve_step(self, raw: dict, abilities: dict[str, Ability],
                      language: str, questions: list[str]) -> Step | None:
        speak_mk = language == "mk"
        description = str(raw.get("description") or raw.get("do") or "").strip()
        name = str(raw.get("capability") or raw.get("skill") or "").strip()
        args = raw.get("args")
        args = dict(args) if isinstance(args, dict) else {}
        if not description and not name:
            return None

        ability = abilities.get(name) or abilities.get(name.lower())
        if ability is None:
            # The honest step. It is kept and spoken — "I can't X" — instead of
            # being quietly replaced with the nearest capability that exists,
            # which is exactly how a plan comes to do the wrong thing
            # confidently.
            return Step(
                description=description or (name or "that"),
                problem=("немам вештина за тоа" if speak_mk
                         else "I have no skill for that"))

        skill = self._registry.get(ability.name)
        denied = self._denied(skill)
        if denied:
            return Step(description=description or ability.description,
                        problem=denied)

        unknown = [key for key in args if key not in ability.params]
        for key in unknown:
            # A model inventing an argument name is not a reason to fail the
            # step: the skill would ignore it anyway. Drop it, loudly.
            logger.info("cowork: dropping unknown arg %r for %s", key,
                        ability.name)
            args.pop(key, None)

        missing = [key for key in ability.required
                   if not str(args.get(key) or "").strip()]
        if missing:
            questions.append(self._ask_for(missing[0], ability, language))

        return Step(
            description=description or ability.description,
            capability=ability.name,
            args=args,
            side_effect=ability.side_effect,
            requires_confirmation=ability.requires_confirmation,
            high_impact=ability.high_impact,
        )

    def _denied(self, skill: Any) -> str:
        """The reason the policy engine refuses this skill right now, or ""."""
        if skill is None or self._policy is None:
            return ""
        try:
            decision = self._policy.gate_skill(skill)
        except Exception:  # noqa: BLE001 - a broken gate must not plan around it
            logger.warning("cowork: policy gate errored while planning",
                           exc_info=True)
            return "I can't check whether that's allowed"
        return decision.reason if decision.denied() else ""

    @staticmethod
    def _ask_for(param: str, ability: Ability, language: str) -> str:
        pretty = param.replace("_", " ")
        if language == "mk":
            return f"Што да ставам за {pretty}?"
        return f"What should I use for the {pretty}?"


def language_of(text: str) -> str:
    """Which language a task was delegated in — the language the plan is said in."""
    return "mk" if mk.is_cyrillic(text or "") else "en"
