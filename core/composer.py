"""Routines: chaining skills together, in YAML, without writing Python.

MEDO already has thirty-odd skills. What it did not have was a way to say "when
I say good morning, do these four things" — that required a new skill, in
Python, registered in main.py. This module makes it a file:

    # routines/morning.yaml
    name: morning briefing
    description: Weather, news, and what's on today.
    trigger:
      phrase: ["good morning", "добро утро"]
      at: "07:30"
      days: [mon, tue, wed, thu, fri]
    params:
      city: Skopje
    steps:
      - say: "Good morning."
      - skill: weather
        args: {city: "{city}"}
      - skill: news

A routine is registered as an ordinary :class:`~skills.base.Skill`, which is the
whole design. It gets the fast path, the LLM tool schema, the policy gate and
the confirmation machinery for free, and nothing in the router needs to know
routines exist.

Three rules that are not negotiable, and why:

* **A routine cannot call a skill that requires confirmation.** Checked when it
  is loaded, refused with a message naming the step. A confirmation prompt
  halfway through a five-step routine has nobody to answer it when the routine
  fired on a timer, and when it fired on a phrase it would turn one "yes" into
  permission for whatever the rest of the file said. Routines are for
  composition, not for privilege.
* **Templating substitutes, it does not evaluate.** ``{city}`` is a lookup in a
  dict. There is no expression language, because a config file that can compute
  is a config file that can be made to compute something else.
* **A failing step does not kill the routine.** It is reported and the rest
  runs — a morning briefing that loses the news should still tell you the
  weather.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

#: Where standalone routine files live, one routine per file.
ROUTINES_DIRNAME = "routines"

#: ``{name}`` — a bare identifier only. Anything else is left alone rather than
#: guessed at, so a literal brace in a phrase survives.
_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

#: A routine that ran away would be a config file with a loop in it. There is no
#: loop construct, but a file can still list a thousand steps.
MAX_STEPS = 20


class RoutineError(ValueError):
    """A routine definition MEDO refuses to load, with the reason."""


def render(template: str, context: dict[str, Any]) -> str:
    """Substitute ``{name}`` placeholders from ``context``.

    An unknown placeholder is left EXACTLY as written rather than blanked. A
    routine whose ``{city}`` silently became "" would send the weather skill a
    request for nowhere and get a confusing answer; left in place, the mistake
    is visible in what MEDO says.
    """
    if not template:
        return ""

    def swap(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            return match.group(0)
        value = context[key]
        return "" if value is None else str(value)

    return _PLACEHOLDER.sub(swap, template)


def render_args(args: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """:func:`render` over a step's arguments, strings only (nested included)."""
    out: dict[str, Any] = {}
    for key, value in (args or {}).items():
        if isinstance(value, str):
            out[key] = render(value, context)
        elif isinstance(value, dict):
            out[key] = render_args(value, context)
        elif isinstance(value, list):
            out[key] = [render(v, context) if isinstance(v, str) else v
                        for v in value]
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class Step:
    """One thing a routine does: run a skill, or say a line."""

    skill: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    say: str | None = None
    #: Store this step's speech under this name, for a later ``{placeholder}``.
    into: str | None = None

    def describe(self) -> str:
        return f"say {self.say!r}" if self.skill is None else f"skill {self.skill}"


@dataclass(frozen=True)
class Trigger:
    """How a routine starts: a spoken phrase, a time of day, or an event."""

    phrases: tuple[str, ...] = ()
    at: str = ""
    days: tuple[str, ...] = ()
    event: str = ""

    @property
    def scheduled(self) -> bool:
        return bool(self.at)


@dataclass(frozen=True)
class Routine:
    """One user-defined chain of skills."""

    name: str
    description: str
    steps: tuple[Step, ...]
    trigger: Trigger = field(default_factory=Trigger)
    params: dict[str, Any] = field(default_factory=dict)
    source: str = ""              # the file it came from, for error messages

    @property
    def skill_name(self) -> str:
        """The registry name: ``routine_morning_briefing``."""
        slug = re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")
        return f"routine_{slug or 'unnamed'}"


def parse_routine(data: dict, source: str = "") -> Routine:
    """One parsed + validated routine. Raises :class:`RoutineError` with the why."""
    if not isinstance(data, dict):
        raise RoutineError(f"{source or 'routine'}: expected a YAML mapping")
    name = str(data.get("name") or "").strip()
    if not name:
        raise RoutineError(f"{source or 'routine'}: needs a 'name'")
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise RoutineError(f"routine {name!r}: needs a non-empty 'steps' list")
    if len(raw_steps) > MAX_STEPS:
        raise RoutineError(
            f"routine {name!r}: {len(raw_steps)} steps is past the {MAX_STEPS} "
            f"limit — split it into smaller routines")

    steps: list[Step] = []
    for index, raw in enumerate(raw_steps, start=1):
        if isinstance(raw, str):
            # Shorthand: a bare string is a skill name with no arguments.
            steps.append(Step(skill=raw.strip()))
            continue
        if not isinstance(raw, dict):
            raise RoutineError(
                f"routine {name!r} step {index}: expected 'skill:' or 'say:'")
        skill = str(raw.get("skill") or "").strip() or None
        say = raw.get("say")
        if not skill and not say:
            raise RoutineError(
                f"routine {name!r} step {index}: needs either 'skill' or 'say'")
        args = raw.get("args") or {}
        if not isinstance(args, dict):
            raise RoutineError(
                f"routine {name!r} step {index}: 'args' must be a mapping")
        steps.append(Step(skill=skill, args=args,
                          say=None if say is None else str(say),
                          into=(str(raw["into"]) if raw.get("into") else None)))

    raw_trigger = data.get("trigger") or {}
    if not isinstance(raw_trigger, dict):
        raise RoutineError(f"routine {name!r}: 'trigger' must be a mapping")
    phrases = raw_trigger.get("phrase") or []
    if isinstance(phrases, str):
        phrases = [phrases]
    trigger = Trigger(
        phrases=tuple(str(p).strip() for p in phrases if str(p).strip()),
        at=str(raw_trigger.get("at") or "").strip(),
        days=tuple(str(d).strip().lower()[:3]
                   for d in (raw_trigger.get("days") or []) if str(d).strip()),
        event=str(raw_trigger.get("event") or "").strip(),
    )
    if trigger.at and not re.fullmatch(r"\d{1,2}:\d{2}", trigger.at):
        raise RoutineError(
            f"routine {name!r}: trigger time must be HH:MM, got {trigger.at!r}")
    params = data.get("params") or {}
    if not isinstance(params, dict):
        raise RoutineError(f"routine {name!r}: 'params' must be a mapping")

    return Routine(name=name,
                   description=str(data.get("description") or
                                   f"Run the {name} routine."),
                   steps=tuple(steps), trigger=trigger,
                   params=dict(params), source=source)


def load_routine_files(directory: str | Path) -> tuple[list[Routine], list[str]]:
    """Every ``*.yaml`` in ``directory``. Returns ``(routines, errors)``.

    Errors are collected rather than raised: one malformed routine must not stop
    the other four from loading, and the message has to survive to be shown.
    """
    path = Path(directory)
    if not path.is_dir():
        return [], []
    import yaml

    routines: list[Routine] = []
    errors: list[str] = []
    for file in sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml")):
        try:
            data = yaml.safe_load(file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a bad file names itself
            errors.append(f"{file.name}: could not be read ({exc})")
            continue
        if data is None:
            continue
        try:
            routines.append(parse_routine(data, source=file.name))
        except RoutineError as exc:
            errors.append(str(exc))
    return routines, errors


class RoutineSkill(Skill):
    """A routine, wearing the ordinary Skill contract.

    Being a real skill is what makes a routine routable: the fast path matches
    its phrases, the LLM sees its tool schema, and the policy engine gates it
    like anything else.
    """

    controls_pc = False

    def __init__(self, routine: Routine, registry) -> None:
        self._routine = routine
        self._registry = registry
        self.name = routine.skill_name
        self.description = routine.description
        # Phrases are matched LITERALLY, via the SAME compiler that custom
        # trigger phrases already use (core/triggers.py). Reused rather than
        # reimplemented because it has already learned the awkward cases: the
        # phrase is escaped (it is words, not regex, and a config file must never
        # be able to inject a catastrophic pattern into the hot path), inner
        # whitespace is made flexible so an odd transcript still matches, and a
        # word boundary is added only where the phrase really starts or ends in a
        # word character — "what's up?" would otherwise compile to a pattern that
        # can never match.
        from core.triggers import phrase_to_pattern

        self.patterns = [pattern for pattern in
                         (phrase_to_pattern(p) for p in routine.trigger.phrases)
                         if pattern is not None]
        self.routing_phrases = list(routine.trigger.phrases)
        # A routine INHERITS what its steps can do. Without this it declared
        # `controls_pc = False` — true of the wrapper, false of the thing it
        # runs — and the router's policy gate saw a harmless skill. With PC
        # control switched OFF, calling `open_website` directly was refused and
        # the identical call inside a routine went through: a config file that
        # silently re-enabled the master actuation switch. Declaring the union
        # puts the whole routine behind the SAME gate as its most privileged
        # step, using the machinery that already exists rather than a second
        # check inside the step loop.
        from security.capabilities import effective_capabilities

        caps: set = set()
        controls = False
        for step in routine.steps:
            if not step.skill:
                continue
            target = registry.get(step.skill)
            if target is None:
                continue
            controls = controls or bool(getattr(target, "controls_pc", False))
            caps |= set(effective_capabilities(target))
        self.controls_pc = controls
        self.capabilities = frozenset(caps)

    @property
    def routine(self) -> Routine:
        return self._routine

    def tool_schema(self) -> dict[str, Any]:
        properties = {
            key: {"type": "string", "description": f"Value for {key}."}
            for key in self._routine.params
        }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": properties},
            },
        }

    async def execute(self, request: SkillRequest) -> SkillResult:
        # Parameters: the routine's declared defaults, overridden by whatever
        # the model passed as tool arguments. The fast path passes nothing, so a
        # spoken routine runs on its defaults, which is what a phrase trigger
        # means.
        context: dict[str, Any] = dict(self._routine.params)
        context.update({k: v for k, v in (request.args or {}).items()
                        if k in self._routine.params})
        spoken: list[str] = []
        failures: list[str] = []

        for index, step in enumerate(self._routine.steps, start=1):
            try:
                said = await self._run_step(step, request, context)
            except Exception:  # noqa: BLE001 - one bad step, not a dead routine
                logger.exception("routine %s: step %d (%s) raised",
                                 self._routine.name, index, step.describe())
                failures.append(step.skill or f"step {index}")
                continue
            if said is None:
                failures.append(step.skill or f"step {index}")
                continue
            if said:
                spoken.append(said)
                context["previous"] = said
                if step.into:
                    context[step.into] = said

        speech = " ".join(s for s in spoken if s).strip()
        if failures:
            # Say what did NOT happen. A routine that quietly drops a step is
            # the worst kind of automation: it looks like it worked.
            missed = ", ".join(f.replace("_", " ") for f in failures)
            speech = (f"{speech} I couldn't do {missed}." if speech
                      else f"I couldn't run {missed}.")
        return SkillResult(
            speech or f"The {self._routine.name} routine did nothing.",
            success=not failures and bool(spoken),
            data={"routine": self._routine.name,
                  "steps": len(self._routine.steps),
                  "failed": failures})

    async def _run_step(self, step: Step, request: SkillRequest,
                        context: dict[str, Any]) -> str | None:
        """Run one step. Returns its speech, or None when the step failed."""
        if step.skill is None:
            return render(step.say or "", context)
        target = self._registry.get(step.skill)
        if target is None:
            logger.warning("routine %s: no skill named %r",
                           self._routine.name, step.skill)
            return None
        args = render_args(step.args, context)
        # The step inherits the ORIGINATING request's context — its language,
        # its source — so a routine invoked from a routine is still unattended,
        # and a Macedonian "добро утро" produces Macedonian answers.
        sub = SkillRequest(text=render(step.say or request.text, context),
                           args=args, context=dict(request.context or {}))
        result = await target.execute(sub)
        if result.needs_confirmation:
            # Unreachable if the routine was validated (validate_steps refuses
            # confirming skills), and refused here anyway: a routine must never
            # be the thing that answers a safety prompt.
            logger.error("routine %s: step %r asked for confirmation; refusing",
                         self._routine.name, step.skill)
            return None
        return result.speech if result.success else None


def validate_steps(routine: Routine, registry) -> list[str]:
    """Problems with a routine's steps, checked against the LIVE registry.

    Separate from :func:`parse_routine` because it needs the registry, which
    does not exist when the YAML is read. Returns messages, not exceptions: the
    caller shows them and skips the routine.
    """
    problems: list[str] = []
    for index, step in enumerate(routine.steps, start=1):
        if step.skill is None:
            continue
        target = registry.get(step.skill)
        if target is None:
            problems.append(
                f"routine {routine.name!r} step {index}: there is no skill "
                f"called {step.skill!r}")
            continue
        if getattr(target, "requires_confirmation", False):
            problems.append(
                f"routine {routine.name!r} step {index}: {step.skill!r} needs a "
                f"spoken confirmation, so it can't be run from a routine — a "
                f"prompt in the middle of a chain has nobody to answer it")
    return problems


def phrase_conflicts(routine: Routine, registry) -> list[str]:
    """Trigger phrases an EARLIER skill already claims on the fast path.

    Registration order sets fast-path precedence, and routines register last —
    so a phrase another skill already matches never reaches the routine, and the
    user gets that other skill's answer instead. Silently. Both shipped examples
    hit this: "focus session" was taken by the window-focus skill, "добро утро"
    by the briefing. Not fatal (the routine is still reachable as an LLM tool
    and by its other phrases), so this warns rather than refuses — but it has to
    warn, because the failure looks like the routine simply not working.
    """
    clashes = []
    for phrase in routine.trigger.phrases:
        hit = registry.find_match(phrase)
        if hit is not None and hit[0].name != routine.skill_name:
            clashes.append(
                f"routine {routine.name!r}: the phrase {phrase!r} is already "
                f"claimed by the {hit[0].name!r} skill, so saying it will run "
                f"that instead — pick different wording")
    return clashes


def register_routines(registry,
                      routines: list[Routine]) -> tuple[int, list[str], list[str]]:
    """Register valid routines as skills.

    Returns ``(count, refused, warnings)``. ``refused`` blocks registration;
    ``warnings`` are things worth saying about routines that DID register.
    """
    registered, refused, warnings = 0, [], []
    for routine in routines:
        issues = validate_steps(routine, registry)
        if issues:
            refused.extend(issues)
            continue
        if registry.get(routine.skill_name) is not None:
            refused.append(
                f"routine {routine.name!r}: a skill called "
                f"{routine.skill_name!r} is already registered")
            continue
        # BEFORE registering: once the routine is in the registry it matches its
        # own phrases, and the conflict becomes invisible.
        warnings.extend(phrase_conflicts(routine, registry))
        registry.register(RoutineSkill(routine, registry))
        registered += 1
    return registered, refused, warnings


def _main(argv=None) -> int:  # pragma: no cover - operator utility
    """``python -m core.composer --check`` — load every routine and report."""
    import argparse

    from core.config import PROJECT_ROOT, load_settings
    from main import Announcer, build_registry

    parser = argparse.ArgumentParser(description="MEDO routines")
    parser.add_argument("--check", action="store_true",
                        help="validate every routine against the live skills")
    parser.add_argument("--dir", default=None, help="routines directory")
    args = parser.parse_args(argv)
    if not args.check:
        parser.print_help()
        return 2

    directory = Path(args.dir) if args.dir else PROJECT_ROOT / ROUTINES_DIRNAME
    routines, errors = load_routine_files(directory)
    registry = build_registry(load_settings(), Announcer())
    for error in errors:
        print(f"REFUSED  {error}")
    ok = 0
    for routine in routines:
        problems = validate_steps(routine, registry)
        if problems:
            errors.extend(problems)
            for problem in problems:
                print(f"REFUSED  {problem}")
            continue
        for clash in phrase_conflicts(routine, registry):
            print(f"WARNING  {clash}")
        ok += 1
        triggers = []
        if routine.trigger.phrases:
            triggers.append("phrase: " + ", ".join(routine.trigger.phrases))
        if routine.trigger.at:
            triggers.append(f"at {routine.trigger.at}")
        if routine.trigger.event:
            triggers.append(f"on {routine.trigger.event}")
        print(f"OK       {routine.skill_name}  ({len(routine.steps)} steps"
              + (f"; {'; '.join(triggers)}" if triggers else "") + ")")
    print(f"\n{ok} routine(s) ready, {len(errors)} refused.")
    return 1 if errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
