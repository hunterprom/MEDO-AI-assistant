"""The plan: an ordered list of steps, each bound to a capability MEDO HAS.

A plan is data, not behaviour. It is built by :mod:`cowork.planner`, said out
loud, edited by the user, approved, and then walked by :mod:`cowork.executor` —
which only ever writes a step's *status*, never its shape.

That split is the trust boundary made structural. Cowork's whole risk is that
something encountered mid-task — a file's contents, a web page, a device's
answer — talks MEDO into an action the user never approved. So the plan
**locks** at approval (:meth:`Plan.lock`): every method that could add, remove or
re-aim a step raises :class:`PlanLocked` from then on. A step's result can be
read, stored, spoken; it cannot become a step. New destructive actions need a
new plan and a new approval, and that is enforced here rather than promised in a
prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: Step lifecycle. A step is PENDING until the executor reaches it; a task that
#: STOPPED leaves the rest PENDING on purpose — "not done" is the honest state,
#: not "cancelled".
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"

#: Languages Cowork speaks. MEDO is bilingual; a plan read out in English for a
#: Macedonian request is the failure mode four English-only review rounds missed.
LANGUAGES = ("en", "mk")


class PlanLocked(RuntimeError):
    """An attempt to change an APPROVED plan. Always a bug, never a prompt."""


@dataclass
class Step:
    """One thing the plan does.

    ``capability`` is the registered skill/tool name — resolved against the live
    registry, never invented. An empty ``capability`` with a ``problem`` is a
    step MEDO genuinely cannot do: it stays in the plan and is SAID, because
    silently dropping it is how a plan comes to promise something it will not
    deliver.
    """

    description: str
    capability: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    #: Does it change something in the world? Read from the skill's declared
    #: capabilities (security.capabilities.SIDE_EFFECT_CAPS), not from the model.
    side_effect: bool = False
    #: The skill declares it needs a spoken yes before it runs.
    requires_confirmation: bool = False
    #: Power, run_command, write_files, device control, plugins, self-modification
    #: — the ones that reconfirm at execution time even inside an approved plan.
    high_impact: bool = False
    #: Why this step can't be done (set together with an empty ``capability``).
    problem: str = ""
    #: Filled in as it runs.
    status: str = PENDING
    speech: str = ""
    error: str = ""

    @property
    def possible(self) -> bool:
        return bool(self.capability) and not self.problem

    @property
    def finished(self) -> bool:
        return self.status in (DONE, FAILED, SKIPPED)

    def label(self) -> str:
        """The one-line form said aloud and shown in the HUD."""
        if not self.possible:
            return f"{self.description} — {self.problem}"
        return self.description


@dataclass
class Plan:
    """A task decomposed into steps, plus everything the user must hear first."""

    task: str
    steps: list[Step] = field(default_factory=list)
    #: The ONE clarifying question the planner needs answered before this plan
    #: is final. Deliberately one: a spoken interrogation is worse than a wrong
    #: default the user can correct.
    question: str = ""
    #: Set when there is no plan to offer at all (too big, nothing understood).
    #: A refusal is a finished answer, not a step.
    refusal: str = ""
    language: str = "en"
    approved: bool = False
    #: Set by :meth:`lock`; every mutator checks it.
    locked: bool = False
    #: True once the executor has stopped on a failure — the plan is not
    #: finished and must not be reported as if it were.
    stopped: bool = False

    # -- shape (refused once approved) ---------------------------------------

    def _mutable(self) -> None:
        if self.locked:
            raise PlanLocked(
                "this plan was approved; a new action needs a new plan")

    def add_step(self, step: Step, at: int | None = None) -> None:
        """Add a step. Raises once approved — see the module docstring.

        This is the method a prompt-injected "and now delete the folder" would
        have to reach. It cannot.
        """
        self._mutable()
        if at is None:
            self.steps.append(step)
        else:
            self.steps.insert(max(0, at), step)

    def remove_step(self, number: int) -> Step | None:
        """Drop step ``number`` (1-based, as spoken). None if there is no such step."""
        self._mutable()
        if not 1 <= number <= len(self.steps):
            return None
        return self.steps.pop(number - 1)

    def set_arg(self, number: int, key: str, value: Any) -> bool:
        """Re-aim one step's argument ("change the file to X"). False if absent."""
        self._mutable()
        if not 1 <= number <= len(self.steps):
            return False
        self.steps[number - 1].args[key] = value
        return True

    def lock(self) -> None:
        """Freeze the plan's shape. Called at approval, never undone."""
        self.locked = True
        self.approved = True

    # -- reading --------------------------------------------------------------

    @property
    def possible_steps(self) -> list[Step]:
        return [s for s in self.steps if s.possible]

    @property
    def impossible_steps(self) -> list[Step]:
        return [s for s in self.steps if not s.possible]

    @property
    def side_effect_steps(self) -> list[Step]:
        return [s for s in self.steps if s.possible and s.side_effect]

    def next_pending(self) -> tuple[int, Step] | None:
        """The next step to run, 1-based, or None when the plan is walked."""
        for index, step in enumerate(self.steps, start=1):
            if step.status == PENDING:
                return index, step
        return None

    @property
    def finished(self) -> bool:
        return all(s.finished for s in self.steps) if self.steps else True

    def counts(self) -> dict[str, int]:
        out = {PENDING: 0, RUNNING: 0, DONE: 0, FAILED: 0, SKIPPED: 0}
        for step in self.steps:
            out[step.status] = out.get(step.status, 0) + 1
        return out

    # -- speech ---------------------------------------------------------------

    def summary(self) -> str:
        """The plan, said out loud, ending in the approval question.

        Numbered because the user edits it by number ("skip step 3"), and short
        because it is SPOKEN: a plan you cannot hold in your head is not a
        visible plan.
        """
        if self.refusal:
            return self.refusal
        speak_mk = self.language == "mk"
        if not self.steps:
            return ("Немам чекори за таа задача." if speak_mk
                    else "I don't have any steps for that task.")
        blocked = self.impossible_steps
        if len(blocked) == len(self.steps):
            # Nothing in the plan is doable. Asking "shall I go ahead?" about
            # that would be theatre.
            return (("Не можам да го направам тоа: " if speak_mk
                     else "I can't do that: ")
                    + "; ".join(s.problem for s in blocked) + ".")
        body = " ".join(f"{i}. {s.label()}"
                        for i, s in enumerate(self.steps, start=1))
        head = "Еве го планот:" if speak_mk else "Here's my plan:"
        tail = "Да продолжам?" if speak_mk else "Shall I go ahead?"
        note = ""
        if blocked:
            note = (" Ќе го прескокнам тоа што не можам." if speak_mk
                    else " I'll skip the ones I can't do.")
        return f"{head} {body}.{note} {tail}"

    def progress(self) -> str:
        """"Two of five done" — for the HUD and for a spoken status check."""
        done = self.counts()[DONE]
        total = len(self.steps)
        if self.language == "mk":
            return f"{done} од {total} завршени"
        return f"{done} of {total} done"

    def as_dict(self) -> dict[str, Any]:
        """HUD/event payload. Plain types only — this crosses the SSE boundary."""
        return {
            "task": self.task,
            "language": self.language,
            "approved": self.approved,
            "stopped": self.stopped,
            "steps": [
                {"n": i, "description": s.description, "capability": s.capability,
                 "args": dict(s.args), "status": s.status,
                 "side_effect": s.side_effect,
                 "requires_confirmation": s.requires_confirmation,
                 "high_impact": s.high_impact,
                 "problem": s.problem, "speech": s.speech, "error": s.error}
                for i, s in enumerate(self.steps, start=1)
            ],
        }


#: "skip step 3" / "прескокни го чекор 3" — the two edits worth understanding
#: without a model, because they are what people actually say to a read-out list.
_SKIP = re.compile(
    r"\b(?:skip|drop|remove|forget|прескокни|избриши|тргни)\b[^\d]{0,20}(\d+)",
    re.IGNORECASE)
#: "change step 2 city to Ohrid" / "set step 1 file to report.md"
_SET = re.compile(
    r"\b(?:change|set|make|промени|стави)\b[^\d]{0,20}(\d+)\W{0,4}"
    r"(?:'s\s+)?(?P<key>[a-zA-Z_]+)?\s*\b(?:to|на)\s+(?P<value>.+)",
    re.IGNORECASE)


def parse_edit(text: str) -> tuple[str, dict[str, Any]] | None:
    """Understand a spoken plan edit, or None.

    Only the two shapes above, and only with an explicit step NUMBER. An edit
    MEDO half-understood would change a plan the user then approves believing
    something else — worse than not understanding it at all, which merely asks
    them to say it again.
    """
    if not text:
        return None
    skip = _SKIP.search(text)
    if skip:
        return "skip", {"number": int(skip.group(1))}
    setter = _SET.search(text)
    if setter:
        value = setter.group("value").strip(" .!?")
        key = (setter.group("key") or "").strip().lower()
        if value:
            return "set", {"number": int(setter.group(1)), "key": key,
                           "value": value}
    return None
