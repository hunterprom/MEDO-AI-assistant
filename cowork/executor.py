"""S3 — walk an approved plan, one step at a time, and STOP when one fails.

The executor is deliberately dumb about what to do next: the plan says. Its only
decisions are safety ones, and they are all re-taken here rather than trusted
from planning time:

* **The gate runs again, per step.** PC control may have been switched off, a
  device may have gone offline, ``security.enabled`` may have changed since the
  plan was approved. A step the policy engine now denies is a stop, not a skip.
* **High-impact steps ask again.** Approving a plan is not a blank cheque. The
  ordinary side effects (open an app, start a recording) are covered by the
  approval — that IS the batch — but power, run_command, write_files, device
  control, plugin installs and self-modification re-ask at the moment they run,
  naming what they are about to do.
* **A failure stops the task.** No auto-replan, no substitute action, no "let me
  try something else": those are how a bounded orchestrator turns into an
  unbounded one, and on a local brain they are also how it turns into a demo
  that does something nobody asked for. Report where it stopped, offer retry /
  skip / abort, and wait.

Undo is offered only where a skill genuinely provides it (an ``undo``
coroutine). Nothing in MEDO does today, so the honest answer to "undo that" is
that it can't — which is what the executor says, rather than pretending.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from cowork.plan import DONE, FAILED, PENDING, RUNNING, SKIPPED, Plan, Step
from cowork.speech import say

logger = logging.getLogger(__name__)

#: What ``run_step`` did. Not the same as a step's *status*: CONFIRM leaves the
#: step PENDING, because nothing has happened yet.
RAN = "done"
SKIPPED_OUT = "skipped"
CONFIRM = "confirm"
FAILURE = "failed"


@dataclass
class StepOutcome:
    """One step's result, as the session needs to hear it."""

    kind: str
    speech: str
    number: int
    step: Step
    reason: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def stops(self) -> bool:
        return self.kind == FAILURE


class Executor:
    """Run the steps of an approved plan against the live skill registry."""

    def __init__(self, registry: Any, policy: Any = None, *,
                 actor: Any = None, on_progress: Any = None) -> None:
        self._registry = registry
        self._policy = policy
        self._actor = actor
        #: Called with the plan after every status change — the HUD's feed.
        self._on_progress = on_progress

    # -- one step -------------------------------------------------------------

    def needs_reconfirmation(self, step: Step) -> bool:
        """Does this step ask again at execution time, inside an approved plan?

        The skill's own ``requires_confirmation`` and the high-impact capability
        set. Everything else was covered by approving the plan — asking about
        each of six steps trains the user to say yes without listening, which
        makes the prompts that matter worthless.
        """
        return bool(step.requires_confirmation or step.high_impact)

    async def run_step(self, plan: Plan, number: int, step: Step,
                       context: dict[str, Any] | None = None, *,
                       confirmed: bool = False) -> StepOutcome:
        lang = plan.language
        context = dict(context or {})

        if not step.possible:
            # Known at planning time and approved with that known — the user
            # heard "I'll skip the ones I can't do".
            step.status = SKIPPED
            step.error = step.problem
            await self._progress(plan)
            return StepOutcome(SKIPPED_OUT,
                               say("step_skipped", lang, number=number,
                                   reason=step.problem),
                               number, step, reason=step.problem)

        skill = self._registry.get(step.capability)
        if skill is None:
            # The registry changed under us (a device unregistered, a plugin
            # unloaded). That is a stop: the plan the user approved no longer
            # describes what MEDO can do.
            return self._fail(plan, number, step,
                              say("denied", lang, number=number,
                                  reason=f"{step.capability} is gone"),
                              reason=f"{step.capability} is no longer registered")

        denial = self._gate(skill)
        if denial:
            return self._fail(plan, number, step,
                              say("denied", lang, number=number, reason=denial),
                              reason=denial)

        if self.needs_reconfirmation(step) and not confirmed:
            prompt = self._confirmation_prompt(skill, step, plan, context)
            return StepOutcome(CONFIRM,
                               say("step_confirm", lang, number=number,
                                   prompt=prompt),
                               number, step)

        step.status = RUNNING
        await self._progress(plan)
        try:
            result = await skill.execute(
                self._request(plan, step, context, confirmed=confirmed))
        except Exception as exc:  # noqa: BLE001 - a failing step is data, not a crash
            logger.warning("cowork: step %d (%s) raised", number,
                           step.capability, exc_info=True)
            return self._fail(plan, number, step,
                              say("step_failed", lang, number=number,
                                  reason=self._reason(exc)),
                              reason=self._reason(exc))

        if getattr(result, "needs_confirmation", False):
            # A skill that gates itself per-argument (delete vs. read) lands
            # here rather than in needs_reconfirmation. Its own prompt is the
            # better question — it names the file.
            step.status = PENDING
            await self._progress(plan)
            return StepOutcome(CONFIRM,
                               say("step_confirm", lang, number=number,
                                   prompt=result.speech),
                               number, step)

        if not getattr(result, "success", False):
            return self._fail(plan, number, step,
                              say("step_failed", lang, number=number,
                                  reason=result.speech or "it didn't work"),
                              reason=result.speech or "it didn't work")

        step.status = DONE
        step.speech = (result.speech or "").strip()
        await self._progress(plan)
        # NOTE the step's speech is DATA. It is narrated and stored, and it is
        # never parsed for further instructions — the plan is locked, so there
        # is nothing for an instruction hidden in a file or a web page to do.
        spoken = (say("step_done", lang, number=number, speech=step.speech)
                  if step.speech
                  else say("step_done_quiet", lang, number=number))
        return StepOutcome(RAN, spoken, number, step,
                           data=dict(getattr(result, "data", {}) or {}))

    # -- the rest of the plan -------------------------------------------------

    def skip(self, plan: Plan, number: int, step: Step) -> StepOutcome:
        """Mark a step skipped after a failure the user chose to step over."""
        step.status = SKIPPED
        return StepOutcome(SKIPPED_OUT,
                           say("step_skipped", plan.language, number=number,
                               reason=step.error or "you said to skip it"),
                           number, step)

    def retry(self, step: Step) -> None:
        """Put a failed step back to PENDING so the walk picks it up again."""
        step.status = PENDING
        step.error = ""

    def undo_target(self, plan: Plan) -> tuple[int, Step] | None:
        """The last DONE step that can genuinely be undone, or None."""
        for number, step in reversed(list(enumerate(plan.steps, start=1))):
            if step.status != DONE:
                continue
            skill = self._registry.get(step.capability)
            if skill is not None and callable(getattr(skill, "undo", None)):
                return number, step
            return None      # the last thing done is not reversible: say so
        return None

    async def undo_last(self, plan: Plan,
                        context: dict[str, Any] | None = None) -> str:
        """Undo the last reversible step, or say plainly that it can't be."""
        target = self.undo_target(plan)
        if target is None:
            last = next((s for s in reversed(plan.steps) if s.status == DONE),
                        None)
            return say("undo_impossible", plan.language,
                       what=(last.description if last else "nothing ran"))
        number, step = target
        skill = self._registry.get(step.capability)
        try:
            await skill.undo(self._request(plan, step, dict(context or {})))
        except Exception:  # noqa: BLE001
            logger.warning("cowork: undo of step %d failed", number, exc_info=True)
            return say("undo_impossible", plan.language, what=step.description)
        step.status = SKIPPED
        await self._progress(plan)
        return say("undo_done", plan.language, number=number)

    # -- internals ------------------------------------------------------------

    def _request(self, plan: Plan, step: Step, context: dict[str, Any], *,
                 confirmed: bool = False):
        """The SkillRequest a step runs as.

        ``utterance`` is the user's ORIGINAL task, not the step description: it
        is what the user actually said, so a skill picking its reply language
        (or checking that the user really asked to lock the screen) reads the
        real words rather than a description the model wrote.
        """
        from skills.base import SkillRequest

        ctx = {**context, "via": "cowork", "utterance": plan.task}
        if confirmed:
            ctx["confirmed"] = True
        return SkillRequest(text=step.description, args=dict(step.args),
                            context=ctx)

    def _confirmation_prompt(self, skill: Any, step: Step, plan: Plan,
                             context: dict[str, Any]) -> str:
        try:
            prompt = skill.confirmation_prompt(
                self._request(plan, step, context))
        except Exception:  # noqa: BLE001 - a skill's own prompt is optional
            prompt = ""
        return prompt or say("generic_confirm", plan.language,
                             description=step.description)

    def _gate(self, skill: Any) -> str:
        """The policy engine's refusal for this skill right now, or ""."""
        if self._policy is None:
            return ""
        try:
            kwargs = {"actor": self._actor} if self._actor is not None else {}
            decision = self._policy.gate_skill(skill, **kwargs)
        except Exception:  # noqa: BLE001 - fail closed
            logger.warning("cowork: policy gate errored mid-task", exc_info=True)
            return "I couldn't check whether that's allowed"
        return decision.reason if decision.denied() else ""

    def _fail(self, plan: Plan, number: int, step: Step, speech: str,
              reason: str) -> StepOutcome:
        step.status = FAILED
        step.error = reason
        plan.stopped = True
        logger.info("cowork: stopped at step %d (%s): %s", number,
                    step.capability, reason)
        return StepOutcome(FAILURE, speech, number, step, reason=reason)

    @staticmethod
    def _reason(exc: BaseException) -> str:
        text = str(exc).strip()
        return text or exc.__class__.__name__

    async def _progress(self, plan: Plan) -> None:
        if self._on_progress is None:
            return
        try:
            result = self._on_progress(plan)
            if hasattr(result, "__await__"):
                await result
        except Exception:  # noqa: BLE001 - the HUD must never break the task
            logger.debug("cowork: progress callback failed", exc_info=True)
