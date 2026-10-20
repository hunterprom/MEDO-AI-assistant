"""S2 — the conversation around a plan: approve it, edit it, answer for it, stop it.

One delegated task is a dialogue, not a call:

    "prep my demo"      -> here's my plan: 1… 2… 3… — shall I go ahead?
    "skip step 3"       -> dropped step 3. Here's my plan: … — shall I go ahead?
    "yes"               -> right, starting.  (the plan LOCKS here)
    …step 2 is high impact -> Step 2: I'll start recording. Shall I?
    "yes"               -> Step 2 done: recording started.
    …step 3 fails       -> Step 3 failed: the dog didn't answer. Retry, skip, or stop?
    "stop"              -> Stopped. 2 of 4 done.

The session owns that state machine; :mod:`cowork.executor` owns what one step
does. Splitting them is what makes both testable without a router, a microphone
or a model.

Two rules the states exist to enforce:

* **Nothing runs before approval.** There is no path from :meth:`start` to a
  skill's ``execute`` — approval is the only transition into RUNNING, and it is
  also the moment the plan is locked against any later change.
* **"Stop" is heard everywhere.** Whatever MEDO is waiting for, an abort word
  ends the task cleanly and says what did and didn't happen.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from core.safety import is_affirmative, is_negative
from cowork.executor import CONFIRM, FAILURE, RAN, SKIPPED_OUT, Executor
from cowork.plan import DONE, FAILED, PENDING, SKIPPED, Plan, parse_edit
from cowork.speech import say

logger = logging.getLogger(__name__)

#: Session states.
IDLE = "idle"
AWAITING_ANSWER = "awaiting_answer"        # the planner's one question
AWAITING_APPROVAL = "awaiting_approval"    # the plan is on the table
AWAITING_STEP = "awaiting_step"            # a step is asking to run
AWAITING_CHOICE = "awaiting_choice"        # a step failed: retry / skip / stop

#: "stop" in both languages, plus the words people actually use to mean it.
#: Checked in EVERY waiting state — an abort that only works at one point in the
#: dialogue is not an abort.
_ABORT = re.compile(
    r"^\s*(?:stop|abort|cancel|quit|enough|never\s+mind|forget\s+it|"
    r"запри|стоп|стопирај|откажи|доста|прекини|заборави)\b",
    re.IGNORECASE)
_RETRY = re.compile(
    r"\b(?:retry|again|try\s+again|once\s+more|повтори|пак|обиди)\b",
    re.IGNORECASE)
_SKIP = re.compile(
    r"\b(?:skip|next|move\s+on|прескокни|следно|следен|продолжи)\b",
    re.IGNORECASE)


@dataclass
class Reply:
    """What the session hands back to the skill wrapping it."""

    speech: str
    await_reply: bool = False
    success: bool = True
    data: dict[str, Any] = field(default_factory=dict)


class CoworkSession:
    """One delegated task, from plan to finish.

    ``announce`` is an optional async ``(text) -> None`` — the same Announcer
    timers and builds use. With it, each step is narrated the moment it happens
    and the returned :class:`Reply` carries only what MEDO is waiting on; without
    it (``--once``, tests) the narration is accumulated into the reply, so a
    non-voice caller still sees every step.
    """

    def __init__(self, planner: Any, executor: Executor, *,
                 announce: Any = None, dry_run: bool = False) -> None:
        self._planner = planner
        self._executor = executor
        self._announce = announce
        self._dry_run = dry_run
        self.state: str = IDLE
        self.plan: Plan | None = None
        self._context: dict[str, Any] = {}
        self._pending_step: tuple[int, Any] | None = None
        self._narration: list[str] = []

    @property
    def active(self) -> bool:
        return self.state != IDLE

    # -- starting -------------------------------------------------------------

    async def start(self, task: str, context: dict[str, Any] | None = None,
                    language: str = "en") -> Reply:
        """Plan the task and put the plan to the user. Executes NOTHING."""
        self._context = dict(context or {})
        self._narration = []
        self._pending_step = None
        plan = await self._planner.plan(task, language=language)
        self.plan = plan
        if plan.refusal:
            self.state = IDLE
            return Reply(plan.refusal, success=False)
        if self._dry_run:
            # A rehearsal: say the plan, change nothing, wait for nothing.
            self.state = IDLE
            return Reply(say("dry_run", plan.language, summary=plan.summary()),
                         data=plan.as_dict())
        if plan.question:
            self.state = AWAITING_ANSWER
            return Reply(plan.question, await_reply=True, data=plan.as_dict())
        self.state = AWAITING_APPROVAL
        return Reply(plan.summary(), await_reply=True, data=plan.as_dict())

    # -- the one entry point for every following utterance --------------------

    async def reply(self, text: str,
                    context: dict[str, Any] | None = None) -> Reply:
        """Handle the captured next utterance, whatever the session is waiting on."""
        if context:
            self._context.update(context)
        plan = self.plan
        if plan is None or self.state == IDLE:
            return Reply(say("nothing_running", self._language()), success=False)

        if _ABORT.match(text or "") and self.state != AWAITING_APPROVAL:
            # Before approval an abort is simply "no" — handled below, where it
            # says "dropped it" rather than reporting progress that never began.
            return self._abort()

        if self.state == AWAITING_ANSWER:
            return await self._answered(text)
        if self.state == AWAITING_APPROVAL:
            return await self._approval(text)
        if self.state == AWAITING_STEP:
            return await self._step_answer(text)
        if self.state == AWAITING_CHOICE:
            return await self._choice(text)
        return Reply(say("nothing_running", self._language()), success=False)

    # -- states ---------------------------------------------------------------

    async def _answered(self, text: str) -> Reply:
        """The planner's clarifying question got an answer: re-plan with it."""
        plan = self.plan
        assert plan is not None
        if is_negative(text):
            self.state = IDLE
            return Reply(say("cancelled", plan.language))
        task = f"{plan.task}. {text.strip()}".strip()
        return await self.start(task, self._context, language=plan.language)

    async def _approval(self, text: str) -> Reply:
        plan = self.plan
        assert plan is not None
        lang = plan.language

        edit = parse_edit(text)
        if edit is not None:
            return self._edit(edit)
        if is_affirmative(text):
            plan.lock()                      # the shape is final from here
            self.state = "running"
            first = say("approved", lang)
            await self._say(first)
            return await self._walk(prefix=[] if self._announce else [first])
        if is_negative(text):
            self.state = IDLE
            return Reply(say("cancelled", lang))
        # Not yes, not no, not an edit. Do NOT guess with a plan that is about
        # to act on the machine.
        return Reply(say("not_understood", lang), await_reply=True)

    def _edit(self, edit: tuple[str, dict[str, Any]]) -> Reply:
        plan = self.plan
        assert plan is not None
        kind, payload = edit
        number = int(payload.get("number", 0))
        if kind == "skip":
            if plan.remove_step(number) is None:
                return Reply(say("no_such_step", plan.language, number=number),
                             await_reply=True)
            if not plan.steps:
                self.state = IDLE
                return Reply(say("cancelled", plan.language))
            return Reply(say("step_removed", plan.language, number=number,
                             summary=plan.summary()), await_reply=True,
                         data=plan.as_dict())
        key = str(payload.get("key") or "").strip()
        if not key and 1 <= number <= len(plan.steps):
            # "change step 2 to Ohrid" with no argument named: only unambiguous
            # when the step takes exactly one argument.
            args = plan.steps[number - 1].args
            keys = list(args) or self._param_names(plan.steps[number - 1])
            key = keys[0] if len(keys) == 1 else ""
        if not key:
            return Reply(say("not_understood", plan.language), await_reply=True)
        if not plan.set_arg(number, key, payload.get("value")):
            return Reply(say("no_such_step", plan.language, number=number),
                         await_reply=True)
        return Reply(say("step_changed", plan.language, number=number,
                         summary=plan.summary()), await_reply=True,
                     data=plan.as_dict())

    async def _step_answer(self, text: str) -> Reply:
        """Yes/no to a step that asked before running."""
        plan = self.plan
        assert plan is not None and self._pending_step is not None
        number, step = self._pending_step
        if is_affirmative(text):
            self._pending_step = None
            return await self._walk(confirm_number=number)
        if is_negative(text):
            # Declining ONE step stops the task. The steps after it were planned
            # on the assumption this one happened; carrying on would be MEDO
            # deciding for itself that the rest still makes sense, which is the
            # improvising this whole design refuses to do.
            step.status = SKIPPED
            plan.stopped = True
            self.state = IDLE
            self._pending_step = None
            return Reply(say("stopped", plan.language, number=number),
                         data=plan.as_dict())
        return Reply(say("step_not_understood", plan.language, number=number),
                     await_reply=True)

    async def _choice(self, text: str) -> Reply:
        """retry / skip / stop, after a step failed."""
        plan = self.plan
        assert plan is not None and self._pending_step is not None
        number, step = self._pending_step
        if _RETRY.search(text or ""):
            self._executor.retry(step)
            self._pending_step = None
            plan.stopped = False
            return await self._walk()
        if _SKIP.search(text or ""):
            outcome = self._executor.skip(plan, number, step)
            self._pending_step = None
            plan.stopped = False
            await self._say(outcome.speech)
            return await self._walk(prefix=[] if self._announce
                                    else [outcome.speech])
        if is_negative(text):
            return self._abort()
        return Reply(say("options", plan.language), await_reply=True)

    # -- the walk -------------------------------------------------------------

    async def _walk(self, *, confirm_number: int | None = None,
                    prefix: list[str] | None = None) -> Reply:
        """Run steps until one asks, one fails, or the plan is done."""
        plan = self.plan
        assert plan is not None
        lines: list[str] = list(prefix or [])
        while True:
            nxt = plan.next_pending()
            if nxt is None:
                self.state = IDLE
                lines.append(self._final_report(plan))
                return Reply(" ".join(x for x in lines if x).strip(),
                             data=plan.as_dict())
            number, step = nxt
            confirmed = confirm_number == number
            confirm_number = None
            outcome = await self._executor.run_step(
                plan, number, step, self._context, confirmed=confirmed)

            if outcome.kind == CONFIRM:
                self.state = AWAITING_STEP
                self._pending_step = (number, step)
                lines.append(outcome.speech)
                return Reply(" ".join(x for x in lines if x).strip(),
                             await_reply=True, data=plan.as_dict())
            if outcome.kind == FAILURE:
                self.state = AWAITING_CHOICE
                self._pending_step = (number, step)
                lines.append(outcome.speech)
                return Reply(" ".join(x for x in lines if x).strip(),
                             await_reply=True, success=False,
                             data=plan.as_dict())
            # RAN or SKIPPED_OUT: narrate and carry on.
            if outcome.kind in (RAN, SKIPPED_OUT):
                await self._say(outcome.speech)
                if not self._announce:
                    lines.append(outcome.speech)

    def _final_report(self, plan: Plan) -> str:
        counts = plan.counts()
        gaps = [f"{i}" for i, s in enumerate(plan.steps, start=1)
                if s.status in (SKIPPED, FAILED)]
        if not gaps:
            return say("finished", plan.language, progress=plan.progress())
        listed = ", ".join(gaps)
        one = len(gaps) == 1
        if plan.language == "mk":
            gap_text = (f"Не се направи чекорот {listed}." if one
                        else f"Не се направија чекорите {listed}.")
        else:
            gap_text = (f"Step {listed} didn't happen." if one
                        else f"Steps {listed} didn't happen.")
        logger.info("cowork: finished with gaps %s (%s)", listed, counts)
        return say("finished_with_gaps", plan.language,
                   progress=plan.progress(), gaps=gap_text)

    def _abort(self) -> Reply:
        plan = self.plan
        assert plan is not None
        plan.stopped = True
        self.state = IDLE
        self._pending_step = None
        return Reply(say("aborted", plan.language, progress=plan.progress()),
                     data=plan.as_dict())

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _param_names(step: Any) -> list[str]:
        return list(step.args)

    def _language(self) -> str:
        return self.plan.language if self.plan is not None else "en"

    async def _say(self, text: str) -> None:
        """Narrate a step as it happens, when there is somewhere to say it."""
        if not text or self._announce is None:
            return
        try:
            result = self._announce(text)
            if hasattr(result, "__await__"):
                await result
        except Exception:  # noqa: BLE001 - narration must not stop the task
            logger.debug("cowork: announce failed", exc_info=True)


__all__ = ["CoworkSession", "Reply", "AWAITING_ANSWER", "AWAITING_APPROVAL",
           "AWAITING_STEP", "AWAITING_CHOICE", "IDLE", "DONE", "PENDING"]
