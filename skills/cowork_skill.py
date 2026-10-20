"""MEDO Cowork as an ordinary skill: hand it a task, hear a plan, approve it.

Being a :class:`~skills.base.Skill` is the whole integration. Cowork gets the
fast path, the LLM tool schema, the policy gate and the HUD for free, and the
router needs to know nothing about it — the same trick routines, connectors and
MEDO Link devices already use.

Two deliberate choices worth defending:

* **``requires_confirmation`` is False.** Not because Cowork is harmless, but
  because its confirmation is a *plan*, not a yes/no: the router's gate would
  ask "do you want me to cowork?" before MEDO had anything to show, and then a
  plain "yes" would authorise a plan the user has not heard. Approval happens
  inside, against a plan that has been said out loud, and nothing executes
  before it (:mod:`cowork.session`).
* **It declares no capabilities of its own.** A routine declares the union of
  its steps' capabilities because its steps are known when it loads; a Cowork
  plan is not. Instead every step is re-checked at the live policy gate at the
  moment it runs (:mod:`cowork.executor`), which is strictly stronger — it
  catches PC control being switched off *after* the plan was approved.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core import mk
from core.config import PROJECT_ROOT
from cowork.executor import Executor
from cowork.planner import Planner, language_of
from cowork.save import save_plan_as_routine
from cowork.session import CoworkSession
from cowork.speech import say
from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

#: Turn sources with nobody there to approve a plan. Mirrors the router's own
#: set — a plan is a conversation, and a timer cannot have one.
UNATTENDED_SOURCES = frozenset({"routine", "webhook", "trigger", "schedule",
                                "briefing"})

#: "dry run" — say the plan, do none of it.
_DRY = re.compile(r"\b(?:dry[\s-]?run|practice\s+run|rehears\w*|"
                  r"на\s+суво|проба)\b", re.IGNORECASE)


class CoworkSkill(Skill):
    """Bounded, visible orchestration of a multi-step task."""

    name = "cowork"
    description = (
        "Carry out a multi-step task the user hands over ('prep my demo: open "
        "OBS, start recording, then save the clip'). Makes a plan, gets it "
        "approved, runs it one step at a time and stops if a step fails. Use "
        "only when the user delegates a whole task with several parts.")

    #: Anchored and explicit. Cowork must never grab an ordinary one-shot
    #: command, and "do this for me" is deliberately absent: it belongs to the
    #: screen agent, which was here first.
    patterns = [
        re.compile(r"^\s*(?:cowork|co-?work)\b[,:]?\s*(?P<task>.+)",
                   re.IGNORECASE),
        re.compile(r"\b(?:plan\s+and\s+(?:do|run|execute)|"
                   r"make\s+a\s+plan\s+(?:to|for))\b"
                   r"(?:\s+(?:this|that|the\s+following))?[,:]?\s*(?P<task>.+)",
                   re.IGNORECASE),
        re.compile(r"\btake\s+care\s+of\s+(?:this|the\s+following)\b[,:]?"
                   r"\s*(?P<task>.+)", re.IGNORECASE),
        re.compile(r"^\s*(?:испланирај(?:\s+и\s+направи)?|направи\s+план)\b"
                   r"[,:]?\s*(?P<task>.+)", re.IGNORECASE),
        re.compile(r"^\s*(?:преземи\s+го\s+ова|среди\s+го\s+ова)\b[,:]?"
                   r"\s*(?P<task>.+)", re.IGNORECASE),
        # Saving the last plan. "routine"/"планот" is required so this can never
        # swallow "save this as a Word document".
        re.compile(r"\bsave\s+(?:this|that|the)\s+plan\s+as\s+(?:a\s+)?"
                   r"(?P<save>.+)", re.IGNORECASE),
        re.compile(r"\bsave\s+(?:this|that|it)\s+as\s+(?:a\s+)?(?P<save2>.+?)"
                   r"\s+routine\b", re.IGNORECASE),
        re.compile(r"\bзачувај\s+(?:го\s+)?(?:ова|планот)\s+како\s+"
                   r"(?P<save3>.+)", re.IGNORECASE),
    ]

    routing_phrases = [
        "prep my demo: open OBS, start recording and save the clip",
        "plan and do this for me",
        "take care of this: open the report and email it",
        "испланирај и направи го ова",
    ]

    def __init__(self, config: Any, registry: Any, *, think: Any = None,
                 policy: Any = None, announce: Any = None,
                 routines_dir: Any = None) -> None:
        self._config = config
        self._registry = registry
        self._think = think
        self._policy = policy
        self._announce = announce
        self._routines_dir = routines_dir or (PROJECT_ROOT / "routines")
        self._session: CoworkSession | None = None
        #: The last plan, kept only so "save that as demo prep" can work.
        self._last_plan = None

    # -- routing --------------------------------------------------------------

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string",
                                 "description": "The whole multi-step task, in "
                                                "the user's own words."},
                    },
                    "required": ["task"],
                },
            },
        }

    # -- the one entry point --------------------------------------------------

    async def execute(self, request: SkillRequest) -> SkillResult:
        ctx = request.context or {}
        language = self._language(request)
        if not getattr(self._config, "enabled", False):
            return SkillResult(say("disabled", language), success=False)

        # 1. Already in a conversation about a plan: this is the answer to it.
        if ctx.get("captured_reply") and self._session is not None \
                and self._session.active:
            reply = await self._session.reply(request.text, dict(ctx))
            self._last_plan = self._session.plan
            return self._result(reply)

        # 2. "save that as demo prep" — a finished plan becomes a routine.
        wanted = self._save_name(request)
        if wanted is not None:
            return self._save(wanted, language)

        # 3. A new task.
        if ctx.get("untrusted") or ctx.get("provenance") == "untrusted":
            # TRUST BOUNDARY. A task that arrived inside a document, a web page
            # or another program's call is content, and content never becomes a
            # plan — which is the same rule the software connectors keep, and
            # the reason `cowork` is on the MCP server's never-expose list.
            logger.info("cowork: refusing a task from untrusted content")
            return SkillResult(
                "Преземам задачи само кога ти ќе ми ги кажеш — не од документ "
                "или страница." if language == "mk" else
                "I only take on a task when you ask me directly — not one that "
                "came from a document, a page, or another program.",
                success=False)

        task = self._task_of(request)
        if not task:
            return SkillResult(
                "Што да преземам?" if language == "mk"
                else "What would you like me to take care of?", success=False)
        if str(ctx.get("source") or "") in UNATTENDED_SOURCES \
                and not getattr(self._config, "allow_unattended", False):
            # A plan needs someone to approve it. On a timer, nobody can — and
            # the next step of the same schedule must never be the thing that
            # answers for them.
            logger.info("cowork: refusing a task from %r", ctx.get("source"))
            return SkillResult(say("unattended", language), success=False)

        self._session = self._new_session(request)
        # Planning is one call to the strong brain over a catalogue of every
        # capability MEDO has — a minute or two on this machine. Say something
        # first: a silence that long is indistinguishable from being ignored.
        await self._announce_planning(language)
        reply = await self._session.start(task, dict(ctx), language=language)
        self._last_plan = self._session.plan
        return self._result(reply)

    # -- pieces ---------------------------------------------------------------

    def _new_session(self, request: SkillRequest) -> CoworkSession:
        planner = Planner(self._registry, self._think,
                          max_steps=int(getattr(self._config, "max_steps", 6)),
                          policy=self._policy)
        executor = Executor(self._registry, self._policy,
                            on_progress=self._progress)
        narrate = bool(getattr(self._config, "narrate", True))
        return CoworkSession(planner, executor,
                             announce=self._announce if narrate else None,
                             dry_run=self._is_dry_run(request))

    def _result(self, reply: Any) -> SkillResult:
        return SkillResult(
            reply.speech, success=reply.success,
            data=dict(reply.data or {}),
            await_reply=reply.await_reply,
            # OPEN, not an offer: the answer to "shall I go ahead?" is as likely
            # to be "skip step 3" or "change the file to X" as it is to be yes,
            # and an offer-capture would re-route those to whichever skill their
            # words happen to match. "stop" always gets out.
            reply_is_open=reply.await_reply)

    async def _announce_planning(self, language: str) -> None:
        if self._announce is None:
            return
        try:
            result = self._announce(say("planning", language))
            if hasattr(result, "__await__"):
                await result
        except Exception:  # noqa: BLE001 - a filler line must never block a task
            logger.debug("cowork: could not announce planning", exc_info=True)

    def _progress(self, plan: Any) -> None:
        """Per-step status, for the HUD and the log.

        The full plan (every step and its status) rides on the ROUTED event's
        ``data`` at every turn, which is the HUD's existing channel — no second
        render path, per the frozen HUD contract.
        """
        logger.info("cowork: %s | %s", plan.progress(),
                    ", ".join(f"{i}:{s.status}"
                              for i, s in enumerate(plan.steps, start=1)))

    def _save(self, name: str, language: str) -> SkillResult:
        plan = self._last_plan
        if plan is None or not plan.steps:
            return SkillResult(say("nothing_to_save", language), success=False)
        if not getattr(self._config, "save_routines", True):
            return SkillResult(say("save_refused", language,
                                   reason="saving routines is switched off"),
                               success=False)
        saved, error = save_plan_as_routine(plan, name, self._registry,
                                            self._routines_dir)
        if error and not saved:
            return SkillResult(say("save_refused", language, reason=error),
                               success=False)
        if error:
            return SkillResult(f"{say('saved', language, name=saved)} {error}")
        return SkillResult(say("saved", language, name=saved))

    # -- reading the request --------------------------------------------------

    def _task_of(self, request: SkillRequest) -> str:
        groups = request.match.groupdict() if request.match else {}
        task = (request.args.get("task") or groups.get("task") or "").strip()
        if task:
            return task.strip(" .!?,:")
        # Reached by MEANING (semantic tier) or by a tool call with no argument:
        # the whole utterance IS the task.
        return (request.utterance or request.text or "").strip()

    def _save_name(self, request: SkillRequest) -> str | None:
        groups = request.match.groupdict() if request.match else {}
        for key in ("save", "save2", "save3"):
            value = (groups.get(key) or "").strip()
            if value:
                return value
        return None

    def _is_dry_run(self, request: SkillRequest) -> bool:
        if bool(getattr(self._config, "dry_run", False)):
            return True
        return bool(_DRY.search(request.utterance or request.text or ""))

    @staticmethod
    def _language(request: SkillRequest) -> str:
        spoken = request.utterance or request.text or ""
        return "mk" if mk.is_cyrillic(spoken) else language_of(spoken)
