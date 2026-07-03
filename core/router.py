"""Intent Router — the heart of MEDO.

For each utterance it decides between two paths and records which one handled it:

* **FAST**  — a registered skill's regex matched; execute deterministically, no LLM.
* **LLM**   — no rule matched; hand the text to Ollama to converse (tool calling in M3).

Routing stats are tallied for the README.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.config import Settings
from core.events import Event, EventBus, EventType, RoutePath
from core.memory import ConversationMemory
from core.safety import is_affirmative, is_negative
from llm.client import LLMUnavailableError, OllamaClient
from llm.prompts import system_prompt
from llm.tools import build_tools, dispatch_tool
from skills.base import Skill, SkillRegistry, SkillRequest

#: How many tool rounds before we force a final text answer (loop guard).
MAX_TOOL_ROUNDS = 4
#: Tools whose output is raw and needs the model to synthesize a spoken reply.
#: Everything else returns a ready-to-speak sentence, so we skip the second LLM
#: hop — faster, and it can't be undone by a small model second-guessing itself.
SYNTHESIS_TOOLS = {"web_search"}

logger = logging.getLogger(__name__)

OFFLINE_LLM_REPLY = (
    "I can't reach my language model right now. Make sure Ollama is running and a "
    "model is installed."
)
OFFLINE_CLOUD_REPLY = (
    "I can't reach the cloud model. Check your API key, the base URL, and your "
    "internet connection — or switch back to the local provider."
)
CANCELLED_REPLY = "Okay, cancelled."


def _looks_like_tool_json(text: str) -> bool:
    """Detect a small model leaking a function-call blob into its text reply."""
    stripped = text.lstrip()
    return stripped.startswith("{") and any(
        key in text for key in ('"name"', '"parameters"', '"arguments"', '"action"')
    )


@dataclass
class RouteResult:
    """Outcome of routing one utterance."""

    path: RoutePath
    speech: str
    skill_name: str | None = None
    latency_ms: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)


class Router:
    """Routes utterances to the fast path or the LLM path."""

    def __init__(
        self,
        settings: Settings,
        registry: SkillRegistry,
        llm: OllamaClient,
        bus: EventBus,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._llm = llm
        self._bus = bus
        #: Active model for the LLM path; set by the app / model picker.
        self.model: str | None = settings.llm.default_model
        #: Routing tallies for the README.
        self.stats: dict[RoutePath, int] = {RoutePath.FAST: 0, RoutePath.LLM: 0}
        #: A destructive action awaiting a spoken yes/no, if any.
        self._pending: tuple[Skill, SkillRequest] | None = None
        #: Rolling context so follow-ups ("and tomorrow?") resolve.
        self.conversation = ConversationMemory(settings.memory.max_turns)

    @property
    def llm(self) -> OllamaClient:
        """The LLM client (shared with the companion API for model discovery)."""
        return self._llm

    @property
    def awaiting_confirmation(self) -> bool:
        """True when the last reply asked for confirmation (UI should re-listen)."""
        return self._pending is not None

    async def route(self, text: str, context: dict[str, Any] | None = None) -> RouteResult:
        text = text.strip()
        started = time.perf_counter()

        # Announce the utterance so any UI (HUD) can show it, whatever the source.
        await self._bus.emit(Event(EventType.TRANSCRIPT, text))
        result = await self._route_inner(text, context or {})
        result.latency_ms = (time.perf_counter() - started) * 1000.0

        self.stats[result.path] += 1
        # Remember the turn (unless we're mid-confirmation, where the follow-up is
        # a yes/no that shouldn't pollute conversational context).
        if not self.awaiting_confirmation:
            self.conversation.add_turn(text, result.speech)
        logger.info(
            "[%s] %s (%.0f ms)%s",
            result.path.value,
            text,
            result.latency_ms,
            f" -> {result.skill_name}" if result.skill_name else "",
        )
        await self._bus.emit(Event(EventType.ROUTED, result))
        return result

    async def _route_inner(self, text: str, context: dict[str, Any]) -> RouteResult:
        # --- CONFIRMATION GATE ---
        # A destructive action is waiting on a yes/no; this reply answers it.
        if self._pending is not None:
            return await self._resolve_confirmation(text, context)

        # --- FAST PATH ---
        if self._settings.router.fast_path_enabled:
            match = self._registry.find_match(text)
            if match is not None:
                skill, regex_match = match
                request = SkillRequest(text=text, match=regex_match, context=context)
                return await self._run_skill(skill, request)

        # --- LLM PATH ---
        return await self._llm_reply(text, context)

    async def _run_skill(self, skill: Skill, request: SkillRequest) -> RouteResult:
        """Execute a fast-path skill, deferring for confirmation if it asks."""
        outcome = await skill.execute(request)
        if outcome.needs_confirmation and self._settings.safety.confirm_destructive:
            # Stash the request; the next utterance is treated as the yes/no.
            self._pending = (skill, request)
        return RouteResult(
            path=RoutePath.FAST,
            speech=outcome.speech,
            skill_name=skill.name,
            data=outcome.data,
        )

    async def _resolve_confirmation(
        self, text: str, context: dict[str, Any]
    ) -> RouteResult:
        skill, request = self._pending  # type: ignore[misc]
        if is_affirmative(text):
            self._pending = None
            request.context = {**request.context, "confirmed": True}
            outcome = await skill.execute(request)  # now performs the action
            return RouteResult(
                path=RoutePath.FAST,
                speech=outcome.speech,
                skill_name=skill.name,
                data=outcome.data,
            )
        if is_negative(text):
            self._pending = None
            return RouteResult(path=RoutePath.FAST, speech=CANCELLED_REPLY)
        # Anything else: don't guess with a destructive action — cancel and re-route.
        self._pending = None
        logger.info("ambiguous confirmation %r; cancelling pending action", text)
        return await self._route_inner(text, context)

    @property
    def _offline_reply(self) -> str:
        """The provider-appropriate 'can't reach the model' message."""
        if self._settings.llm.provider == "openai":
            return OFFLINE_CLOUD_REPLY
        return OFFLINE_LLM_REPLY

    async def _llm_reply(self, text: str, context: dict[str, Any]) -> RouteResult:
        if not self.model:
            return RouteResult(path=RoutePath.LLM, speech=self._offline_reply)

        tools = build_tools(self._registry)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt(self._settings.personality)},
            *self.conversation.recent_messages(),   # rolling context for follow-ups
            {"role": "user", "content": text},
        ]

        try:
            for _ in range(MAX_TOOL_ROUNDS):
                message = await self._llm.chat(self.model, messages, tools=tools)
                tool_calls = message.get("tool_calls") or []
                if not tool_calls:
                    reply = (message.get("content") or "").strip()
                    if reply and _looks_like_tool_json(reply):
                        # Small model leaked a botched tool call as text; a plain
                        # retry (no tools) gets a clean spoken answer.
                        retry = await self._llm.chat(self.model, messages)
                        reply = (retry.get("content") or "").strip()
                    return RouteResult(path=RoutePath.LLM, speech=reply or self._offline_reply)

                messages.append(message)  # the assistant turn that requested tools
                pending = await self._run_tool_calls(tool_calls, text, context, messages)
                if pending is not None:
                    return pending  # a destructive tool needs confirmation first

            # Ran out of rounds: ask once more for a plain answer.
            message = await self._llm.chat(self.model, messages)
            return RouteResult(
                path=RoutePath.LLM, speech=(message.get("content") or self._offline_reply).strip()
            )
        except LLMUnavailableError as exc:
            logger.warning("LLM path unavailable: %s", exc)
            return RouteResult(path=RoutePath.LLM, speech=OFFLINE_LLM_REPLY)

    async def _run_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        text: str,
        context: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> RouteResult | None:
        """Execute each tool call, appending results.

        Returns a RouteResult to end the turn — either because a tool needs
        confirmation, or because the tools produced ready-to-speak answers (we
        skip a second, error-prone synthesis hop). Returns None to keep looping
        only when a synthesis tool (web search) needs the model to summarize.
        """
        direct: list[str] = []
        needs_synthesis = False
        used_skill: str | None = None

        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments") or {}
            result = await dispatch_tool(self._registry, name, args, context)
            used_skill = name

            if result.needs_confirmation and self._settings.safety.confirm_destructive:
                skill = self._registry.get(name)
                if skill is not None:
                    request = SkillRequest(text=text, args=args, context=context)
                    self._pending = (skill, request)
                return RouteResult(path=RoutePath.LLM, speech=result.speech, skill_name=name)

            messages.append({"role": "tool", "name": name, "content": result.speech})
            if name in SYNTHESIS_TOOLS:
                needs_synthesis = True
            elif result.success and result.speech:
                direct.append(result.speech)

        # Structured tools already speak for themselves — answer directly.
        if not needs_synthesis and direct:
            return RouteResult(path=RoutePath.LLM, speech=" ".join(direct), skill_name=used_skill)
        return None
