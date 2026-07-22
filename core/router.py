"""Intent Router — the heart of MEDO.

For each utterance it decides between two paths and records which one handled it:

* **FAST**  — a registered skill's regex matched; execute deterministically, no LLM.
* **LLM**   — no rule matched; hand the text to Ollama to converse (tool calling in M3).

Routing stats are tallied for the README.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.config import Settings
from core.events import Event, EventBus, EventType, RoutePath
from core.facts import FactsStore
from core.memory import ConversationMemory
from core.metrics import MetricsStore
from core.safety import is_affirmative, is_negative
from llm.client import LLMUnavailableError, OllamaClient
from llm.prompts import system_prompt
from llm.tools import build_tools, dispatch_tool
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult

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
OFFLINE_CLI_REPLY = (
    "I can't reach the {name} command-line agent. Make sure it is installed, "
    "logged in, and on your PATH — or switch provider in the HUD."
)
CANCELLED_REPLY = "Okay, cancelled."
PC_CONTROL_OFF_REPLY = (
    "PC control is switched off — flip the PC CONTROL switch in the HUD "
    "settings if you want me to do that."
)
#: Spoken when the model returns nothing usable (e.g. reasoning truncated mid-think).
EMPTY_REPLY = "Sorry — I lost my train of thought. Ask me that again?"


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
        #: Routing tallies (this session) + persisted per-request metrics that
        #: feed the README's Performance table (python -m core.metrics --report).
        self.stats: dict[RoutePath, int] = {RoutePath.FAST: 0, RoutePath.LLM: 0}
        self.metrics: MetricsStore | None = (
            MetricsStore(settings.memory.db_path)
            if settings.logging.routing_stats else None
        )
        #: A destructive action awaiting a spoken yes/no, if any.
        self._pending: tuple[Skill, SkillRequest] | None = None
        #: Rolling context so follow-ups ("and tomorrow?") resolve.
        self.conversation = ConversationMemory(settings.memory.max_turns)
        #: Long-term user facts, injected into the system prompt each LLM turn.
        #: With an embed model configured, facts are ranked semantically against
        #: the utterance (local Ollama embeddings); otherwise newest-N.
        embedder = None
        if settings.memory.embed_model:
            from core.embeddings import embed_texts

            embed_model, embed_host = settings.memory.embed_model, settings.llm.host

            def embedder(texts):  # noqa: E731 - tiny closure over config
                return embed_texts(texts, embed_model, embed_host)

        self.facts = FactsStore(settings.memory.db_path, embedder)
        #: Personality layer (M7): occasionally decorates fast-path replies.
        #: Tests may replace it (or inject a seeded rng) for determinism.
        from core.persona import Persona

        self._persona = Persona(settings.personality)

    @property
    def llm(self) -> OllamaClient:
        """The LLM client (shared with the companion API for model discovery)."""
        return self._llm

    @property
    def awaiting_confirmation(self) -> bool:
        """True when the last reply asked for confirmation (UI should re-listen)."""
        return self._pending is not None

    async def route(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        on_delta: Any = None,
    ) -> RouteResult:
        """Route one utterance. ``on_delta`` (optional ``Callable[[str], None]``)
        receives LLM content chunks as they stream, so the voice loop can speak
        sentences while the reply is still generating."""
        text = text.strip()
        started = time.perf_counter()

        # Announce the utterance so any UI (HUD) can show it, whatever the source.
        await self._bus.emit(Event(EventType.TRANSCRIPT, text))
        result = await self._route_inner(text, context or {}, on_delta)
        result.latency_ms = (time.perf_counter() - started) * 1000.0

        self.stats[result.path] += 1
        if self.metrics is not None:
            # to_thread: a slow disk must never delay the spoken reply's caller.
            await asyncio.to_thread(
                self.metrics.record, result.path.value, result.skill_name,
                result.latency_ms,
            )
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

    async def _route_inner(
        self, text: str, context: dict[str, Any], on_delta: Any = None
    ) -> RouteResult:
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
        return await self._llm_reply(text, context, on_delta)

    async def _run_skill(self, skill: Skill, request: SkillRequest) -> RouteResult:
        """Execute a fast-path skill, deferring for confirmation if it asks."""
        # LION MODE overrides the PC-control switch: the whole point of the
        # mode is "stop asking me". The path whitelist is NOT overridden —
        # see SafetyConfig.lion_mode for why that line is drawn there.
        lion = self._settings.safety.lion_mode
        if skill.controls_pc and not (self._settings.safety.pc_control_enabled or lion):
            return RouteResult(path=RoutePath.FAST, speech=PC_CONTROL_OFF_REPLY,
                              skill_name=skill.name)
        if lion:
            # Skills gate themselves on this too (the confirmation branch reads
            # it), so tell them the answer is already yes.
            request.context = {**request.context, "confirmed": True}
        outcome = await skill.execute(request)
        if (outcome.needs_confirmation and self._settings.safety.confirm_destructive
                and not lion):
            # Stash the request; the next utterance is treated as the yes/no.
            self._pending = (skill, request)
        elif outcome.success and not skill.requires_confirmation:
            # Personality (M7) decorates only successful, non-gated outcomes —
            # errors, safety prompts, and destructive actions stay literal.
            outcome.speech = self._persona.decorate(
                outcome.speech, skill_name=skill.name, user_text=request.text
            )
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
        provider = self._settings.llm.provider
        if provider in ("openai", "anthropic"):
            return OFFLINE_CLOUD_REPLY
        if provider == "claude-code":
            return OFFLINE_CLI_REPLY.format(name="Claude Code")
        if provider == "codex":
            return OFFLINE_CLI_REPLY.format(name="Codex")
        return OFFLINE_LLM_REPLY

    async def _llm_reply(
        self, text: str, context: dict[str, Any], on_delta: Any = None
    ) -> RouteResult:
        if not self.model:
            return RouteResult(path=RoutePath.LLM, speech=self._offline_reply)

        tools = build_tools(self._registry)
        if not self._settings.safety.pc_control_enabled:
            # PC control is off: actuation tools aren't even offered, so the
            # model answers around them instead of calling and being refused.
            gated = {s.name for s in self._registry.all() if s.controls_pc}
            tools = [t for t in tools if t["function"]["name"] not in gated]
        try:
            facts = await asyncio.to_thread(
                self.facts.relevant, text, self._settings.memory.max_facts
            )
        except Exception:  # a broken facts DB must never take down the LLM path
            logger.exception("could not load remembered facts")
            facts = []
        # Answer in the language the user spoke. The voice loop puts the code
        # Whisper detected into the context; without it we say nothing and the
        # model keeps its default (English), which is the right fallback for
        # typed input.
        prompt = system_prompt(self._settings.personality, facts)
        spoken = context.get("language")
        if spoken and str(spoken).lower() != "en":
            from core import languages

            name = languages.english_name(spoken, "")
            if name:
                prompt += (
                    f"\n\nThe user is speaking {name}. Reply in {name}, "
                    f"in plain spoken prose."
                )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": prompt,
            },
            *self.conversation.recent_messages(),   # rolling context for follow-ups
            {"role": "user", "content": text},
        ]

        # Passed as **kwargs so test fakes with a plain chat(model, messages,
        # tools=...) signature keep working when streaming isn't requested.
        stream_kw = {"on_delta": on_delta} if on_delta is not None else {}
        try:
            for _ in range(MAX_TOOL_ROUNDS):
                message = await self._llm.chat(self.model, messages, tools=tools, **stream_kw)
                tool_calls = message.get("tool_calls") or []
                if not tool_calls:
                    reply = (message.get("content") or "").strip()
                    if reply and _looks_like_tool_json(reply):
                        # Small model leaked a botched tool call as text; a plain
                        # retry (no tools) gets a clean spoken answer.
                        retry = await self._llm.chat(self.model, messages)
                        reply = (retry.get("content") or "").strip()
                    return RouteResult(path=RoutePath.LLM, speech=reply or EMPTY_REPLY)

                messages.append(message)  # the assistant turn that requested tools
                pending = await self._run_tool_calls(tool_calls, text, context, messages)
                if pending is not None:
                    return pending  # a destructive tool needs confirmation first

            # Ran out of rounds: ask once more for a plain answer.
            message = await self._llm.chat(self.model, messages)
            return RouteResult(
                path=RoutePath.LLM,
                speech=(message.get("content") or "").strip() or EMPTY_REPLY,
            )
        except LLMUnavailableError as exc:
            logger.warning("LLM path unavailable: %s", exc)
            return RouteResult(path=RoutePath.LLM, speech=self._offline_reply)

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
            gated_skill = self._registry.get(name)
            lion = self._settings.safety.lion_mode
            if (gated_skill is not None and gated_skill.controls_pc
                    and not (self._settings.safety.pc_control_enabled or lion)):
                # Belt to the tool-filter's braces: even a hallucinated call
                # to an actuation tool is refused when PC control is off.
                result = SkillResult(PC_CONTROL_OFF_REPLY, success=False)
            else:
                if lion:
                    context = {**context, "confirmed": True}
                result = await dispatch_tool(self._registry, name, args, context)
            used_skill = name

            if (result.needs_confirmation
                    and self._settings.safety.confirm_destructive and not lion):
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
