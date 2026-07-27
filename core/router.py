"""Intent Router — the heart of MEDO.

For each utterance it decides between two paths and records which one handled it:

* **FAST**  — a registered skill's regex matched; execute deterministically, no LLM.
* **LLM**   — no rule matched; hand the text to Ollama to converse (tool calling in M3).

Routing stats are tallied for the README.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from core.config import Settings
from core.events import Event, EventBus, EventType, RoutePath
from core.facts import FactsStore
from core.memory import ConversationMemory
from core.metrics import MetricsStore
from core.safety import is_affirmative, is_negative
from llm.client import CLI_PROVIDERS, LLMUnavailableError, OllamaClient
from llm.prompts import system_prompt
from llm.tools import build_tools, coerce_args, dispatch_tool
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult

#: A query "needs live info" (search the web, current events) when it trips one
#: of these. Used only to decide whether to borrow the tool-brain — a false
#: positive just means a slightly slower answer, never a wrong one.
_LIVE_INFO_RE = re.compile(
    r"\b(latest|news|headlines?|today|tonight|right\s+now|currently|current|"
    r"recent(?:ly)?|happening|search|look\s+up|google|online|internet|"
    r"who\s+(?:is|are|won|winning)|when\s+(?:is|does|did)|score|prices?|stocks?|"
    r"weather|forecast|release[ds]?|this\s+(?:week|month|year)|"
    r"tariff\w*|sanction\w*|election\w*|inflation|gdp|market\w*|"
    r"вест\w*|новост\w*|најнов\w*|што\s+се\s+случува|пребар\w*|"
    r"царин\w*|санкц\w*|на\s+интернет)\b",
    re.IGNORECASE,
)

#: Skills whose answer is live/online info. When one of these fires, the NEXT
#: question is treated as a follow-up that may also need live info — so
#: "is Macedonia on that tariff list?" right after the news reaches the
#: tool-brain even though it trips no keyword of its own.
LIVE_INFO_SKILLS = frozenset({
    "news", "weather", "briefing", "web_search", "web_fetch",
})

#: A follow-up worth chaining onto the previous live-info turn: a question, or a
#: continuation. Liberal on purpose — a false positive only borrows the (local)
#: tool-brain for one turn, it never changes the answer.
_FOLLOWUP_RE = re.compile(
    r"^\s*(?:and|also|but|what|which|who|whom|whose|where|when|why|how|is|are|"
    r"was|were|do|does|did|can|could|would|will|should|has|have|had|any|is\s+"
    r"there|other|more|about|tell|дали|што|кој|каде|кога|зошто|како|а)\b",
    re.IGNORECASE,
)


def _is_followup_question(text: str) -> bool:
    return "?" in text or bool(_FOLLOWUP_RE.search(text))

#: How many tool rounds before we force a final text answer (loop guard).
MAX_TOOL_ROUNDS = 4
#: Re-check a DOWN council tool-brain no more than once per this many seconds,
#: so a parallel convene never fires the blocking ~2s probe per specialist.
_COUNCIL_PROBE_TTL = 30.0
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
    #: True only when ``speech`` is the model's own reply that was already
    #: emitted sentence-by-sentence via ``on_delta`` (so the voice loop must
    #: NOT speak it again). Tool answers and confirmation prompts leave this
    #: False — their text never went through the streamer, so the loop must
    #: voice them even when a filler preamble WAS streamed first.
    streamed_reply: bool = False


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
        #: Lazily-built local Ollama client used to answer live-info queries when
        #: the selected brain is a CLI agent that can't use MEDO's tools.
        self._tool_brain: OllamaClient | None = None
        #: Set once the tool-brain has answered an availability probe True (it
        #: rarely drops mid-session), so the council never re-probes after. A
        #: DOWN result isn't sticky — it's re-checked after _COUNCIL_PROBE_TTL so
        #: a later `ollama serve` is picked up — but `_tool_brain_probe_at`
        #: bounds the blocking ~2s probe to once per TTL, so a parallel convene
        #: with the tool-brain down can't fire it per specialist.
        self._tool_brain_ok = False
        self._tool_brain_probe_at: float | None = None
        #: True right after a live-info turn (news/weather/web…), so the next
        #: question is treated as a follow-up that may also need live info even
        #: if it trips no keyword of its own ("is Macedonia on that list?").
        self._last_live_info = False
        #: Set within a turn when a live-info tool (web_search…) runs on the LLM
        #: path, so the follow-up chain survives web_search's skill_name-less
        #: synthesis result. Reset at the top of every route().
        self._turn_used_live_info = False
        #: Active model for the LLM path; set by the app / model picker.
        self.model: str | None = settings.llm.default_model
        #: Routing tallies (this session) + persisted per-request metrics that
        #: feed the README's Performance table (python -m core.metrics --report).
        self.stats: dict[RoutePath, int] = {
            RoutePath.FAST: 0, RoutePath.SEMANTIC: 0, RoutePath.LLM: 0}
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
        # Load the yes/no confirmation banks for the ACTIVE languages (S3): the
        # gate matches only the languages MEDO is currently listening for.
        from core.safety import configure_confirm_words

        configure_confirm_words(settings.active_languages(), settings.primary_language())
        #: Personality layer (M7): occasionally decorates fast-path replies.
        #: Tests may replace it (or inject a seeded rng) for determinism.
        from core.persona import Persona

        self._persona = Persona(settings.personality,
                                active=settings.active_languages(),
                                primary=settings.primary_language())
        #: Tier-2 semantic route index (M2.5), built lazily on the first miss —
        #: embedding the opt-in skills is one Ollama call, then cached in SQLite.
        #: None = not built yet; False = permanently unavailable (no embedder).
        self._embedder = embedder
        self._route_index: Any = None
        #: Adaptive route memory (M2.5e): learned exemplars consulted alongside
        #: the curated index. Same None/False sentinels. `_last_query_vec` caches
        #: the utterance embedding computed in _semantic_route so a learn on the
        #: same turn costs zero extra embeds.
        self._route_memory: Any = None
        self._last_query_vec: Any = None

    async def _ensure_route_index(self):
        """Build the semantic route index once; None when it can't be built."""
        if self._route_index is not None:
            # Explicit sentinel test (not truthiness): a SkillRouteIndex also
            # defines __len__, so `or None` would misfire if a stored index were
            # ever empty. It isn't today (the len==0 guard below stores False),
            # but keep this robust and consistent with _ensure_route_memory.
            return self._route_index if self._route_index is not False else None
        if self._embedder is None:
            self._route_index = False
            return None
        from core.route_index import SkillRouteIndex

        idx = SkillRouteIndex(self._settings.memory.db_path, self._embedder,
                              self._settings.memory.embed_model or "none")
        # Only skills that OPT IN with curated routing_phrases are eligible: a
        # bare description is too vague to route on, and actuation skills that
        # need extracted parameters belong on the LLM path, not here.
        sources = [s for s in self._registry.all()
                   if getattr(s, "routing_phrases", None)
                   and self._semantic_safe(s)]
        try:
            await asyncio.to_thread(idx.build, sources)
        except Exception:
            logger.exception("semantic route index build failed — tier disabled")
            self._route_index = False
            return None
        if len(idx) == 0:
            self._route_index = False
            return None
        self._route_index = idx
        logger.info("semantic route index ready: %d skill(s)", len(idx))
        return idx

    @staticmethod
    def _semantic_safe(skill: "Skill") -> bool:
        """Can this skill be REACHED BY MEANING without a parsed match?

        On the semantic tier a skill is dispatched with a bare SkillRequest — no
        regex ``match``, no ``args``. That is only safe when the skill is not
        destructive/gated (``controls_pc`` / ``requires_confirmation`` go through
        the LLM+confirmation path, never a meaning shortcut) AND it can actually
        run argument-free: either its tool schema marks nothing required, or it
        promises via ``semantic_from_text`` to derive what it needs from
        ``request.text``. Without this gate, flipping the HUD SEMANTIC TIER to
        LIVE would make an arg-hungry query skill deflect ("What should I search
        for?") on a perfectly clear request.
        """
        if getattr(skill, "controls_pc", False) or \
                getattr(skill, "requires_confirmation", False):
            return False
        if getattr(skill, "semantic_from_text", False):
            return True
        try:
            required = (skill.tool_schema().get("function", {})
                        .get("parameters", {}).get("required") or [])
        except Exception:      # a broken tool_schema shouldn't crash index build
            return False
        return not required

    @staticmethod
    def _is_learnable(skill: "Skill") -> bool:
        """A skill safe to LEARN and REPLAY on the semantic tier: a query skill
        (no PC control, no confirmation) that already opted into the curated tier
        with routing_phrases. A learned exemplar can therefore only ever shortcut
        to something the curated tier could also reach — never a destructive one.
        """
        return (not getattr(skill, "controls_pc", False)
                and not getattr(skill, "requires_confirmation", False)
                and bool(getattr(skill, "routing_phrases", None)))

    async def _ensure_route_memory(self):
        """Build the learned-exemplar store once; None when it can't/shouldn't be
        used (no embedder, or the feature is off). Same sentinels as the index."""
        if self._route_memory is not None:
            # An EMPTY RouteMemory is falsy (it defines __len__), and it
            # legitimately starts empty — so `or None` would collapse a
            # valid-but-empty store to None and the very first exemplar could
            # never be written (chicken-and-egg). Test the False sentinel itself.
            return self._route_memory if self._route_memory is not False else None
        if self._embedder is None or not self._settings.router.route_memory_enabled:
            self._route_memory = False
            return None
        from core.route_memory import RouteMemory

        self._route_memory = RouteMemory(self._settings.memory.db_path,
                                         self._settings.memory.embed_model or "none")
        return self._route_memory

    async def _semantic_route(self, text: str) -> "tuple[Skill, float] | None":
        """Best skill for ``text`` by MEANING, or None (embedder down / no
        confident match). Consults the curated index AND — when enabled — the
        learned route memory, embedding the utterance once for both. Never raises
        into routing; byte-identical to the curated-only path when memory is off.
        """
        idx = await self._ensure_route_index()
        mem = await self._ensure_route_memory()
        # Preserve today's short-circuit: with nothing to consult, don't embed.
        if idx is None and mem is None:
            return None
        if self._embedder is None:
            return None
        try:
            qv = await asyncio.to_thread(self._embedder, [text])
        except Exception:
            return None
        if not qv:
            return None
        self._last_query_vec = qv[0]
        best: tuple[Skill, float] | None = None
        if idx is not None:
            hit = idx.match(qv[0], self._settings.router.semantic_threshold,
                            self._settings.router.semantic_margin)
            if hit is not None:
                skill = self._registry.get(hit[0])
                if skill is not None:
                    best = (skill, hit[1])
        if mem is not None:
            try:
                mhit = await asyncio.to_thread(
                    mem.match, qv[0], self._settings.router.route_memory_threshold)
            except Exception:
                mhit = None
            if mhit is not None:
                skill = self._registry.get(mhit[0])
                # Guard replay too: only shortcut to a still-learnable skill, and
                # let a curated hit win an exact tie (a learned one takes it only
                # on a strictly higher score).
                if (skill is not None and self._is_learnable(skill)
                        and (best is None or mhit[1] > best[1])):
                    best = (skill, mhit[1])
        return best

    async def _maybe_learn_route(self, text: str, result: RouteResult,
                                 answering_confirmation: bool) -> None:
        """Learn (utterance -> skill) from a CLEAN single-skill LLM resolution.

        Deliberately narrow: only the LLM path, only a real skill_name, only when
        exactly one tool was called, never while a destructive action is mid
        confirmation, and only for a learnable (query) skill. Everything else —
        synthesis answers (skill_name=None), multi-tool turns, errors — is skipped.
        """
        if not self._settings.router.route_memory_enabled or answering_confirmation:
            return
        if result.path is not RoutePath.LLM or not result.skill_name:
            return
        if self._pending is not None:               # a destructive tool is waiting
            return
        if result.data.get("tool_call_count") != 1:  # exactly one resolved skill
            return
        skill = self._registry.get(result.skill_name)
        if skill is None or not self._is_learnable(skill):
            return
        await self._learn_route(text, skill)

    async def _learn_route(self, text: str, skill: "Skill") -> None:
        """Store one learned exemplar. Best-effort — never breaks a turn."""
        try:
            mem = await self._ensure_route_memory()
            if mem is None:
                return
            from core.safety import _normalize

            norm = _normalize(text)
            if len(norm) < 3:
                return
            vec = self._last_query_vec
            if vec is None:                          # semantic tier was off; embed now
                qv = await asyncio.to_thread(self._embedder, [text])
                vec = qv[0] if qv else None
            if vec is None:
                return
            await asyncio.to_thread(mem.remember, norm, skill.name, vec,
                                    self._settings.router.route_memory_max)
        except Exception:
            logger.debug("route-memory learn failed", exc_info=True)

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
        on_llm_start: Any = None,
    ) -> RouteResult:
        """Route one utterance. ``on_delta`` (optional ``Callable[[str], None]``)
        receives LLM content chunks as they stream, so the voice loop can speak
        sentences while the reply is still generating."""
        text = text.strip()
        started = time.perf_counter()

        # Capture BEFORE routing clears it: was this utterance a yes/no answering
        # a pending confirmation? (self._pending is cleared inside _route_inner.)
        answering_confirmation = self._pending is not None
        self._turn_used_live_info = False
        # Cleared each turn so a stale embedding can never attach to a later
        # learned route; set by _semantic_route when it embeds this utterance.
        self._last_query_vec = None

        # Announce the utterance so any UI (HUD) can show it, whatever the source.
        await self._bus.emit(Event(EventType.TRANSCRIPT, text))
        result = await self._route_inner(text, context or {}, on_delta, on_llm_start)
        result.latency_ms = (time.perf_counter() - started) * 1000.0

        self.stats[result.path] += 1
        if self.metrics is not None:
            # to_thread: a slow disk must never delay the spoken reply's caller.
            await asyncio.to_thread(
                self.metrics.record, result.path.value, result.skill_name,
                result.latency_ms,
            )
        # Remember the turn, UNLESS this utterance was the yes/no that answered a
        # pending confirmation — that's what shouldn't pollute the context. (The
        # command that TRIGGERED a confirmation is recorded; the "yes" is not.)
        if not answering_confirmation:
            self.conversation.add_turn(text, result.speech)
            # Track whether this turn produced live/online info, so the NEXT
            # question can be recognised as a follow-up that also needs it. Set
            # by a live-info skill OR a live-info tool used on the LLM path
            # (web_search synthesizes, so its RouteResult carries no skill_name).
            self._last_live_info = (result.skill_name in LIVE_INFO_SKILLS
                                    or self._turn_used_live_info)
        # Adaptive route memory (M2.5e): a clean single-skill LLM resolution is
        # worth learning so the same phrasing skips the LLM next time.
        await self._maybe_learn_route(text, result, answering_confirmation)
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
        self, text: str, context: dict[str, Any], on_delta: Any = None,
        on_llm_start: Any = None,
    ) -> RouteResult:
        # --- CONFIRMATION GATE ---
        # A destructive action is waiting on a yes/no; this reply answers it.
        if self._pending is not None:
            return await self._resolve_confirmation(text, context)

        # Expose the PREVIOUS spoken reply so a skill can replay it verbatim
        # ("say that again"). route() calls add_turn() AFTER this, so right now
        # last_reply() is the prior turn's reply, exactly what should be echoed.
        context = {**context, "last_reply": self.conversation.last_reply()}

        # --- FAST PATH ---
        if self._settings.router.fast_path_enabled:
            match = self._registry.find_match(text)
            if match is not None:
                skill, regex_match = match
                request = SkillRequest(text=text, match=regex_match, context=context)
                return await self._run_skill(skill, request)

        # --- SEMANTIC TIER (M2.5) ---
        # No regex matched; try to reach a query-style skill by MEANING before
        # paying for the LLM. Shadow mode logs the would-be route and falls
        # through, so it can be proven on real usage before going live.
        if self._settings.router.semantic_enabled:
            sem = await self._semantic_route(text)
            if sem is not None:
                skill, score = sem
                if self._settings.router.semantic_shadow:
                    logger.info("[semantic-shadow] %r -> %s (%.2f); using LLM",
                                text, skill.name, score)
                else:
                    logger.info("[semantic] %r -> %s (%.2f)", text, skill.name, score)
                    return await self._run_skill(
                        skill, SkillRequest(text=text, context=context),
                        path=RoutePath.SEMANTIC)

        # --- LLM PATH ---
        # Signal the caller (the voice loop) that we've committed to the LLM,
        # which is slow enough to want a filler; the fast path above never gets
        # here, so a filler armed on this hook is LLM-only by construction.
        if on_llm_start is not None:
            try:
                on_llm_start()
            except Exception:      # a UI nicety must never break routing
                logger.debug("on_llm_start hook raised", exc_info=True)
        return await self._llm_reply(text, context, on_delta)

    async def _run_skill(self, skill: Skill, request: SkillRequest,
                         path: RoutePath = RoutePath.FAST) -> RouteResult:
        """Execute a fast-path (or semantic-tier) skill, deferring for
        confirmation if it asks.

        There is deliberately NO profile that relaxes this gate. Lion mode
        (mode.lion) surfaces extra skills and reskins the HUD, but the
        confirmation and PC-control checks below run identically whether it is
        on or off — see docs/Decisions.md. ``path`` only labels the result
        (FAST vs SEMANTIC); the safety checks are the same either way.
        """
        if skill.controls_pc and not self._settings.safety.pc_control_enabled:
            return RouteResult(path=path, speech=PC_CONTROL_OFF_REPLY,
                              skill_name=skill.name)
        outcome = await skill.execute(request)
        if (outcome.needs_confirmation and self._settings.safety.confirm_destructive):
            # Stash the request; the next utterance is treated as the yes/no.
            self._pending = (skill, request)
        elif outcome.success and not skill.requires_confirmation:
            # Personality (M7) decorates only successful, non-gated outcomes —
            # errors, safety prompts, and destructive actions stay literal.
            outcome.speech = self._persona.decorate(
                outcome.speech, skill_name=skill.name, user_text=request.text,
                language=request.context.get("language"),
            )
        return RouteResult(
            path=path,
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

    def _tool_brain_client(self) -> OllamaClient | None:
        """A local Ollama client for the configured tool-brain model, or None.

        Built lazily and reused. Same client class as the primary brain, but
        pinned to provider=ollama so it gets MEDO's tool schemas (incl.
        web_search) that the CLI agents never receive.
        """
        name = self._settings.llm.tool_brain_model
        if not name:
            return None
        if self._tool_brain is None:
            cfg = self._settings.llm.model_copy()
            cfg.provider = "ollama"
            cfg.default_model = name
            self._tool_brain = type(self._llm)(cfg)
        return self._tool_brain

    def council_brain(self) -> "tuple[OllamaClient, str | None]":
        """(client, model) for the specialist council / circuit helper.

        The council role-plays experts on a plain text prompt — it wants a FAST
        model, not MEDO's tools or a web connection. A CLI-agent brain is the
        wrong tool twice over: it spawns a whole process PER specialist, and
        ``convene_council`` fires two or three of them in PARALLEL, which is
        exactly what left "the agents" hanging. So borrow the local tool-brain
        when the selected brain is a CLI agent; otherwise use the selected brain
        unchanged (Ollama users keep the model they picked).

        The availability probe (a blocking ~2s HTTP GET) is bounded to at most
        once per _COUNCIL_PROBE_TTL — cached forever once it succeeds — so it
        never repeats per specialist across the parallel consults of a convene,
        whether the tool-brain is up OR down.
        """
        if self._settings.llm.provider in CLI_PROVIDERS:
            tb = self._tool_brain_client()
            if tb is not None and self._tool_brain_available(tb):
                return tb, self._settings.llm.tool_brain_model
        return self._llm, self.model

    def _tool_brain_available(self, tb: OllamaClient) -> bool:
        """Is the local tool-brain reachable? Probes at most once per
        _COUNCIL_PROBE_TTL. Once True it stays cached (Ollama rarely drops
        mid-session); a False result is re-checked only after the TTL, so a
        parallel convene with the tool-brain down doesn't re-block per specialist
        yet a later `ollama serve` is still picked up. council_brain() has no
        await before this call, so the first gathered specialist's probe
        completes and stamps the time before any other specialist runs.
        """
        if self._tool_brain_ok:
            return True
        now = time.monotonic()
        if (self._tool_brain_probe_at is not None
                and now - self._tool_brain_probe_at < _COUNCIL_PROBE_TTL):
            return False                       # recently probed down — don't re-block
        self._tool_brain_probe_at = now
        if tb.is_available():
            self._tool_brain_ok = True
            return True
        return False

    def _pick_brain(self, text: str) -> tuple[OllamaClient, str | None, bool]:
        """Choose (client, model, borrowed) for this turn.

        Borrow the local tool-brain only when the selected brain is a CLI agent
        (which can't use MEDO's tools) AND the query looks like it needs live
        info. Otherwise use the selected brain unchanged — so normal chat still
        goes to the model you picked, and MEDO 'returns' to it automatically.
        """
        if (self._settings.llm.provider in CLI_PROVIDERS
                and self._wants_live_info(text)):
            tb = self._tool_brain_client()
            if tb is not None and tb.is_available():
                logger.info("auto tool-brain: %r needs live info -> %s",
                            text, self._settings.llm.tool_brain_model)
                return tb, self._settings.llm.tool_brain_model, True
        return self._llm, self.model, False

    def _wants_live_info(self, text: str) -> bool:
        """Whether this turn likely needs live/online info the CLI agent can't get.

        Either the query itself trips the keywords, OR it's a follow-up question
        right after a live-info answer — the case that made "is Macedonia on the
        tariff list?" (after the news) deflect with "grant me web access", because
        it names no keyword of its own. A false positive only borrows the local
        tool-brain for one turn; it never changes the answer.
        """
        if _LIVE_INFO_RE.search(text):
            return True
        return self._last_live_info and _is_followup_question(text)

    def _failure_reply(self, exc: Exception) -> str:
        """What to say when an LLM call FAILED — with the real reason.

        A CLI agent that is installed and on PATH but exited with an error (a
        usage limit, an auth hiccup, a bad model name) used to collapse into
        "make sure it is installed, logged in, and on your PATH", which sends
        you hunting a problem you don't have. Only claim that when the CLI
        genuinely isn't there; otherwise repeat what the agent actually said.
        """
        provider = self._settings.llm.provider
        if provider not in CLI_PROVIDERS:
            return self._offline_reply
        name = "Claude Code" if provider == "claude-code" else "Codex"
        detail = str(exc).strip()
        low = detail.lower()
        if ("not installed" in low or "not on path" in low
                or "could not launch" in low):
            return OFFLINE_CLI_REPLY.format(name=name)     # genuinely missing
        if "timed out" in low:
            return (f"{name} took too long to answer, so I stopped waiting. "
                    f"Try again, or switch provider in the HUD.")
        # Real error text from the agent — say it, trimmed to something speakable.
        said = detail.split(":", 1)[-1].strip() if ":" in detail else detail
        said = " ".join(said.split())
        if len(said) > 140:
            said = said[:137].rstrip() + "…"
        return f"{name} couldn't answer: {said}" if said else \
            OFFLINE_CLI_REPLY.format(name=name)

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
        # Selected brain for normal chat; auto-borrow the local tool-brain when
        # a CLI agent hits a live-info query (then we're back on the selected
        # brain next turn — nothing is mutated).
        llm, model, borrowed = self._pick_brain(text)
        if not model:
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
                message = await llm.chat(model, messages, tools=tools, **stream_kw)
                tool_calls = message.get("tool_calls") or []
                if not tool_calls:
                    reply = (message.get("content") or "").strip()
                    retried = False
                    if reply and _looks_like_tool_json(reply):
                        # Small model leaked a botched tool call as text; a plain
                        # retry (no tools) gets a clean spoken answer.
                        retry = await llm.chat(model, messages)
                        reply = (retry.get("content") or "").strip()
                        retried = True
                    # This reply IS what streamed through on_delta (unless we
                    # had to retry off-stream) — flag it so the loop doesn't
                    # re-speak it. A retried reply never streamed, so leave it
                    # False and let the loop voice it.
                    return RouteResult(
                        path=RoutePath.LLM, speech=reply or EMPTY_REPLY,
                        streamed_reply=(on_delta is not None and not retried
                                        and bool(reply)))

                messages.append(message)  # the assistant turn that requested tools
                pending = await self._run_tool_calls(tool_calls, text, context, messages)
                if pending is not None:
                    return pending  # a destructive tool needs confirmation first

            # Ran out of rounds: ask once more for a plain answer.
            message = await llm.chat(model, messages)
            return RouteResult(
                path=RoutePath.LLM,
                speech=(message.get("content") or "").strip() or EMPTY_REPLY,
            )
        except LLMUnavailableError as exc:
            logger.warning("LLM path unavailable: %s", exc)
            return RouteResult(path=RoutePath.LLM, speech=self._failure_reply(exc))

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
            # Coerce string-typed args now, so a stashed confirmation request
            # (re-executed on "yes") carries clean args too — not just dispatch.
            args = coerce_args(gated_skill, args)
            if (gated_skill is not None and gated_skill.controls_pc
                    and not self._settings.safety.pc_control_enabled):
                # Belt to the tool-filter's braces: even a hallucinated call
                # to an actuation tool is refused when PC control is off.
                result = SkillResult(PC_CONTROL_OFF_REPLY, success=False)
            else:
                result = await dispatch_tool(self._registry, name, args, context)
            used_skill = name

            if (result.needs_confirmation
                    and self._settings.safety.confirm_destructive):
                skill = self._registry.get(name)
                if skill is not None:
                    request = SkillRequest(text=text, args=args, context=context)
                    self._pending = (skill, request)
                return RouteResult(path=RoutePath.LLM, speech=result.speech, skill_name=name)

            if name in LIVE_INFO_SKILLS:
                # web_search answers via synthesis (no skill_name on the final
                # RouteResult), so flag it here or the follow-up chain breaks.
                self._turn_used_live_info = True
            messages.append({"role": "tool", "name": name, "content": result.speech})
            if name in SYNTHESIS_TOOLS:
                needs_synthesis = True
            elif not result.success:
                # A failed tool must not be silently dropped while its siblings
                # speak their success — otherwise the user hears "opened X" and
                # never learns Y failed (success-by-omission). Its message is
                # already in `messages`, so force the model to synthesize a
                # reply that reconciles the whole batch instead of returning
                # only the successes.
                needs_synthesis = True
            elif result.speech:
                direct.append(result.speech)

        # Structured tools already speak for themselves — answer directly.
        if not needs_synthesis and direct:
            # tool_call_count lets adaptive route memory learn ONLY clean,
            # single-skill resolutions (existing callers ignore data).
            return RouteResult(path=RoutePath.LLM, speech=" ".join(direct),
                               skill_name=used_skill,
                               data={"tool_call_count": len(tool_calls)})
        return None


# --- M2.5a: build/inspect the semantic route index from the CLI --------------

def build_route_index(settings=None):
    """Construct a SkillRouteIndex over the live registry and build it.

    Reuses the same local embedder as facts/RAG (memory.embed_model on the
    Ollama host). Returns (index, report). Used by ``--index`` and by the app
    later; kept here so the sidecar/tests share one construction path.
    """
    from core.config import load_settings
    from core.embeddings import embed_texts
    from core.route_index import SkillRouteIndex

    settings = settings or load_settings()
    model, host = settings.memory.embed_model, settings.llm.host
    # Generous timeout: this is a one-time batched startup build, and the first
    # call also pays for the embed model loading into Ollama. Per-utterance
    # query embedding (M2.5b) uses the default short timeout instead.
    embedder = (lambda texts: embed_texts(texts, model, host, timeout=60.0)) \
        if model else None
    # build_registry lives in main; import lazily to avoid a heavy import here.
    from main import Announcer, build_registry

    registry = build_registry(settings, Announcer())
    index = SkillRouteIndex(settings.memory.db_path, embedder, model or "none")
    report = index.build(registry.all())
    index.prune(s.name for s in registry.all())
    return index, report


def _print_index_report(report) -> None:
    print(f"\nSemantic route index — embedder: "
          f"{'enabled' if report.enabled else 'UNAVAILABLE (Ollama/model down)'}\n")
    print(f"  {'SKILL':24} {'PHRASES':>7}  STATUS")
    print("  " + "-" * 44)
    for row in report.rows:
        print(f"  {row.skill:24} {row.phrases:>7}  {row.status}")
    print(f"\n  {len(report.rows)} skills · {report.cached} cached · "
          f"{report.fresh} fresh · {report.skipped} skipped\n")


def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m core.router",
                                     description="MEDO router tools (M2.5)")
    parser.add_argument("--index", action="store_true",
                        help="(re)build and print the semantic skill index")
    args = parser.parse_args(argv)
    if args.index:
        _, report = build_route_index()
        _print_index_report(report)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
