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
from core.safety import PathWhitelist, is_affirmative, is_negative
from llm.client import CLI_PROVIDERS, LLMUnavailableError, OllamaClient
from llm.prompts import system_prompt
from llm.tools import build_tools, coerce_args, dispatch_tool
from security.audit import (
    CLOUD_CALL,
    CONFIRM_DENIED,
    CONFIRM_GRANTED,
    POLICY_DENY,
    AuditLog,
)
from security.capabilities import effective_capabilities
from security.policy import Actor, PolicyEngine, Provenance
from security.secrets import Secrets
from security.trust import DANGEROUS_CAPS, UNTRUSTED_CONTENT_TOOLS, mark_untrusted
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
#: Retry a failed/empty semantic-index build at most this often, so a cold embed
#: model recovers (the build isn't latched off) without every miss re-attempting.
_ROUTE_INDEX_RETRY_TTL = 20.0
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
        #: The central deny-by-default policy engine (S1). Every side-effectful
        #: route is gated through it; it reads settings.security/safety live, so a
        #: runtime PC-control toggle is honoured at once. Built here (not injected)
        #: so existing callers that construct a Router unchanged keep working.
        self._policy = PolicyEngine(
            settings.security, settings.safety,
            PathWhitelist(settings.safety.whitelist_dirs))
        if getattr(settings.security, "yolo", False):
            logger.warning("⚠ SECURITY LAYER OFF (security.yolo) — every action "
                           "is allowed and confirmations are auto-accepted. "
                           "DEV MACHINE ONLY; never ship with this on.")
        #: Tamper-evident audit trail (S6), off unless security.audit_enabled.
        #: Secret values are redacted out of every entry.
        self._audit_log = (AuditLog(redactor=Secrets(settings).redact)
                           if getattr(settings.security, "audit_enabled", False)
                           else None)
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
        #: Trust boundary (S3): tainted once an untrusted-content tool runs.
        self._turn_untrusted_context = False
        #: Set within a turn when an ambiguous confirmation reply (neither yes nor
        #: no) is cancelled and RE-ROUTED as a genuine command — so route() records
        #: that fresh command as a normal turn instead of suppressing it like a
        #: yes/no answer. Reset at the top of every route().
        self._rerouted_from_confirmation = False
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
        #: A skill awaiting a free-text follow-up (e.g. self-dev's "what should I
        #: fix?"), so the NEXT utterance is captured as its answer.
        self._pending_reply: tuple[Skill, SkillRequest] | None = None
        #: True when that pending reply is a yes/no OFFER ("want the breakdown?"),
        #: so a reply that is neither yes nor no is a fresh command and re-routes
        #: instead of being swallowed into the offer. Set with `_pending_reply`.
        self._reply_is_offer = False
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
        #: Monotonic time of the last route-index build attempt, so an empty
        #: build (cold embedder) is retried on a later miss instead of latching
        #: the tier off — but not re-attempted on every single miss.
        self._route_index_retry_at: float | None = None
        #: Adaptive route memory (M2.5e): learned exemplars consulted alongside
        #: the curated index. Same None/False sentinels. `_last_query_vec` caches
        #: the utterance embedding computed in _semantic_route so a learn on the
        #: same turn costs zero extra embeds.
        self._route_memory: Any = None
        self._last_query_vec: Any = None
        #: Spoken tie-break (M2.5f): the two candidate skills MEDO offered on a
        #: near-tie, the ORIGINAL utterance, and its context — awaiting a one-word
        #: choice. Kept entirely SEPARATE from `_pending` (the destructive-action
        #: confirmation gate) so neither state can weaken the other.
        self._pending_choice: "tuple[list[Skill], str, dict] | None" = None
        #: Set when a choice reply was neither offered skill (a no / a different
        #: command) and got re-routed, so route() records it as a normal turn.
        self._rerouted_from_choice = False

    async def _ensure_route_index(self):
        """Build the semantic route index once; None when it can't be built.

        An empty build is treated as TRANSIENT when an embedder is configured —
        the embed model was cold or briefly unreachable for that one call — so
        the sentinel is left None and the next miss retries (bounded by a
        cooldown), rather than latched False for the whole session. Only a
        genuinely absent embedder, or a roster with nothing to route on, latches
        False permanently.
        """
        if self._route_index is not None:
            # Explicit sentinel test (not truthiness): a SkillRouteIndex also
            # defines __len__, so `or None` would misfire if a stored index were
            # ever empty. It isn't today, but keep this robust and consistent
            # with _ensure_route_memory.
            return self._route_index if self._route_index is not False else None
        if self._embedder is None:
            self._route_index = False        # no embedder ever: genuinely inert
            return None
        # Bound retries so a persistently-unreachable embedder can't make every
        # miss re-attempt a build; a cold model recovers on the next miss past
        # the cooldown (mirrors the council tool-brain probe TTL in this file).
        now = time.monotonic()
        if (self._route_index_retry_at is not None
                and now - self._route_index_retry_at < _ROUTE_INDEX_RETRY_TTL):
            return None
        self._route_index_retry_at = now
        from core.route_index import SkillRouteIndex

        idx = SkillRouteIndex(self._settings.memory.db_path, self._embedder,
                              self._settings.memory.embed_model or "none")
        # Only skills that OPT IN with curated routing_phrases are eligible: a
        # bare description is too vague to route on, and actuation skills that
        # need extracted parameters belong on the LLM path, not here.
        sources = [s for s in self._registry.all()
                   if getattr(s, "routing_phrases", None)
                   and self._semantic_safe(s)]
        if not sources:
            self._route_index = False        # nothing opts in: nothing to route to
            return None
        try:
            await asyncio.to_thread(idx.build, sources)
        except Exception:
            # A crash mid-build is transient too (a locked DB, a flaky embed
            # call) — leave the sentinel None so a later miss can rebuild.
            logger.exception("semantic route index build failed — will retry")
            return None
        if len(idx) == 0:
            # Every source was skipped: the embed model was down/cold for this
            # one call. Do NOT latch False — that killed the tier for the whole
            # session on a single cold-start miss. Retry on the next miss.
            logger.info("semantic route index empty (embed model cold/down?) "
                        "— will retry")
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
        # Final safety invariant, stated once at the dispatch decision: the
        # semantic tier must NEVER reach a PC-controlling or confirmation-gated
        # skill by meaning. The index build and the learn/replay guards already
        # exclude those, but re-assert it here so a stale cached vector or a
        # future refactor can't quietly open a meaning-shortcut to a destructive
        # action — the one route that has no regex and no confirmation prompt.
        if best is not None and not self._semantic_safe(best[0]):
            logger.warning("semantic tier rejected unsafe skill %r at dispatch",
                           best[0].name)
            return None
        return best

    async def _semantic_clarify(self, text: str) -> "list[Skill] | None":
        """The two safe skills to offer on a near-tie, or None.

        Reuses the query vector :meth:`_semantic_route` just cached (a near-tie
        decline still leaves it set), so there is NO second embed. Returns
        exactly two distinct ``_semantic_safe`` skills in score order, or None
        when there is no clarifiable tie or the index isn't ready.
        """
        vec = self._last_query_vec
        if vec is None:                       # tier was off / didn't embed
            return None
        idx = await self._ensure_route_index()
        if idx is None:
            return None
        pair = idx.top_pair(vec, self._settings.router.semantic_threshold,
                            self._settings.router.semantic_margin)
        skills: list[Skill] = []
        seen: set[str] = set()
        for name, _score in pair:
            if name in seen:
                continue
            skill = self._registry.get(name)
            if skill is not None and self._semantic_safe(skill):
                seen.add(name)
                skills.append(skill)
        return skills if len(skills) == 2 else None

    def _clarify_question(self, skills: "list[Skill]") -> str:
        """A one-word either/or prompt naming the two candidate skills."""
        from core.agents import _STAR_LABELS, prettify

        def label(s: "Skill") -> str:
            return (_STAR_LABELS.get(s.name) or prettify(s.name)
                    or s.name.replace("_", " ")).lower()

        return f"Did you mean the {label(skills[0])}, or the {label(skills[1])}?"

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
    def registry(self) -> SkillRegistry:
        """The live skill registry (the companion API reads skills through this)."""
        return self._registry

    @property
    def awaiting_confirmation(self) -> bool:
        """True when the last reply asked for confirmation (UI should re-listen)."""
        return self._pending is not None

    @property
    def awaiting_choice(self) -> bool:
        """True when the last reply asked a one-word tie-break (UI re-listens)."""
        return self._pending_choice is not None

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
        # Scoped to the SAME channel that armed it — a remote /ask or a routine
        # must never resolve, or silently cancel, a destructive action the voice
        # user is mid-confirming (and vice versa).
        answering_confirmation = (
            self._pending is not None
            and self._pending[1].context.get("source")
            == (context or {}).get("source"))
        # Same capture for a pending one-word tie-break: the bare answer must not
        # pollute the follow-up chain, but a re-routed non-answer must.
        answering_choice = (
            self._pending_choice is not None
            and self._pending_choice[2].get("source")
            == (context or {}).get("source"))
        self._turn_used_live_info = False
        # Trust boundary (S3): set when a tool that ingests untrusted external
        # content (RAG/web) runs this turn, so a LATER high-impact action is
        # gated as possibly injection-induced. Reset each turn.
        self._turn_untrusted_context = False
        self._rerouted_from_confirmation = False
        self._rerouted_from_choice = False
        # Cleared each turn so a stale embedding can never attach to a later
        # learned route; set by _semantic_route when it embeds this utterance.
        self._last_query_vec = None

        # Announce the utterance so any UI (HUD) can show it, whatever the source.
        await self._bus.emit(Event(EventType.TRANSCRIPT, text))
        result = await self._route_inner(text, context or {}, on_delta, on_llm_start)
        result.latency_ms = (time.perf_counter() - started) * 1000.0

        # An ambiguous confirmation reply (neither yes nor no) was cancelled and
        # re-routed by _route_inner as a fresh command — that command is a normal
        # turn, so record it and arm the follow-up chain rather than suppress it.
        if self._rerouted_from_confirmation:
            answering_confirmation = False
        if self._rerouted_from_choice:
            answering_choice = False
        # A turn that just ASKED a tie-break question is an incomplete exchange:
        # the resolution turn records the real Q&A (under the original question),
        # so don't also record the bare question turn here. Source-scoped, or a
        # normal turn on ANOTHER channel would be wrongly dropped while a choice
        # is pending on the first (the question turn always armed it same-source).
        asked_choice = (self._pending_choice is not None and not answering_choice
                        and self._pending_choice[2].get("source")
                        == (context or {}).get("source"))

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
        if not answering_confirmation and not answering_choice and not asked_choice:
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
        # YOLO/dev mode: auto-accept the older skill-driven confirmation gate
        # (Power/Quit/file-edit "are you sure?") so nothing blocks development.
        # The policy engine already allows every action in this mode.
        if getattr(self._settings.security, "yolo", False):
            context = {**context, "confirmed": True}
        # --- CHOICE GATE ---
        # A near-tie asked a one-word either/or; this reply answers it. Same-
        # source scoped like the confirmation gate below and kept independent of
        # it (the two pending states never overlap within a turn).
        if (self._pending_choice is not None
                and self._pending_choice[2].get("source") == context.get("source")):
            return await self._resolve_choice(text, context)

        # --- CONFIRMATION GATE ---
        # A destructive action is waiting on a yes/no; this reply answers it —
        # but only when it comes from the channel that armed it. A turn from a
        # different source (e.g. a remote /ask while the voice user is being
        # asked "are you sure?") routes normally and leaves the confirmation
        # armed, so no channel can confirm/cancel another channel's action.
        if (self._pending is not None
                and self._pending[1].context.get("source")
                == context.get("source")):
            return await self._resolve_confirmation(text, context)

        # --- REPLY CAPTURE ---
        # A skill asked a question and wants this utterance as its answer (e.g.
        # self-dev's "what should I fix?"). Same-source scoped, like confirmation.
        # If the user instead said another COMMAND (it matches a fast-path skill),
        # honour that and drop the capture — so "never mind, what time is it"
        # isn't fed back as a bogus answer.
        if (self._pending_reply is not None
                and self._pending_reply[1].context.get("source")
                == context.get("source")):
            reply_skill = self._pending_reply[0]
            is_offer = self._reply_is_offer
            self._pending_reply = None
            self._reply_is_offer = False
            other = self._registry.find_match(text)
            # A yes/no OFFER ("want the breakdown?") only wants a yes or a no. A
            # reply that is neither is a fresh command ("actually, use the
            # electrical engineer to check that") — fall through and re-route it
            # rather than feed it back as a bland "okay" that silently drops the
            # request. Free-text captures (self-dev's "what should I fix?") set no
            # offer flag and still take ANY utterance as the answer, as before.
            offer_answered = (not is_offer
                              or is_affirmative(text) or is_negative(text))
            if offer_answered and (other is None or other[0] is reply_skill):
                captured = SkillRequest(
                    text=text, context={**context, "captured_reply": True})
                return await self._run_skill(reply_skill, captured)

        # Expose the PREVIOUS turn so a skill can replay the reply verbatim
        # ("say that again") or audit it ("second opinion" -> both the prior
        # question and answer). route() calls add_turn() AFTER this, so right now
        # these are the prior turn, exactly what should be echoed/checked.
        context = {**context,
                   "last_reply": self.conversation.last_reply(),
                   "last_question": self.conversation.last_question()}

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

            # Near-tie tie-break (M2.5f): the tier DECLINED (two safe skills too
            # close to call). Rather than pay for the LLM and maybe guess wrong,
            # offer a one-word either/or and route the spoken choice on the
            # semantic path. Honors semantic_shadow (log the would-be ask only).
            if self._settings.router.semantic_clarify_enabled:
                cands = await self._semantic_clarify(text)
                if cands:
                    if self._settings.router.semantic_shadow:
                        logger.info("[semantic-clarify-shadow] %r -> %s",
                                    text, [s.name for s in cands])
                    else:
                        self._pending_choice = (cands, text, context)
                        return RouteResult(
                            path=RoutePath.SEMANTIC,
                            speech=self._clarify_question(cands),
                            skill_name=None)

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

    @staticmethod
    def _actor_for(context: dict[str, Any] | None) -> Actor:
        """Resolve WHO is acting, for the policy engine's owner-voice gate (S2).

        The verified owner (the voice loop sets ``owner_verified`` after a
        speaker check) is OWNER; a token-authed LAN client (watch/phone/device)
        is LAN_CLIENT; anything else at the machine is LOCAL_USER. While
        ``security.owner_voice`` is off this never changes a decision — the actor
        only matters once high-impact actions are owner-gated."""
        ctx = context or {}
        if ctx.get("owner_verified"):
            return Actor.OWNER
        src = ctx.get("source")
        if src in ("remote", "watch", "phone", "lan", "device", "link"):
            return Actor.LAN_CLIENT
        return Actor.LOCAL_USER

    def _audit(self, event: str, **fields: Any) -> None:
        """Record a security-relevant event (no-op unless auditing is on)."""
        if self._audit_log is not None:
            self._audit_log.record(event, **fields)

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
        gate = self._policy.gate_skill(skill, actor=self._actor_for(request.context))
        if gate.denied():
            self._audit(POLICY_DENY, skill=skill.name, code=gate.code)
            speech = (PC_CONTROL_OFF_REPLY if gate.code == "pc_control_off"
                      else gate.reason)
            return RouteResult(path=path, speech=speech, skill_name=skill.name)
        outcome = await self._safe_execute(skill, request)
        if outcome.await_reply:
            # The skill asked a question; capture the NEXT utterance as its answer.
            self._pending_reply = (skill, request)
            self._reply_is_offer = outcome.reply_is_offer
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
            gate = self._policy.gate_skill(skill, actor=self._actor_for(context))
            if gate.denied():
                # PC control may have been switched off while this action waited
                # on a yes — re-apply the policy gate so a confirmed destructive
                # action can't slip past a now-off switch.
                speech = (PC_CONTROL_OFF_REPLY if gate.code == "pc_control_off"
                          else gate.reason)
                return RouteResult(path=RoutePath.FAST, speech=speech,
                                  skill_name=skill.name)
            request.context = {**request.context, "confirmed": True}
            self._audit(CONFIRM_GRANTED, skill=skill.name)
            outcome = await self._safe_execute(skill, request)  # performs the action
            return RouteResult(
                path=RoutePath.FAST,
                speech=outcome.speech,
                skill_name=skill.name,
                data=outcome.data,
            )
        if is_negative(text):
            self._pending = None
            self._audit(CONFIRM_DENIED, skill=skill.name)
            return RouteResult(path=RoutePath.FAST, speech=CANCELLED_REPLY)
        # Anything else: don't guess with a destructive action — cancel and re-route.
        self._pending = None
        self._rerouted_from_confirmation = True
        logger.info("ambiguous confirmation %r; cancelling pending action", text)
        return await self._route_inner(text, context)

    async def _resolve_choice(
        self, text: str, context: dict[str, Any]
    ) -> RouteResult:
        """Resolve a spoken answer to a near-tie clarify question.

        Dispatches one of the two offered skills on the SEMANTIC path with the
        ORIGINAL utterance (so a query skill parses the real question), then
        learns that phrasing. Never guesses: a no / "neither", a different
        fast-path command, or an undecidable reply cancels the choice and
        re-routes the utterance fresh — mirroring the confirmation gate.
        """
        cands, original_text, _orig_ctx = self._pending_choice  # type: ignore[misc]
        self._pending_choice = None
        from core.safety import _normalize

        norm = _normalize(text)
        if (is_negative(text) or norm in {"neither", "none", "nothing"}
                or self._registry.find_match(text) is not None):
            self._rerouted_from_choice = True
            logger.info("tie-break declined %r; re-routing", text)
            return await self._route_inner(text, context)
        chosen = await self._pick_choice(cands, text, norm)
        if chosen is None:                    # couldn't tell — don't guess
            self._rerouted_from_choice = True
            logger.info("tie-break ambiguous %r; re-routing", text)
            return await self._route_inner(text, context)
        res = await self._run_skill(
            chosen, SkillRequest(text=original_text, context=context),
            path=RoutePath.SEMANTIC)
        # Learn the ORIGINAL phrasing so this tie never recurs (a no-op unless
        # route_memory_enabled; _learn_route re-embeds since route() cleared the
        # per-turn vector at the top of this answering turn).
        await self._learn_route(original_text, chosen)
        # Record the resolved exchange under the ORIGINAL question (route()
        # suppresses the bare one-word answer) and arm the live-info follow-up
        # chain, so a tie-break behaves like a normal semantic route.
        self.conversation.add_turn(original_text, res.speech)
        self._last_live_info = (res.skill_name in LIVE_INFO_SKILLS
                                or self._turn_used_live_info)
        return res

    async def _pick_choice(self, cands: "list[Skill]", text: str, norm: str
                           ) -> "Skill | None":
        """Which offered skill the reply names: by whole-word match first, else
        a CONFIDENT embedding pick (floor + margin). None when it is genuinely
        undecidable, so the caller re-routes rather than guesses."""
        from core.agents import _STAR_LABELS, prettify

        words = set(norm.split())
        hits = []
        for s in cands:
            label = (_STAR_LABELS.get(s.name) or prettify(s.name)
                     or s.name.replace("_", " ")).lower()
            tokens = {s.name.lower(), *label.split()}
            # Whole-token match (not substring) so "app" doesn't hit "happy".
            if any(len(t) >= 3 and t in words for t in tokens):
                hits.append(s)
        if len(hits) == 1:
            return hits[0]
        # Fall back to meaning: embed the (short) reply once and pick the nearer
        # candidate index vector. Any failure -> None (caller re-routes).
        if self._embedder is None:
            return None
        idx = await self._ensure_route_index()
        if idx is None:
            return None
        try:
            rv = await asyncio.to_thread(self._embedder, [text])
        except Exception:
            return None
        if not rv:
            return None
        import numpy as np

        q = np.asarray(rv[0], dtype=np.float32).ravel()
        qn = float(np.linalg.norm(q))
        if not np.isfinite(qn) or qn == 0.0:
            return None
        q = q / qn
        vecs = {e.skill: e.vector for e in idx.entries()}
        scored: list[tuple[Skill, float]] = []
        for s in cands:
            v = vecs.get(s.name)
            if v is None:
                continue
            v = np.asarray(v, dtype=np.float32).ravel()
            vn = float(np.linalg.norm(v))
            if not np.isfinite(vn) or vn == 0.0 or v.shape != q.shape:
                continue
            score = float(q @ (v / vn))
            if np.isfinite(score):
                scored.append((s, score))
        if not scored:
            return None
        scored.sort(key=lambda x: x[1], reverse=True)
        best_skill, best = scored[0]
        runner = scored[1][1] if len(scored) > 1 else -1.0
        # A confident, DISTINGUISHING pick only: an absolute floor plus a margin
        # over the other candidate (mirrors match()/top_pair). Otherwise None, so
        # an "I don't know" / unrelated reply re-routes instead of being forced
        # into one of the two skills.
        if (best >= self._settings.router.semantic_threshold
                and (best - runner) >= self._settings.router.semantic_margin):
            return best_skill
        return None

    async def _safe_execute(self, skill: Skill, request: SkillRequest) -> SkillResult:
        """Run a skill's ``execute`` with a backstop for unexpected exceptions.

        Skills are expected to handle their own failures and return a spoken
        ``SkillResult``, and almost all do. But an unguarded bug (e.g. a bad
        ``float()`` on a model-supplied arg) would otherwise propagate out of
        ``route()`` — merely a silent turn in voice mode, but a HARD CRASH of
        the text REPL and the ``/ask`` endpoint, which have no catch-all. This
        turns any such escape into a graceful spoken error instead.
        """
        try:
            return await skill.execute(request)
        except Exception:
            logger.exception("skill %r raised; returning a spoken error", skill.name)
            return SkillResult(
                "Sorry — something went wrong running that.", success=False)

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

    async def _pick_brain(self, text: str) -> tuple[OllamaClient, str | None, bool]:
        """Choose (client, model, borrowed) for this turn.

        Borrow the local tool-brain only when the selected brain is a CLI agent
        (which can't use MEDO's tools) AND the query looks like it needs live
        info. Otherwise use the selected brain unchanged — so normal chat still
        goes to the model you picked, and MEDO 'returns' to it automatically.

        The availability probe is a blocking ~2s HTTP GET, so it runs in a thread
        (once — the result is cached) rather than freezing the event loop, which
        would stall the companion API and voice for every CLI-provider live query.
        """
        if (self._settings.llm.provider in CLI_PROVIDERS
                and self._wants_live_info(text)):
            tb = self._tool_brain_client()
            if tb is not None and (self._tool_brain_ok
                                   or await asyncio.to_thread(tb.is_available)):
                self._tool_brain_ok = True
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
        llm, model, borrowed = await self._pick_brain(text)
        if not model:
            return RouteResult(path=RoutePath.LLM, speech=self._offline_reply)

        # Cloud data-egress gate (S5): a CLOUD brain gets your question, not your
        # local documents/memory/files — unless you opted THAT brain in.
        from security.egress import (
            LOCAL_CONTENT_TOOLS, is_cloud, local_context_allowed)
        egress_ok = local_context_allowed(self._settings.security,
                                          self._settings.llm.provider)
        if is_cloud(self._settings.llm.provider):
            self._audit(CLOUD_CALL, brain=self._settings.llm.provider,
                        local_context=egress_ok)

        tools = build_tools(self._registry)
        # Don't even OFFER a tool the policy would refuse this turn (e.g. an
        # actuation tool while PC control is off) — the model answers around it
        # instead of calling and being refused. One authority, the policy engine.
        actor = self._actor_for(context)
        gated = {s.name for s in self._registry.all()
                 if self._policy.gate_skill(s, actor=actor).denied()}
        if not egress_ok:
            gated |= LOCAL_CONTENT_TOOLS      # no local-content tools to the cloud
        if gated:
            tools = [t for t in tools if t["function"]["name"] not in gated]
        try:
            facts = await asyncio.to_thread(
                self.facts.relevant, text, self._settings.memory.max_facts
            )
        except Exception:  # a broken facts DB must never take down the LLM path
            logger.exception("could not load remembered facts")
            facts = []
        if not egress_ok and facts:
            # Local memory must not ride along to a cloud brain unopted-in.
            logger.info("withholding %d local facts from cloud brain %s "
                        "(no egress opt-in)", len(facts), self._settings.llm.provider)
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
        actor = self._actor_for(context)
        # Trust boundary (S3): the turn is tainted if an untrusted-content tool
        # already ran, OR one is in THIS batch (proposed alongside the action).
        batch_has_content = any(
            (c.get("function") or {}).get("name", "") in UNTRUSTED_CONTENT_TOOLS
            for c in tool_calls)
        tainted = self._turn_untrusted_context or batch_has_content

        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments") or {}
            gated_skill = self._registry.get(name)
            # Coerce string-typed args now, so a stashed confirmation request
            # (re-executed on "yes") carries clean args too — not just dispatch.
            args = coerce_args(gated_skill, args)
            # An action induced by untrusted content this turn is judged with
            # UNTRUSTED provenance — the post-LLM injection gate.
            caps = (effective_capabilities(gated_skill)
                    if gated_skill is not None else frozenset())
            provenance = (Provenance.UNTRUSTED
                          if tainted and (caps & DANGEROUS_CAPS)
                          else Provenance.USER)
            gate = (self._policy.gate_skill(gated_skill, actor=actor,
                                            provenance=provenance)
                    if gated_skill is not None else None)
            if gate is not None and gate.code == "untrusted_confirm":
                # The model proposed a high-impact action that seems to come from
                # content it read, not from the user. Never silently execute —
                # require an explicit yes (or deny, per untrusted_action_policy).
                self._pending = (gated_skill,
                                 SkillRequest(text=text, args=args, context=context))
                return RouteResult(
                    path=RoutePath.LLM, skill_name=name,
                    speech=("That looks like it came from something I read, not "
                            "from you. Do you want me to go ahead? Say yes to "
                            "confirm."))
            if gate is not None and gate.denied():
                # Belt to the tool-filter's braces: even a hallucinated call to a
                # gated tool is refused (e.g. an actuation tool while PC control
                # is off, or an injected action when the policy is deny) — the
                # policy engine is the single authority.
                speech = (PC_CONTROL_OFF_REPLY if gate.code == "pc_control_off"
                          else gate.reason)
                result = SkillResult(speech, success=False)
            else:
                try:
                    result = await dispatch_tool(self._registry, name, args, context)
                except Exception:
                    # A crashing tool must not take down the turn (silent in
                    # voice, fatal in the text REPL / /ask). Speak an error.
                    logger.exception("tool %r raised; returning a spoken error", name)
                    result = SkillResult(
                        "Sorry — that ran into an error.", success=False)
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
            if name in UNTRUSTED_CONTENT_TOOLS:
                # Taint the turn (a later high-impact action is now gated) AND
                # wrap the content so the model treats it as quoted data.
                self._turn_untrusted_context = True
                content = mark_untrusted(result.speech, name)
            else:
                content = result.speech
            messages.append({"role": "tool", "name": name, "content": content})
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
