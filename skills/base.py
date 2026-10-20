"""The Skill contract and the registry that routes to skills.

Design decision (the whiteboard one): a capability is written **once** as a
:class:`Skill`. It exposes two entry points onto the same ``execute`` method:

* ``patterns`` — regex for the deterministic **fast path** (no LLM).
* ``tool_schema`` — a JSON function-calling schema for the **LLM path** (M3).

The registry is the single source of truth for both the router's keyword matching
and the list of tools handed to Ollama.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # no runtime import — security.capabilities stays a leaf module
    from security.capabilities import Capability


@dataclass
class SkillRequest:
    """Everything a skill needs to act on one request."""

    text: str                                   # the words to act on
    match: re.Match[str] | None = None          # set on the fast path
    args: dict[str, Any] = field(default_factory=dict)  # set on the LLM tool path
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def utterance(self) -> str:
        """What the USER actually said — on every route.

        On the fast and semantic paths ``text`` IS the utterance. On the LLM
        tool path it is NOT: ``dispatch_tool`` synthesizes a stand-in out of the
        tool name and the model's arguments ("smalltalk", "weather Skopje") so
        pattern-driven skills still find their groups. That stand-in is always
        written in English, which made every skill that picked its reply language
        with ``mk.is_cyrillic(request.text)`` answer a Macedonian question in
        English the moment the LLM tier — rather than a regex — dispatched it.
        Asked "Твоја бога.", MEDO sniffed the synthetic word "smalltalk", found
        no Cyrillic, and said "At your service."

        Use this for anything that reasons about how the user PHRASED the
        request — reply language above all; use ``text``/``args`` for the values
        to act on.

        A tool-path request with no utterance in its context returns "", NOT the
        stand-in: it is the model's own words, and a safety check that read it
        back would be confirming the model against itself. ``power`` asks
        whether the user said "lock" — and the stand-in for that very call is
        the string "power lock".
        """
        spoken = str(self.context.get("utterance") or "")
        if spoken:
            return spoken
        return "" if self.context.get("via") == "tool" else self.text


@dataclass
class SkillResult:
    """What a skill hands back: speech for TTS plus optional structured data."""

    speech: str
    success: bool = True
    data: dict[str, Any] = field(default_factory=dict)
    # True when the skill wants a spoken confirmation before it actually runs
    # (wired through core/safety.py in M2).
    needs_confirmation: bool = False
    # True when the skill asked a question and wants to CAPTURE the user's next
    # utterance as free-text input (e.g. self-dev's "what should I fix?"). The
    # router routes that reply back to this skill with context['captured_reply'],
    # unless it matches another command — see Router._route_inner.
    await_reply: bool = False
    # Set ALONGSIDE await_reply when the captured reply is a yes/no OFFER (e.g.
    # "want the full breakdown?"). The router then only feeds an affirmative or a
    # negative back to this skill; any OTHER utterance is treated as a fresh
    # command and re-routed, never swallowed into the offer as a bland "okay".
    reply_is_offer: bool = False
    # Set ALONGSIDE await_reply when the question is OPEN — the skill asked for a
    # free-text description and ANY answer is a valid one ("What should the app
    # do?"). The router's default is to abandon a capture whose reply happens to
    # match another skill, which is right for a narrow question but wrong here:
    # a spec naturally mentions weather, timers, or files, and the matching skill
    # would swallow the answer ("I want the app to track the weather in
    # Macedonia" -> the weather skill reads out a temperature and the build is
    # silently lost). An open capture keeps its reply; the skill handles "never
    # mind" itself.
    reply_is_open: bool = False


class Skill(ABC):
    """Base class for every capability.

    Subclasses set the class attributes below and implement ``execute``;
    override ``tool_schema`` only when the tool takes arguments.
    """

    #: Stable identifier, also the LLM tool name. e.g. ``"datetime"``.
    name: str = "skill"
    #: One-line human description (used in the tool schema and help output).
    description: str = ""
    #: Compiled regex tried in order on the fast path. Empty => LLM-only skill.
    patterns: list[re.Pattern[str]] = []
    #: Destructive/system actions set this so core/safety.py gates them (M2).
    requires_confirmation: bool = False
    #: True for skills that ACT on this computer (type, click, launch, open
    #: sites, power, files...). The HUD's PC CONTROL switch
    #: (safety.pc_control_enabled) blocks them all at one router choke point;
    #: purely sensing/answering skills stay available either way.
    controls_pc: bool = False
    #: The security capabilities this skill needs (security/capabilities.py). The
    #: policy engine grants only declared + permitted ones; read-only sensing
    #: declares nothing. ADDITIVE and backward-compatible: ``controls_pc`` still
    #: works and is the derived truth (any actuation capability => controls_pc),
    #: so un-migrated skills keep their coarse gating via ``effective_capabilities``.
    capabilities: "frozenset[Capability]" = frozenset()
    #: Example utterances that should route here — the skill's SEMANTIC surface
    #: for the Tier-2 router (M2.5). Plain phrases a person says, NOT regex, and
    #: NOT synonyms of the fast-path patterns: those stay in ``patterns``. This
    #: is additive and data-only — the deterministic fast path never reads it,
    #: so populating it can never change what the fast path routes. Empty is
    #: fine; the router then falls back to ``description``.
    routing_phrases: list[str] = []
    #: Opt-in for the SEMANTIC tier when the skill needs an argument. Reached by
    #: MEANING, a skill gets a bare ``SkillRequest`` — no regex ``match``, no
    #: ``args`` — so one that reads a query from the match would deflect ("What
    #: should I search for?"). Setting this True is a promise that ``execute``
    #: falls back to deriving what it needs from ``request.text``; the router
    #: then treats the skill as safe to index even though its tool schema marks
    #: a parameter required. Query skills that need NO argument leave this False
    #: and are indexed automatically.
    semantic_from_text: bool = False

    def routing_surface(self) -> list[str]:
        """The phrases the semantic router embeds to represent this skill.

        ``routing_phrases`` first (curated examples), then the human
        ``description``, then the skill name as a last resort so every skill
        has *some* surface. Purely additive; the fast path never calls this.
        """
        parts = [p for p in self.routing_phrases if p and p.strip()]
        if self.description:
            parts.append(self.description)
        if not parts:
            parts.append(self.name.replace("_", " "))
        return parts

    def routing_text(self) -> str:
        """The single string embedded + cached for this skill (see M2.5a)."""
        return " · ".join(self.routing_surface())

    # -- the confirmation contract (see core/router.py::_preconfirm) ----------
    #
    # A skill that declares ``requires_confirmation`` is never executed by the
    # router until the user has said yes. That guarantee used to live inside
    # each skill ("if not context['confirmed']: return needs_confirmation"),
    # which meant one forgotten line — in a new skill, a plugin, a connector
    # manifest, a device capability — was a destructive action that ran
    # silently. The two hooks below move the guarantee into the router while
    # keeping what the per-skill version was good at:
    #
    #   * ``preflight`` runs FIRST, so "that feature is switched off" or "what
    #     should I do on the page?" still comes back instead of an absurd
    #     "shall I go ahead?" about nothing.
    #   * ``confirmation_prompt`` supplies the actual question, so it can name
    #     the file, the app, or the task rather than the skill.
    #
    # Skills keep their internal ``confirmed`` check as well. It is now
    # unreachable through the router, and that is the point: it is what still
    # protects a direct ``execute`` call from a test or a future caller.

    async def preflight(self, request: SkillRequest) -> "SkillResult | None":
        """Checks that must happen BEFORE the user is asked to confirm.

        Return a :class:`SkillResult` to answer now and never ask (a disabled
        feature, a missing argument); return ``None`` to go ahead and confirm.
        Only consulted for skills that declare ``requires_confirmation``.
        """
        return None

    def confirmation_prompt(self, request: SkillRequest) -> str:
        """The question asked before this skill runs. Override to be specific."""
        return (f"Do you want me to {self.name.replace('_', ' ')}? "
                f"Say yes to confirm.")

    def match(self, text: str) -> re.Match[str] | None:
        """Return the first pattern that matches ``text``, or ``None``."""
        for pattern in self.patterns:
            found = pattern.search(text)
            if found is not None:
                return found
        return None

    @abstractmethod
    async def execute(self, request: SkillRequest) -> SkillResult:
        """Perform the action and return speech + data."""

    def tool_schema(self) -> dict[str, Any]:
        """Ollama/OpenAI-style function schema for the LLM path.

        Default: an argument-less tool built from ``name``/``description`` —
        enough for most plugins, so simple skills need no boilerplate.
        Override to declare parameters.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}},
            },
        }


class SkillRegistry:
    """Holds all skills; used by the router (fast path) and LLM client (tools)."""

    def __init__(self) -> None:
        self._skills: list[Skill] = []

    def register(self, skill: Skill) -> Skill:
        if any(s.name == skill.name for s in self._skills):
            raise ValueError(f"duplicate skill name: {skill.name!r}")
        self._skills.append(skill)
        return skill

    def all(self) -> list[Skill]:
        return list(self._skills)

    def names(self) -> list[str]:
        """Every registered tool name — what a reply must never say out loud."""
        return [s.name for s in self._skills]

    def unregister(self, name: str) -> bool:
        """Remove a skill by name (device re-registration). False if absent."""
        for i, skill in enumerate(self._skills):
            if skill.name == name:
                del self._skills[i]
                return True
        return False

    def get(self, name: str) -> Skill | None:
        return next((s for s in self._skills if s.name == name), None)

    def find_match(self, text: str) -> tuple[Skill, re.Match[str]] | None:
        """First skill whose pattern matches ``text`` (fast-path lookup)."""
        for skill in self._skills:
            found = skill.match(text)
            if found is not None:
                return skill, found
        return None

    def tool_schemas(self) -> list[dict[str, Any]]:
        """All skills' function schemas, for handing to the LLM (M3)."""
        return [s.tool_schema() for s in self._skills]
