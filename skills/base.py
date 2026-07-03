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
from typing import Any


@dataclass
class SkillRequest:
    """Everything a skill needs to act on one request."""

    text: str                                   # raw user utterance
    match: re.Match[str] | None = None          # set on the fast path
    args: dict[str, Any] = field(default_factory=dict)  # set on the LLM tool path
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult:
    """What a skill hands back: speech for TTS plus optional structured data."""

    speech: str
    success: bool = True
    data: dict[str, Any] = field(default_factory=dict)
    # True when the skill wants a spoken confirmation before it actually runs
    # (wired through core/safety.py in M2).
    needs_confirmation: bool = False


class Skill(ABC):
    """Base class for every capability.

    Subclasses set the class attributes below and implement ``execute`` and
    ``tool_schema``.
    """

    #: Stable identifier, also the LLM tool name. e.g. ``"datetime"``.
    name: str = "skill"
    #: One-line human description (used in the tool schema and help output).
    description: str = ""
    #: Compiled regex tried in order on the fast path. Empty => LLM-only skill.
    patterns: list[re.Pattern[str]] = []
    #: Destructive/system actions set this so core/safety.py gates them (M2).
    requires_confirmation: bool = False

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

    @abstractmethod
    def tool_schema(self) -> dict[str, Any]:
        """Ollama/OpenAI-style function schema for the LLM path (used in M3)."""


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
