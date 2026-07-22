"""Ask a specialist, or convene several — MEDO stays the architect.

Two skills over :mod:`core.council`:

* :class:`AskSpecialistSkill` — "ask the electrical engineer about grounding".
  One expert, one answer, named out loud so you know who spoke.
* :class:`ConveneCouncilSkill` — "convene the council on X". MEDO picks the
  two or three specialists whose field the question actually touches, asks
  them in parallel, and then synthesizes one spoken answer, naming who
  contributed. That synthesis step is the architect role: the council does not
  talk to the user, it reports to MEDO.

Neither listens for the wake word and neither holds a conversation — they are
called, they work, they return. Everything runs on whatever brain is already
loaded, so on Ollama this costs no extra processes at all.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from core import mk
from core.config import Settings
from core.council import (
    Specialist,
    enabled_council,
    find_specialist,
    load_council,
    rank_specialists,
    system_prompt,
)
from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

NO_BRAIN = "My brain is offline, so the council can't meet."
NO_BRAIN_MK = "Мозокот е офлајн, па советот не може да работи."


class _CouncilBase(Skill):
    """Shared plumbing: the roster, the LLM hop, and the language rule."""

    controls_pc = False

    def __init__(self, settings: Settings, ask=None) -> None:
        self._settings = settings
        self._ask = ask          # async (system: str, user: str) -> str
        # The full roster is fixed at startup (it comes from config.extra), but
        # WHICH members are on is read per call — the HUD toggles
        # council.disabled at runtime and the next question must see it.
        self._roster = load_council(settings.council.extra)

    @property
    def _council(self) -> tuple[Specialist, ...]:
        return enabled_council(self._roster, self._settings.council.disabled)

    async def _consult(self, member: Specialist, question: str,
                       speak_mk: bool) -> tuple[Specialist, str]:
        """Ask one specialist. Never raises — a dead expert must not kill the
        whole council, so a failure comes back as an empty answer."""
        try:
            answer = await self._ask(system_prompt(member, speak_mk), question)
        except Exception:
            logger.warning("specialist %s failed", member.key, exc_info=True)
            return member, ""
        return member, (answer or "").strip()


class AskSpecialistSkill(_CouncilBase):
    """Route one question to one named expert."""

    name = "ask_specialist"
    description = (
        "Put a question to one specialist on MEDO's council (electrical "
        "engineer, roboticist, physicist, mathematician, lawyer, financial "
        "analyst, economist and others). Use when the user names a field or "
        "an expert."
    )

    patterns = [
        re.compile(r"\bask\s+(?:the\s+)?(?P<who>[\w\s]+?)\s+"
                   r"(?:about|regarding|what|how|why|whether|if)\s+(?P<q>.+)$",
                   re.IGNORECASE),
        re.compile(r"\b(?:what\s+(?:would|does)|ask)\s+(?:the\s+)?(?P<who2>[\w\s]+?)\s+"
                   r"say\s+about\s+(?P<q2>.+)$", re.IGNORECASE),
        # MK: "прашај го електроинженерот за заземјување"
        re.compile(r"\bпрашај\s+(?:го\s+|ја\s+)?(?P<who3>[\w\s]+?)\s+за\s+(?P<q3>.+)$",
                   re.IGNORECASE),
    ]

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        speak_mk = mk.is_cyrillic(request.text)
        who = (request.args.get("specialist") or gd.get("who") or gd.get("who2")
               or gd.get("who3") or "").strip()
        question = (request.args.get("question") or gd.get("q") or gd.get("q2")
                    or gd.get("q3") or "").strip(" ?.!")
        member = find_specialist(who, self._council)
        if member is None:
            names = ", ".join(s.title.removeprefix("the ") for s in self._council[:6])
            return SkillResult(
                f"Немам таков специјалист. Имам: {names}." if speak_mk
                else f"I don't have that specialist. I have: {names}.",
                success=False)
        if not question:
            return SkillResult("Што да го прашам?" if speak_mk
                               else "What should I ask them?", success=False)
        if self._ask is None:
            return SkillResult(NO_BRAIN_MK if speak_mk else NO_BRAIN, success=False)

        _member, answer = await self._consult(member, question, speak_mk)
        if not answer:
            return SkillResult(
                f"{member.title.capitalize()} не одговори." if speak_mk
                else f"I couldn't get an answer from {member.title}.",
                success=False)
        lead = (f"Според {member.title}: " if speak_mk
                else f"From {member.title}: ")
        return SkillResult(lead + answer,
                           data={"specialist": member.key, "question": question})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "specialist": {"type": "string",
                                       "enum": [s.key for s in self._council]},
                        "question": {"type": "string"},
                    },
                    "required": ["specialist", "question"],
                },
            },
        }


class ConveneCouncilSkill(_CouncilBase):
    """Ask the two or three relevant experts, then synthesize."""

    name = "convene_council"
    description = (
        "Put a hard, cross-disciplinary question to several of MEDO's "
        "specialists at once and give one combined answer. Use for design "
        "questions that span fields."
    )

    patterns = [
        re.compile(r"\b(?:convene|assemble|gather)\s+(?:the\s+)?council\b"
                   r"(?:\s+(?:on|about|for)\s+(?P<q>.+))?$", re.IGNORECASE),
        re.compile(r"\bask\s+(?:the\s+)?council\s+(?:about\s+|on\s+)?(?P<q2>.+)$",
                   re.IGNORECASE),
        re.compile(r"\bwhat\s+does\s+the\s+council\s+(?:think|say)\s+"
                   r"(?:about\s+)?(?P<q3>.+)$", re.IGNORECASE),
        # MK: "свикај го советот за …"
        re.compile(r"\bсвикај\s+(?:го\s+)?советот\s*(?:за\s+)?(?P<q4>.+)?$",
                   re.IGNORECASE),
    ]

    def __init__(self, settings: Settings, ask=None, synthesize=None) -> None:
        super().__init__(settings, ask)
        self._synthesize = synthesize     # async (question, notes) -> str

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        speak_mk = mk.is_cyrillic(request.text)
        question = (request.args.get("question") or gd.get("q") or gd.get("q2")
                    or gd.get("q3") or gd.get("q4") or "").strip(" ?.!")
        if not question:
            return SkillResult("За што да го свикам советот?" if speak_mk
                               else "What should I put to the council?",
                               success=False)
        if self._ask is None:
            return SkillResult(NO_BRAIN_MK if speak_mk else NO_BRAIN, success=False)

        picked = rank_specialists(question, self._council,
                                  self._settings.council.max_members)
        if not picked:
            # Nothing matched a field — fall back to the configured generalist
            # rather than answering as a committee of nobody.
            fallback = find_specialist(self._settings.council.default_agent,
                                       self._council)
            picked = [fallback] if fallback else list(self._council[:1])
        if not picked:
            return SkillResult("Советот е празен." if speak_mk
                               else "The council is empty.", success=False)

        # Specialists don't confer — they report independently, which is the
        # point: three framings of the same problem, not one echoed three times.
        results = await asyncio.gather(
            *(self._consult(m, question, speak_mk) for m in picked))
        notes = [(m, a) for m, a in results if a]
        if not notes:
            return SkillResult("Никој не одговори." if speak_mk
                               else "Nobody on the council answered.",
                               success=False)

        names = ", ".join(m.title.removeprefix("the ") for m, _ in notes)
        if self._synthesize is None or len(notes) == 1:
            member, answer = notes[0]
            lead = (f"Според {member.title}: " if speak_mk
                    else f"From {member.title}: ")
            return SkillResult(lead + answer, data={"members": [m.key for m, _ in notes]})

        block = "\n\n".join(f"{m.title.upper()}:\n{a}" for m, a in notes)
        try:
            combined = await self._synthesize(question, block, speak_mk)
        except Exception:
            logger.warning("council synthesis failed", exc_info=True)
            combined = ""
        if not combined:
            combined = " ".join(a for _m, a in notes)
        prefix = (f"Го прашав {names}. " if speak_mk
                  else f"I asked {names}. ")
        return SkillResult(prefix + combined,
                           data={"members": [m.key for m, _ in notes],
                                 "question": question})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
            },
        }
