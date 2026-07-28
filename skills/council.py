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
    # Reached by MEANING: the utterance names the expert, so execute() recovers
    # them (and the question) from request.text. Phrases kept singular/named to
    # separate this from the collective convene_council.
    semantic_from_text = True
    routing_phrases = [
        "get the physicist's take on this",
        "I want the lawyer's opinion on this contract",
        "run this by the electrical engineer",
        "check with the roboticist about the gait",
        "have the economist look at these numbers",
        "let the mechanical engineer weigh in on this bracket",
        "put this question to the software engineer",
        "what would the mathematician make of this proof",
    ]

    patterns = [
        re.compile(r"\bask\s+(?:the\s+)?(?P<who>[\w\s]+?)\s+"
                   r"(?:about|regarding|what|how|why|whether|if)\s+(?P<q>.+)$",
                   re.IGNORECASE),
        re.compile(r"\b(?:what\s+(?:would|does)|ask)\s+(?:the\s+)?(?P<who2>[\w\s]+?)\s+"
                   r"say\s+about\s+(?P<q2>.+)$", re.IGNORECASE),
        # Natural ways to address one expert — "get X's take on", "what would X
        # recommend for", "check with X about", "have X weigh in on", "run this
        # by X". All still funnel through match() so a non-specialist declines.
        re.compile(r"\bget\s+(?:the\s+)?(?P<who4>[\w\s]+?)'?s\s+"
                   r"(?:take|opinion|view|read|thoughts?)\s+on\s+(?P<q4>.+)$",
                   re.IGNORECASE),
        re.compile(r"\bwhat\s+(?:would|does|will|do)\s+(?:the\s+)?(?P<who5>[\w\s]+?)\s+"
                   r"(?:think|recommend|suggest|advise)\s+"
                   r"(?:about|for|on|regarding)\s+(?P<q5>.+)$", re.IGNORECASE),
        re.compile(r"\bcheck\s+with\s+(?:the\s+)?(?P<who6>[\w\s]+?)\s+"
                   r"(?:about|on|regarding)\s+(?P<q6>.+)$", re.IGNORECASE),
        re.compile(r"\bhave\s+(?:the\s+)?(?P<who7>[\w\s]+?)\s+(?:weigh\s+in|look)\s+"
                   r"(?:on|at)\s+(?P<q7>.+)$", re.IGNORECASE),
        re.compile(r"\b(?:run|put)\s+(?P<q8>.+?)\s+(?:by|past)\s+(?:the\s+)?"
                   r"(?P<who8>[\w\s]+?)$", re.IGNORECASE),
        # MK: "прашај го електроинженерот за заземјување"
        re.compile(r"\bпрашај\s+(?:го\s+|ја\s+)?(?P<who3>[\w\s]+?)\s+за\s+(?P<q3>.+)$",
                   re.IGNORECASE),
    ]

    def match(self, text: str):
        """Only claim the utterance when the name captured IS a specialist.

        The patterns have to capture a free-form name — you address an expert
        by title, not by a fixed keyword — which made them greedy: "what does
        this page say about batteries" captured "this page" as the expert, and
        the fast path stopped there to answer "I don't have that specialist"
        instead of letting web_fetch read the page.

        So resolution happens at match time, not execute time. A name the
        council doesn't have is not a match at all, and the router carries on
        to the skills registered after this one.
        """
        found = super().match(text)
        if found is None:
            return None
        gd = found.groupdict()
        who = (gd.get("who") or gd.get("who2") or gd.get("who3") or gd.get("who4")
               or gd.get("who5") or gd.get("who6") or gd.get("who7")
               or gd.get("who8") or "").strip()
        return found if find_specialist(who, self._council) is not None else None

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        speak_mk = mk.is_cyrillic(request.text)
        who = (request.args.get("specialist") or gd.get("who") or gd.get("who2")
               or gd.get("who3") or gd.get("who4") or gd.get("who5")
               or gd.get("who6") or gd.get("who7") or gd.get("who8") or "").strip()
        question = (request.args.get("question") or gd.get("q") or gd.get("q2")
                    or gd.get("q3") or gd.get("q4") or gd.get("q5") or gd.get("q6")
                    or gd.get("q7") or gd.get("q8") or "").strip(" ?.!")
        member = find_specialist(who, self._council)
        if member is None and not request.match and not request.args:
            # Reached by MEANING (semantic tier): no regex groups, no tool args.
            # The utterance itself names the expert — recover them from it — and
            # the whole utterance stands in as the question.
            member = find_specialist(request.text, self._council)
            if member is not None and not question:
                question = request.text.strip(" ?.!")
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
    # Reached by MEANING: the whole utterance is the question to put to the
    # panel. Phrases kept plural/collective to separate this from ask_specialist.
    semantic_from_text = True
    routing_phrases = [
        "get the whole team to weigh in on this design",
        "I want everyone's opinion on whether to use a brushless motor",
        "bring your experts together on this problem",
        "get a panel of specialists to look at this trade-off",
        "pull the specialists together for this decision",
        "get a round-table on this design question",
        "have the panel look at this cross-discipline problem",
        "what would a group of experts say about this build",
    ]

    patterns = [
        re.compile(r"\b(?:convene|assemble|gather)\s+(?:the\s+)?council\b"
                   r"(?:\s+(?:on|about|for)\s+(?P<q>.+))?$", re.IGNORECASE),
        re.compile(r"\bask\s+(?:the\s+)?council\s+(?:about\s+|on\s+)?(?P<q2>.+)$",
                   re.IGNORECASE),
        re.compile(r"\bwhat\s+does\s+the\s+council\s+(?:think|say)\s+"
                   r"(?:about\s+)?(?P<q3>.+)$", re.IGNORECASE),
        # The synonyms the description promises — but "experts / specialists /
        # round-table" only, and "panel / team" ONLY when qualified ("panel of
        # experts"). Bare "assemble the team" / "gather the panel" is everyday
        # talk, not MEDO's council. The broader collective phrasings still reach
        # the council by MEANING via routing_phrases.
        re.compile(r"\b(?:convene|assemble|gather|bring\s+in|pull\s+together)\s+"
                   r"(?:the\s+|your\s+|a\s+)?"
                   r"(?:experts|specialists|round[\s-]?table|"
                   r"(?:panel|team)\s+of\s+(?:experts|specialists))\b"
                   r"(?:\s+(?:on|about|for)\s+(?P<q5>.+))?$", re.IGNORECASE),
        # "what do YOUR experts think about X" — the possessive ties it to
        # MEDO's council, so generic "what do the experts think about climate
        # change" (external authority) falls through to the web/LLM.
        re.compile(r"\bwhat\s+do\s+your\s+(?:experts|specialists)\s+"
                   r"(?:think|say)\s+(?:about\s+)?(?P<q6>.+)$", re.IGNORECASE),
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
                    or gd.get("q3") or gd.get("q4") or gd.get("q5") or gd.get("q6")
                    or gd.get("q7") or "").strip(" ?.!")
        if not question and not request.match and not request.args:
            # Reached by MEANING (semantic tier): the whole utterance is the ask.
            question = request.text.strip(" ?.!")
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


def _resolves_exactly(spoken: str, council: tuple[Specialist, ...]) -> Specialist | None:
    """Like find_specialist but WITHOUT the loose trailing-noun containment, so
    'do you have a legal pad' / 'an electrical outlet' / 'a data analyst' don't
    resolve to the lawyer / electrical engineer / financial analyst via a
    substring alias."""
    want = re.sub(r"\s+", " ", spoken.strip().strip(".,!?")).lower()
    want = re.sub(r"^(?:the|a|an|our|my)\s+", "", want)
    if not want:
        return None
    for member in council:
        title = member.title.lower().removeprefix("the ")
        if want in (member.key, title) or want.rstrip("s") == member.key:
            return member
        if any(want == a.lower() for a in member.aliases):
            return member
    return None


class CouncilRosterSkill(_CouncilBase):
    """Discovery: "who's on the council?" / "do you have a lawyer?"

    The council only works if you know who's in it. This lists the enabled
    specialists (read from config each call, like the other council skills) and
    answers a yes/no "do you have a <field>?". No LLM — it's a registry read.
    """

    name = "list_council"
    description = (
        "List the specialists on MEDO's council, or say whether a particular "
        "expert (lawyer, electrical engineer, physicist...) is available."
    )
    routing_phrases = [
        "who's on your council", "which experts do you have",
        "what specialists can I ask", "who can I consult on this",
        "list the experts you have", "what kind of expert do you have",
    ]

    patterns = [
        re.compile(r"\bwho(?:'?s| is| are)\s+(?:on\s+)?"
                   r"(?:the\s+council|your\s+(?:experts|specialists|panel|council))\b",
                   re.IGNORECASE),
        re.compile(r"\b(?:list|name)\s+(?:your\s+|the\s+)?"
                   r"(?:experts|specialists|council(?:\s+members)?)\b", re.IGNORECASE),
        re.compile(r"\bwhich\s+(?:experts|specialists)\s+"
                   r"(?:do\s+you\s+have|are\s+(?:there|available))\b", re.IGNORECASE),
        # "do you have a lawyer" — gated in match() so "do you have a minute"
        # (not a specialist) falls through.
        re.compile(r"\bdo\s+you\s+have\s+(?:an?\s+)?(?P<who>[\w\s]+?)"
                   r"(?:\s+on\s+(?:the\s+)?council)?\s*[?.!]*$", re.IGNORECASE),
        # MK: "кој е во советот", "кои специјалисти ги имаш"
        re.compile(r"\bкој\s+е\s+во\s+советот\b|\bкои\s+специјалисти\b",
                   re.IGNORECASE),
    ]

    def match(self, text: str):
        found = super().match(text)
        if found is None:
            return None
        who = (found.groupdict().get("who") or "").strip()
        # The broad "do you have a X" only claims when X resolves EXACTLY to a
        # specialist (not via substring, so 'a legal pad' / 'an electrical
        # outlet' fall through); the roster patterns (no 'who' group) always claim.
        if who and _resolves_exactly(who, self._council) is None:
            return None
        return found

    async def execute(self, request: SkillRequest) -> SkillResult:
        speak_mk = mk.is_cyrillic(request.text)
        gd = request.match.groupdict() if request.match else {}
        who = (request.args.get("specialist") or gd.get("who") or "").strip()
        members = self._council
        if not members:
            return SkillResult("Советот е празен." if speak_mk
                               else "The council is empty.", success=False)
        names = ", ".join(m.title.removeprefix("the ") for m in members)

        if who:
            member = find_specialist(who, members)
            if member is not None:
                return SkillResult(
                    f"Да, го имам {member.title} на советот." if speak_mk
                    else f"Yes — I have {member.title} on the council.",
                    data={"specialist": member.key})
            return SkillResult(
                f"Немам таков специјалист. Имам: {names}." if speak_mk
                else f"I don't have that one. I have: {names}.", success=False)

        return SkillResult(
            f"На советот се: {names}." if speak_mk
            else f"On the council I have: {names}.",
            data={"members": [m.key for m in members]})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "specialist": {
                            "type": "string",
                            "description": "Optional: check for one field/expert.",
                        }
                    },
                    "required": [],
                },
            },
        }
