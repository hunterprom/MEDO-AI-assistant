"""Wiring help — how to connect it, and the code to drive it.

"I'm building a toy car: battery, Arduino, an LED — how do I wire it and can
you write the code?" That question needs two different answers in one breath:
a **connection list** (what joins to what, and which pin) and **firmware**.
Asking a general model gets prose about electronics; asking it with an
electrical-engineer framing and an explicit output contract gets something you
can actually build from.

The code is written to a file when the user asks for it, so it can go straight
into the Arduino IDE instead of being read aloud pin by pin — a sketch is the
one answer nobody wants spoken.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from core import mk
from core.config import Settings, expand_path
from core.council import HOUSE_STYLE, find_specialist, load_council
from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

#: The output contract. Without the explicit sections a model answers in an
#: essay, and you cannot wire a board from an essay.
WIRING_PROMPT = (
    "The user is building something and needs to know how to wire it.\n"
    "Answer in exactly these sections, in plain text, no markdown:\n"
    "CONNECTIONS: one line per connection, in the form "
    "'<component pin> -> <component pin>'. Name real pins (GND, 5V, D9, "
    "VIN...). Include every ground. State resistor values with units.\n"
    "WATCH OUT: one or two lines on what would damage the parts here — "
    "reversed polarity, missing current-limiting resistor, drawing motor "
    "current through the microcontroller, shared-ground mistakes.\n"
    "CODE: a complete, compilable Arduino sketch, if code makes sense for "
    "this. Otherwise write CODE: none.\n"
    "Be specific. If a critical detail is missing (supply voltage, part "
    "number), state the assumption you are making rather than asking."
)

#: Spoken summary rules — the connection list is for the eyes, not the ears.
SPOKEN_PROMPT = (
    "Summarize this wiring answer in two or three spoken sentences: the shape "
    "of the circuit and the single most important thing not to get wrong. "
    "Plain text, no markdown, no pin-by-pin list."
)


def split_sections(answer: str) -> dict[str, str]:
    """Split the model's reply into CONNECTIONS / WATCH OUT / CODE. Pure.

    Tolerant on purpose: a model that drops a heading or renames it slightly
    should degrade to "everything is prose", not to an exception.
    """
    sections: dict[str, str] = {}
    pattern = re.compile(r"^\s*(CONNECTIONS|WATCH\s*OUT|CODE)\s*:",
                         re.IGNORECASE | re.MULTILINE)
    marks = list(pattern.finditer(answer or ""))
    for index, mark in enumerate(marks):
        key = re.sub(r"\s+", "", mark.group(1)).lower()
        end = marks[index + 1].start() if index + 1 < len(marks) else len(answer)
        sections[key] = answer[mark.end():end].strip()
    return sections


def extract_code(section: str) -> str:
    """The sketch itself, with any code fence stripped. Empty when there's none."""
    text = (section or "").strip()
    if not text or text.lower().startswith("none"):
        return ""
    fenced = re.search(r"```(?:\w+)?\s*\n(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip("`").strip()


class CircuitSkill(Skill):
    """Explain how to wire something, and write the sketch for it."""

    name = "circuit_help"
    controls_pc = False          # writing the sketch is opt-in, inside the whitelist
    description = (
        "Explain how to wire electronic components together (microcontroller, "
        "battery, LED, motor, sensor) and write the Arduino sketch to drive "
        "them. Use for 'how do I connect/wire X' build questions."
    )
    # Reached by MEANING: the whole utterance is the build question, so wiring
    # questions that don't hit the verb templates still get the electrical-
    # engineer framing instead of vague electronics prose from the general LLM.
    semantic_from_text = True
    routing_phrases = [
        "how should I hook up a battery, an arduino and an LED",
        "what pins do I connect this servo motor to",
        "I'm building a little robot car, how do I wire the motors",
        "help me build the circuit for a temperature sensor",
        "wiring diagram for an ultrasonic distance sensor",
        "write the arduino sketch to blink an LED",
        "how do you connect an OLED display to an ESP32",
        "what resistor do I need for this LED",
    ]

    patterns = [
        # Subject is i|you|we ("how do YOU wire up an LED"), and the same
        # networking-sense decline pattern 2 uses — otherwise "how do I connect
        # my phone to the wifi" was answered as an electronics build.
        re.compile(r"\bhow\s+(?:do|would|can|could|should)\s+(?:i|you|we)\s+"
                   r"(?:wire|connect|hook\s+up)\s+"
                   r"(?!.*\b(?:wi-?fi|internet|network|bluetooth|phone|laptop|"
                   r"printer|projector|monitor|tv|account|server|vpn|router|"
                   # social / comms / idiom senses of "connect" are not circuits
                   r"people|someone|somebody|anyone|anybody|others|users?|"
                   r"customers?|clients?|colleagues?|humans?|agent|support|"
                   r"sales|emotionally|dots|deeper|"
                   # everyday plumbing / travel / phone-call / towing senses
                   r"flights?|hoses?|pipes?|trailers?|water|sprinklers?|faucets?|"
                   r"tap|dishwashers?|reception|call|mother|father)\b)"
                   r"(?P<q>.+)$", re.IGNORECASE),
        # "connect X to Y" is overloaded — decline the everyday networking sense
        # ("connect my phone to the wifi") so it isn't answered as a circuit.
        re.compile(r"\b(?:wire|wiring|connect)\s+(?:up\s+)?"
                   r"(?!.*\b(?:wi-?fi|internet|network|bluetooth|phone|laptop|"
                   r"printer|projector|monitor|tv|account|server|vpn|router|"
                   # social / comms / idiom senses of "connect" are not circuits
                   r"people|someone|somebody|anyone|anybody|others|users?|"
                   r"customers?|clients?|colleagues?|humans?|agent|support|"
                   r"sales|emotionally|dots|deeper|"
                   # everyday plumbing / travel / phone-call / towing senses
                   r"flights?|hoses?|pipes?|trailers?|water|sprinklers?|faucets?|"
                   r"tap|dishwashers?|reception|call|mother|father)\b)"
                   r"(?P<q2>.+?)\s+(?:to|with|and)\s+(?P<q2b>.+)$", re.IGNORECASE),
        re.compile(r"\bhelp\s+me\s+(?:wire|build)\s+(?P<q3>.+)$", re.IGNORECASE),
        # "wiring DIAGRAM for X", "circuit SCHEMATIC for X" — allow the noun.
        # Decline the fitness "circuit training" sense ("circuit for my abs").
        re.compile(r"\b(?:circuit|schematic|wiring)(?:\s+(?:diagram|layout|schematic))?"
                   r"\s+for\s+(?!(?:my\s+|the\s+|your\s+)?"
                   r"(?:abs|workout|gym|legs?|arms?|chest|cardio|reps?|training|"
                   r"exercises?|fitness)\b)(?P<q4>.+)$", re.IGNORECASE),
        # MK: "како да поврзам ардуино со диода" (спојам with CYRILLIC ј U+0458,
        # not a Latin j — the old spelling never matched real Cyrillic input).
        re.compile(r"\bкако\s+да\s+(?:го\s+|ја\s+)?(?:поврзам|врзам|спојам)\s+(?P<qm>.+)$",
                   re.IGNORECASE),
        re.compile(r"\bшема\s+за\s+(?P<qm2>.+)$", re.IGNORECASE),
    ]

    def __init__(self, settings: Settings, ask=None, summarize=None) -> None:
        self._settings = settings
        self._ask = ask                # async (system, user) -> str
        self._summarize = summarize    # async (system, user) -> str

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        speak_mk = mk.is_cyrillic(request.text)
        question = (request.args.get("build") or gd.get("q") or gd.get("q3")
                    or gd.get("q4") or gd.get("qm") or gd.get("qm2") or "")
        if not question and gd.get("q2"):
            question = f"{gd['q2']} to {gd.get('q2b', '')}"
        question = question.strip(" ?.!")
        if not question and not request.match and not request.args:
            # Reached by MEANING (semantic tier): the utterance IS the question.
            question = request.text.strip(" ?.!")
        if not question:
            return SkillResult("Што сакаш да поврзеш?" if speak_mk
                               else "What are you trying to wire up?",
                               success=False)
        if self._ask is None:
            return SkillResult(
                "Мозокот е офлајн." if speak_mk
                else "My brain is offline, so I can't work out the wiring.",
                success=False)

        # Borrow the council's electrical engineer: the framing that makes the
        # difference between "electronics prose" and a buildable answer.
        sparky = find_specialist("electrical", load_council(self._settings.council.extra))
        role = sparky.prompt if sparky else "You are a senior electrical engineer."
        system = f"{role}\n\n{HOUSE_STYLE}\n\n{WIRING_PROMPT}"
        if speak_mk:
            system += "\n\nWrite the prose sections in Macedonian; keep pin "
            system += "names, units and the code in English."
        try:
            answer = await self._ask(system, question)
        except Exception:
            logger.warning("circuit help failed", exc_info=True)
            answer = ""
        if not answer.strip():
            return SkillResult("Не успеав да го сметам колото." if speak_mk
                               else "I couldn't work that circuit out.",
                               success=False)

        sections = split_sections(answer)
        code = extract_code(sections.get("code", ""))
        saved: Path | None = None
        if code and _wants_code(request.text):
            saved = self._save_sketch(question, code)

        spoken = await self._spoken_summary(answer, speak_mk)
        if saved is not None:
            spoken += (f" Скицата е зачувана во {saved.name}." if speak_mk
                       else f" I've saved the sketch to {saved.name}.")
        return SkillResult(spoken, data={
            "connections": sections.get("connections", ""),
            "watch_out": sections.get("watchout", ""),
            "code": code,
            "sketch_path": str(saved) if saved else None,
            "full": answer,
        })

    async def _spoken_summary(self, answer: str, speak_mk: bool) -> str:
        """Two or three sentences for the ear; the detail lives in ``data``."""
        if self._summarize is None:
            return answer[:600]
        prompt = SPOKEN_PROMPT + ("\n\nAnswer in Macedonian." if speak_mk else "")
        try:
            spoken = (await self._summarize(prompt, answer) or "").strip()
        except Exception:
            spoken = ""
        return spoken or answer[:600]

    def _save_sketch(self, question: str, code: str) -> Path | None:
        """Write the sketch next to the user's other dictated work."""
        slug = re.sub(r"[^\w]+", "-", question.lower()).strip("-")[:40] or "sketch"
        target = expand_path(self._settings.conversation.dictation_dir) / f"{slug}.ino"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(code + "\n", encoding="utf-8")
            return target
        except OSError:
            logger.warning("could not save sketch to %s", target)
            return None

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "build": {"type": "string",
                                  "description": "What is being wired, with the "
                                                 "parts involved."},
                    },
                    "required": ["build"],
                },
            },
        }


def _wants_code(text: str) -> bool:
    """Did they ask for firmware, or only for the wiring?"""
    return bool(re.search(r"\bcode\b|\bsketch\b|\bprogram\b|\bfirmware\b"
                          r"|\bкод\b|\bскица\b|\bпрограм", text, re.IGNORECASE))
