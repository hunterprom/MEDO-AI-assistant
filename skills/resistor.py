"""Resistor colour-band decoder — both directions, and never fabricated.

"What does brown black red gold mean?" has no fast pattern anywhere else, so it
falls to the LLM — and a small model reliably swaps the multiplier and the
tolerance band, reading a 1 kΩ resistor back as something else entirely. This is
pure lookup-and-multiply arithmetic, so MEDO answers it deterministically:

    "brown black red gold"        -> "That's 1k ohms, 5 percent tolerance."
    "colour bands for 4.7k"       -> "The bands are yellow, violet, red, gold."

Both directions, English and Macedonian (Whisper writes the colours in Cyrillic
when you're speaking Macedonian), and voice-shaped output. A workshop skill for
a workshop assistant; no network, no model, no state.

The logic lives in module-level pure functions so it is trivially unit-tested
without the async skill wrapper.
"""

from __future__ import annotations

import re
from typing import Any

from core import mk
from skills.base import Skill, SkillRequest, SkillResult

# --- the colour code ---------------------------------------------------------

#: Colour -> significant digit (0-9). "purple" and "gray" are common spoken
#: variants of violet/grey.
DIGIT: dict[str, int] = {
    "black": 0, "brown": 1, "red": 2, "orange": 3, "yellow": 4,
    "green": 5, "blue": 6, "violet": 7, "purple": 7, "grey": 8, "gray": 8,
    "white": 9,
}

#: Colour -> multiplier EXPONENT (the band multiplies by 10**exp). The digit
#: colours reuse their digit as the exponent; gold/silver are the fractional
#: multipliers.
MULT_EXP: dict[str, int] = {**DIGIT, "gold": -1, "silver": -2}

#: Colour -> tolerance in percent. A resistor with no tolerance band is 20%.
TOLERANCE: dict[str, float] = {
    "brown": 1.0, "red": 2.0, "green": 0.5, "blue": 0.25, "violet": 0.1,
    "grey": 0.05, "gray": 0.05, "gold": 5.0, "silver": 10.0,
}

#: Digit -> canonical colour name, for the value -> bands direction.
DIGIT_NAME: tuple[str, ...] = (
    "black", "brown", "red", "orange", "yellow",
    "green", "blue", "violet", "grey", "white",
)

#: Macedonian colour word -> English key (Whisper transcribes spoken colours in
#: Cyrillic). Applied before any lookup.
MK_COLOR: dict[str, str] = {
    "црна": "black", "црн": "black",
    "кафеава": "brown", "кафена": "brown", "браон": "brown",
    "црвена": "red", "црвен": "red",
    "портокалова": "orange", "портокал": "orange", "портокалива": "orange",
    "жолта": "yellow", "жолт": "yellow",
    "зелена": "green", "зелен": "green",
    "сина": "blue", "плава": "blue", "син": "blue",
    "виолетова": "violet", "виолетов": "violet", "виолетна": "violet",
    "сива": "grey", "сив": "grey",
    "бела": "white", "бел": "white",
    "златна": "gold", "злато": "gold",
    "сребрена": "silver", "сребро": "silver", "сребренаста": "silver",
}

#: English key -> spoken Macedonian colour, for the value -> bands direction.
EN_TO_MK: dict[str, str] = {
    "black": "црна", "brown": "кафеава", "red": "црвена", "orange": "портокалова",
    "yellow": "жолта", "green": "зелена", "blue": "сина", "violet": "виолетова",
    "purple": "виолетова", "grey": "сива", "gray": "сива", "white": "бела",
    "gold": "златна", "silver": "сребрена",
}

#: Preferred values (E24 series) — the reverse direction snaps to the nearest.
_E24 = (10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30, 33, 36, 39, 43, 47,
        51, 56, 62, 68, 75, 82, 91)

_COLOR_WORDS = set(DIGIT) | {"gold", "silver"}


# --- pure functions (the whole feature, testable with no skill/async) --------

def normalize_colors(tokens: list[str]) -> list[str]:
    """Spoken colour tokens (either language) -> English keys, in order."""
    out: list[str] = []
    for tok in tokens:
        t = tok.strip().lower()
        if t in _COLOR_WORDS:
            out.append(t)
        elif t in MK_COLOR:
            out.append(MK_COLOR[t])
    return out


def decode_bands(colors: list[str]) -> tuple[float, float | None]:
    """Colour bands -> (ohms, tolerance percent or None for a bare 3-band).

    Structure is decided by the band COUNT, the way you read a real resistor:
    3 = 2 digits + multiplier, 4 = 2 digits + multiplier + tolerance,
    5/6 = 3 digits + multiplier + tolerance (a 6th temperature band is ignored).
    Raises ValueError on an unusable count or an unknown colour in a slot.
    """
    colors = [c.strip().lower() for c in colors]
    n = len(colors)
    if n == 3:
        ndigits, has_tol = 2, False
    elif n == 4:
        ndigits, has_tol = 2, True
    elif n in (5, 6):
        ndigits, has_tol = 3, True
    else:
        raise ValueError(f"need 3 to 6 bands, got {n}")

    digits = colors[:ndigits]
    mult = colors[ndigits]
    try:
        figure = int("".join(str(DIGIT[d]) for d in digits))
        value = figure * (10.0 ** MULT_EXP[mult])
    except KeyError as exc:
        raise ValueError(f"not a resistor colour: {exc.args[0]}") from exc
    tol = None
    if has_tol:
        tol = TOLERANCE.get(colors[ndigits + 1])
        if tol is None:
            raise ValueError(f"{colors[ndigits + 1]} is not a tolerance band")
    return float(value), tol


def value_to_bands(ohms: float, five_band: bool = False) -> list[str]:
    """Ohms -> the 4- (or 5-) band colour sequence, snapped to E24, 5% (gold)."""
    if ohms <= 0:
        raise ValueError("resistance must be positive")
    sig = float(ohms)
    exp = 0
    while sig >= 100:
        sig /= 10
        exp += 1
    while sig < 10:
        sig *= 10
        exp -= 1
    snapped = min(_E24, key=lambda e: abs(e - sig))
    d1, d2 = divmod(snapped, 10)
    if five_band:
        # 3 significant figures: the third digit is 0 for an E24 value.
        return [DIGIT_NAME[d1], DIGIT_NAME[d2], DIGIT_NAME[0],
                _mult_name(exp - 1), "gold"]
    return [DIGIT_NAME[d1], DIGIT_NAME[d2], _mult_name(exp), "gold"]


def _mult_name(exp: int) -> str:
    if exp == -1:
        return "gold"
    if exp == -2:
        return "silver"
    if 0 <= exp <= 9:
        return DIGIT_NAME[exp]
    raise ValueError(f"multiplier 10^{exp} has no single band")


def human_ohms(ohms: float) -> str:
    """Spoken-friendly magnitude: 4700 -> '4.7k', 1_000_000 -> '1M', 220 -> '220'."""
    if ohms >= 1_000_000:
        value, unit = ohms / 1_000_000, "M"
    elif ohms >= 1000:
        value, unit = ohms / 1000, "k"
    else:
        value, unit = ohms, ""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{text}{unit}"


def parse_ohms(text: str) -> float | None:
    """A resistance named in the utterance -> ohms, or None. Case matters for
    the mega/kilo suffix, so this reads the ORIGINAL (non-lowered) text."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*"
                  r"(k(?:ohms?|ilohms?)?|meg(?:ohms?)?|m(?:ohms?)?|"
                  r"r|ohms?|Ω)?\b", text, re.IGNORECASE)
    if not m:
        return None
    num = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit.startswith("k"):
        num *= 1000
    elif unit.startswith("meg") or unit == "m" or unit.startswith("mohm"):
        num *= 1_000_000        # resistors: "m"/"M"/"meg" == mega, not milli
    return num


def _spoken_bands(bands: list[str], speak_mk: bool) -> str:
    names = [EN_TO_MK.get(b, b) if speak_mk else b for b in bands]
    return ", ".join(names)


# --- the skill ---------------------------------------------------------------

class ResistorSkill(Skill):
    """Decode resistor colour bands, or give the bands for a value."""

    name = "resistor_colors"
    controls_pc = False
    description = (
        "Decode a resistor's colour bands into its resistance and tolerance, or "
        "give the colour bands for a resistance value (e.g. 'what does brown "
        "black red gold mean', 'colour bands for a 4.7k resistor')."
    )
    # Reached by MEANING too: execute() reads the colours/value from request.text.
    semantic_from_text = True
    routing_phrases = [
        "what does brown black red gold mean",
        "decode these resistor bands",
        "what colour bands for a four point seven k resistor",
        "what colours is a two twenty ohm resistor",
        "read out these resistor colour bands",
        "what value is this resistor",
    ]

    _COLOR_RE = ("black|brown|red|orange|yellow|green|blue|violet|purple|"
                 "grey|gray|white|gold|silver")
    _MK_COLOR_RE = "|".join(sorted(MK_COLOR, key=len, reverse=True))

    patterns = [
        # A run of 3+ colour words IS a band spec: "brown black red gold",
        # "red red brown". This is the case with no "resistor"/"band" keyword.
        re.compile(rf"\b(?:{_COLOR_RE})(?:[\s,]+(?:and\s+)?(?:{_COLOR_RE})){{2,}}\b",
                   re.IGNORECASE),
        re.compile(rf"\b(?:{_MK_COLOR_RE})(?:[\s,]+(?:и\s+)?(?:{_MK_COLOR_RE})){{2,}}\b",
                   re.IGNORECASE),
        # value -> bands, and generic resistor/band context.
        re.compile(r"\b(?:colou?r\s+bands?|resistor\s+colou?rs?|colou?r\s+code)\b",
                   re.IGNORECASE),
        re.compile(r"\bresistor\b.*\b(?:band|colou?r|value|ohms?)\b", re.IGNORECASE),
        re.compile(r"\b(?:band|colou?r)s?\b.*\bresistor\b", re.IGNORECASE),
        # MK: "кои бои се за отпорник", "отпорник ... бои/оми"
        re.compile(r"\bкои\s+бои\b", re.IGNORECASE),
        re.compile(r"\bотпорник\b.*\b(?:бои|боја|оми|ом|вредност)\b", re.IGNORECASE),
    ]

    def _need_more(self, speak_mk: bool) -> SkillResult:
        return SkillResult(
            "Дај ми барем три бои, или вредност како 4.7k." if speak_mk
            else "Give me at least three colour bands, or a value like 4.7k.",
            success=False)

    async def execute(self, request: SkillRequest) -> SkillResult:
        speak_mk = mk.is_cyrillic(request.text)
        # LLM tool path may pass explicit args; otherwise read the utterance
        # (the same text also carries the colours on the semantic path).
        band_src = str(request.args.get("bands") or "") or request.text
        colors = normalize_colors(re.findall(r"[A-Za-zА-Ша-шЀ-џ]+", band_src))

        # Direction 1: three or more colours -> decode to a value.
        if len(colors) >= 3:
            try:
                ohms, tol = decode_bands(colors)
            except ValueError:
                return self._need_more(speak_mk)
            tol_text = f"{tol:g}" if tol is not None else "20"
            return SkillResult(
                f"Тоа е {human_ohms(ohms)} оми, толеранција {tol_text} проценти."
                if speak_mk else
                f"That's {human_ohms(ohms)} ohms, {tol_text} percent tolerance.",
                data={"ohms": ohms, "tolerance": tol, "bands": colors})

        # Direction 2: a value named -> give the bands.
        arg_ohms = request.args.get("ohms")
        ohms = (float(arg_ohms) if arg_ohms not in (None, "")
                else parse_ohms(request.text))
        if ohms is not None:
            try:
                bands = value_to_bands(ohms)
            except ValueError:
                return self._need_more(speak_mk)
            return SkillResult(
                f"Боите се {_spoken_bands(bands, speak_mk)}." if speak_mk
                else f"The bands are {_spoken_bands(bands, False)}.",
                data={"ohms": ohms, "bands": bands})

        return self._need_more(speak_mk)

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bands": {
                            "type": "string",
                            "description": "Colour bands to decode, e.g. "
                                           "'brown black red gold'.",
                        },
                        "ohms": {
                            "type": "number",
                            "description": "A resistance to convert to bands.",
                        },
                    },
                    "required": [],
                },
            },
        }
