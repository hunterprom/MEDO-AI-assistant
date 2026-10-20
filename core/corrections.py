"""Spotting the moment a user says "no, that's not what I meant".

The fast path is thirty-odd regexes, and a regex that fires on the wrong
sentence produces a confident, sub-millisecond, *wrong* answer — indistinguishable
in every metric MEDO records from a right one. Latency looks great. The user is
saying "no, I meant the weather" and nothing writes that down.

This module recognises that sentence. It is the ground truth for the fast-path
misfire rate (``python -m core.metrics --routing``), and the trigger for
interrupting and re-answering mid-generation.

**Both languages, from the start.** A correction detector written only in
English would report a 0% misfire rate in Macedonian — the same blind spot that
let four English-only review rounds pass while MEDO was broken for nearly every
Macedonian utterance. The Macedonian side is not a translation of the English
list: "не, мислев" and "не тоа" are what people actually say, and the negation
particle ``не`` is written separately from the verb, so it anchors cleanly.

Deliberately **narrow**. A false positive here mislabels a perfectly good route
as a misfire and poisons the very statistic the module exists to produce, so a
phrase only counts as a correction when it *disowns the last turn* — not merely
because it is negative ("no" alone is an answer to a question), and not because
it contains the word "wrong" ("what's wrong with my code" is a question).
"""

from __future__ import annotations

import re

#: Seconds within which a follow-up can still be read as a correction of the
#: previous turn. Long enough for someone to hear a wrong answer out and object
#: (MEDO often speaks for ten seconds), short enough that the *next* unrelated
#: request half a minute later isn't blamed on the last one.
DEFAULT_WINDOW_S = 25.0

#: English. Each entry has to carry the "I'm taking back what you just did"
#: meaning on its own — that is why bare "no" and bare "wrong" are absent.
_EN = (
    r"\bno,?\s+i\s+(?:meant|said|wanted|asked)\b",
    r"\bthat'?s\s+not\s+what\s+i\s+(?:meant|said|asked)\b",
    r"\bi\s+didn'?t\s+(?:say|mean|ask\s+for)\b",
    r"\bnot\s+(?:that|what\s+i\s+meant)\b",
    r"\bi\s+meant\b",
    r"\bwrong\s+(?:one|thing|answer|skill|app|command)\b",
    r"\bthat'?s\s+(?:the\s+)?wrong\b",
    r"\bno,?\s+(?:not|don'?t)\b",
    r"\bstop,?\s+i\s+(?:meant|said|wanted)\b",
    r"\bactually,?\s+i\s+(?:meant|wanted|asked)\b",
    r"\bthat'?s\s+not\s+(?:it|right)\b",
)

#: Macedonian. "мислев" (I meant) and "реков" (I said) are the load-bearing
#: verbs; both take the negation particle in front, written separately.
_MK = (
    r"\bне,?\s+мислев\b",
    r"\bне\s+мислев\s+тоа\b",
    r"\bне\s+го\s+реков\s+тоа\b",
    r"\bне\s+тоа\b",
    r"\bне\s+така\b",
    r"\bмислев\s+на\b",
    r"\bреков\b.{0,20}\bне\b",
    r"\bпогрешн[оаи]\w*\s+(?:команда|апликација|одговор|работа)\b",
    r"\bтоа\s+не\s+е\s+тоа\b",
    r"\bне\s+те\s+прашав\s+за\b",
)

_CORRECTION_RE = re.compile("|".join(_EN + _MK), re.IGNORECASE)

#: A mid-generation *steer* rather than a disowning: the answer is the right
#: kind of answer, it just needs to be different. These do NOT count as routing
#: misfires — the route was fine — but they are what "interrupt and correct"
#: listens for.
_REFINEMENTS = (
    r"\bmake\s+it\s+(?:shorter|longer|simpler|briefer|clearer|faster)\b",
    r"\b(?:shorter|briefer|in\s+one\s+sentence|less\s+detail)\b",
    r"\bmore\s+(?:detail|concise|specific)\b",
    r"\bin\s+(?:english|macedonian)\b",
    r"\bпократко\b",
    r"\bподолго\b",
    r"\bпоедноставно\b",
    r"\bна\s+(?:англиски|македонски)\b",
    r"\bсо\s+повеќе\s+детали\b",
)

_REFINEMENT_RE = re.compile("|".join(_REFINEMENTS), re.IGNORECASE)

#: Words that mark a follow-up as a *revision of the last turn* rather than a
#: fresh request. "Actually, make it shorter" is a correction; "make it shorter"
#: on its own could be the first thing said.
_PIVOTS = (
    r"^\s*(?:no|nope|actually|wait|hang on|hold on)\b",
    r"^\s*(?:не|чекај|почекај|всушност)\b",
)

_PIVOT_RE = re.compile("|".join(_PIVOTS), re.IGNORECASE)


def is_correction(text: str) -> bool:
    """Does ``text`` disown what MEDO just did? (Bilingual, deliberately strict.)"""
    return bool(text) and _CORRECTION_RE.search(text) is not None


def is_refinement(text: str) -> bool:
    """Does ``text`` ask for the same answer, differently?

    "Make it shorter" keeps the topic and changes the shape. Used by the
    interrupt-and-correct path; NOT counted as a routing misfire, because the
    route was right and only the reply was wrong.
    """
    if not text:
        return False
    return _REFINEMENT_RE.search(text) is not None


def starts_with_pivot(text: str) -> bool:
    """Does ``text`` open with a word that marks it as a change of mind?"""
    return bool(text) and _PIVOT_RE.search(text) is not None


def corrects_previous(text: str, *, elapsed_s: float,
                      window_s: float = DEFAULT_WINDOW_S) -> bool:
    """True when ``text`` reads as taking back a route made ``elapsed_s`` ago.

    The time bound is half the signal. "I meant the other one" ten seconds after
    a bad route is evidence about that route; the same words two minutes later
    are about something else entirely, and attributing them would quietly
    corrupt the misfire statistic in the direction that makes the fast path look
    worse than it is.
    """
    # A window of zero (or less) means the feature is OFF, and it has to mean
    # that even when no time has passed. `elapsed > window` alone did not: on
    # Windows time.monotonic() ticks every 15.6 ms, so two routes in the same
    # turn are timestamped IDENTICALLY and `0.0 > 0.0` is False — a disabled
    # window still attributed the correction.
    if window_s <= 0 or elapsed_s < 0 or elapsed_s > window_s:
        return False
    if is_correction(text):
        return True
    # A refinement only counts as a correction of the ROUTE when it is also
    # pivoting away from it ("no, shorter") — a bare "shorter" is a follow-up
    # instruction, and the skill that ran was the right one.
    return is_refinement(text) and starts_with_pivot(text)
