"""Does a summary only say what its source said?

The failure this exists for, from a live session:

    YOU   Search up for Mazda RX-7
    MEDO  "... with nearly half a million RX-7s sold across the model's
           seven-year run in markets like the US, Australia, and Japan."

The RX-7 ran 1978-2002 and no snippet mentioned a sales figure at all. Asked
again here with the real DuckDuckGo results, the summarizer produced "Over
800,000 units of the RX-7 were produced during its lifetime" — a different
invented number, from the model's weights rather than from the page.

Everything else in the answer was fine. That is what makes this shape of
hallucination expensive: it rides along with true prose, in the one register
(a confident figure) a listener has no way to check. Tightening the prompt was
tried first and did NOT stop it — llama3.2:3b still stated 800,000 when told in
plain words to use only the results. So the check has to be mechanical.

Deliberately narrow. A summary may reword, reorder, and drop things freely;
only FIGURES are checked, and only the kind a listener would take as a claim:

* magnitudes (>= 100, or any number with a "%", or a spoken "million"), because
  that is what gets invented — production totals, prices, percentages;
* NOT years (1900-2100), which are the numbers that legitimately get reworded
  ("from 1978 to 2002" -> "in the late seventies");
* NOT small counts ("three generations"), which paraphrase constantly.

A figure counts as grounded when its digits match one the source states, so
"800,000" grounds "800000" and formatting never trips it. Matching is exact,
which does flag a ROUNDING ("over 800,000" against a source saying 811,634).
That is the trade taken on purpose: the cost of a false flag is one re-ask and
possibly a dropped clause, and the cost of a miss is an invented number spoken
as fact.
"""

from __future__ import annotations

import re

#: A run of digits, with the separators people write inside numbers.
_NUMBER = re.compile(r"\d[\d,.]*\d|\d")
#: Spoken magnitudes. A fabricated total often arrives with no digits at all
#: ("nearly half a million"), which a digit-only check would sail past.
_MAGNITUDE = re.compile(
    r"\b(?:hundreds?|thousands?|millions?|billions?|"
    r"стотин\w*|илјад\w*|милион\w*|милијард\w*)\b", re.IGNORECASE)
#: Sentence split for the fallback. Crude on purpose — it only has to find the
#: sentence carrying a figure, and an over-long "sentence" costs one extra clause.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _digits(text: str) -> str:
    """The digits of a number, separators removed ("800,000" -> "800000")."""
    return re.sub(r"[^\d]", "", text)


def _source_numbers(source: str) -> set[str]:
    """Every number the source states, normalized to its digits.

    A SET of whole numbers, not one concatenated digit blob. Concatenating was
    the first cut and it grounds figures by accident: a source reading "5 GB,
    13 units" yields "...513..." and would happily vouch for a summary that
    invented "513". Each figure has to match a figure.
    """
    return {d for d in (_digits(m.group()) for m in _NUMBER.finditer(source or ""))
            if d}


def _claim_numbers(text: str) -> list[tuple[str, str]]:
    """(surface form, digits) for every number that reads as a CLAIM.

    Years and small counts are skipped — see the module docstring for why.
    """
    out: list[tuple[str, str]] = []
    for m in _NUMBER.finditer(text or ""):
        surface = m.group().strip(" ,.")
        digits = _digits(surface)
        if not digits:
            continue
        percent = text[m.end():m.end() + 1] == "%"
        value = int(digits)
        is_year = len(digits) == 4 and 1900 <= value <= 2100
        if is_year and not percent:
            continue
        if value < 100 and not percent:
            continue
        out.append((surface, digits))
    return out


def ungrounded_figures(summary: str, source: str) -> list[str]:
    """Figures the summary states that the source never did.

    Empty means the summary invented no numbers — not that it is true, only
    that every figure in it came from the text it was given.
    """
    if not summary or not source:
        return []
    stated = _source_numbers(source)
    found: list[str] = []
    for surface, digits in _claim_numbers(summary):
        if digits and digits not in stated and surface not in found:
            found.append(surface)
    for m in _MAGNITUDE.finditer(summary):
        word = m.group()
        # Compare on the stem: "millions" in the summary is grounded by
        # "million" in the source, and Macedonian inflects heavily.
        stem = word.lower()[:6]
        if stem not in source.lower() and word not in found:
            found.append(word)
    return found


def drop_unsupported(summary: str, figures: list[str]) -> str:
    """The summary minus every sentence carrying one of ``figures``.

    The last resort, after a re-ask has already failed. Dropping the sentence
    is better than dropping the answer: in the RX-7 case two accurate sentences
    survive and only the invented total goes.
    """
    if not summary or not figures:
        return summary
    kept = [s for s in _SENTENCE.split(summary)
            if not any(f in s for f in figures)]
    return " ".join(part.strip() for part in kept if part.strip()).strip()


def repair_demand(figures: list[str]) -> str:
    """What to say to the model when its summary invented a figure."""
    named = ", ".join(f"'{f}'" for f in figures)
    return (
        f"The search results do not contain {named}. You added that from "
        f"memory, and a figure nobody can check is worse than no figure. Say "
        f"the summary again in two or three plain spoken sentences using only "
        f"what the results actually state, leaving that out entirely."
    )


async def ground_summary(summary: str, source: str, reask) -> str:
    """``summary``, re-asked once if it states figures ``source`` doesn't.

    ``reask`` is an async callable taking the demand and returning a fresh
    summary ("" when the model is unavailable). Whatever comes back is checked
    again; anything still unsupported has its sentence dropped, because a
    second failure means the model can't do it, not that it needs asking twice.
    """
    bad = ungrounded_figures(summary, source)
    if not bad:
        return summary
    second = ""
    try:
        second = (await reask(repair_demand(bad)) or "").strip()
    except Exception:  # a failed repair must never cost the whole answer
        second = ""
    if second:
        still = ungrounded_figures(second, source)
        if not still:
            return second
        return drop_unsupported(second, still) or drop_unsupported(summary, bad)
    return drop_unsupported(summary, bad)
