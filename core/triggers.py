"""User-defined trigger phrases: extra fast-path wording, from config.

Every skill ships regex patterns, and adding a phrase used to mean editing
Python. That is the wrong place for it — which words a person actually says is
their business, not the code's, and it differs per language and per household
("покажи ми ги вестите", "гимме the news", "što ima novo").

So ``skills.triggers`` in config.yaml maps a skill name to extra phrases:

    skills:
      triggers:
        news: ["што има ново", "brief me on the news"]
        weather: ["дали ќе врне"]

Each phrase becomes one more pattern on that skill's fast path — the same
mechanism the built-in patterns use, appended to the same list, matched by the
same registry. There is no second matching path and no branch anywhere that
knows a trigger came from config rather than from code.

Two deliberate limits:

* Phrases are matched **literally**, not as regex. A user typing ``what's up?``
  should get what they typed, not a syntax error from ``?`` — and a
  config file should never be able to inject a catastrophic backtracking
  pattern into the hot path.
* A trigger can only *add* wording to a skill that already exists. It cannot
  create a skill, change what one does, or bypass a confirmation gate: a
  destructive skill reached by a custom phrase is still destructive and still
  asks.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

#: A phrase shorter than this matches far too much ("go", "hi") and would
#: shadow every skill registered after it. Rejected rather than silently
#: hijacking the router.
MIN_PHRASE_LEN = 3


def phrase_to_pattern(phrase: str) -> re.Pattern[str] | None:
    """Compile ONE spoken phrase into a fast-path pattern, or None if unusable.

    The phrase is escaped (it is words, not regex), inner whitespace becomes
    flexible so "what s up" survives a transcript with odd spacing, and word
    boundaries are added at each end only where the phrase actually starts or
    ends in a word character — so "?" or a trailing "!" doesn't produce a
    boundary that can never match.
    """
    text = " ".join((phrase or "").split())
    if len(text) < MIN_PHRASE_LEN:
        return None
    body = r"\s+".join(re.escape(word) for word in text.split(" "))
    prefix = r"\b" if text[0].isalnum() else ""
    suffix = r"\b" if text[-1].isalnum() else ""
    try:
        return re.compile(prefix + body + suffix, re.IGNORECASE)
    except re.error:                          # pragma: no cover - escaped input
        return None


def apply_triggers(registry, triggers: dict | None) -> dict[str, list[str]]:
    """Attach configured phrases to their skills. Returns a report.

    The report is ``{"added": [...], "unknown": [...], "rejected": [...]}`` and
    is logged rather than raised: a typo in one trigger must not stop MEDO from
    booting, but it must not vanish silently either — an "unknown skill" line
    in the log is how the user finds out their phrase does nothing.

    Patterns are appended to a **copy** of the skill's list and assigned to the
    instance, so the class attribute (shared by every other instance, including
    the ones tests build) is never mutated.
    """
    report: dict[str, list[str]] = {"added": [], "unknown": [], "rejected": []}
    if not triggers:
        return report
    for name, phrases in triggers.items():
        skill = registry.get(str(name))
        if skill is None:
            report["unknown"].append(str(name))
            continue
        if isinstance(phrases, str):          # a single phrase, unbracketed
            phrases = [phrases]
        extra = []
        for phrase in phrases or []:
            pattern = phrase_to_pattern(str(phrase))
            if pattern is None:
                report["rejected"].append(f"{name}: {phrase!r}")
                continue
            extra.append(pattern)
            report["added"].append(f"{name}: {phrase}")
        if extra:
            # Custom phrases go FIRST: the user wrote them for this skill
            # specifically, so they should win over a built-in pattern that
            # merely happens to overlap.
            skill.patterns = extra + list(skill.patterns)
    if report["unknown"]:
        logger.warning(
            "skills.triggers names skills that don't exist: %s — check the "
            "spelling against `python main.py --list-skills`",
            ", ".join(sorted(set(report["unknown"]))))
    if report["rejected"]:
        logger.warning("skills.triggers ignored %d unusable phrase(s): %s",
                       len(report["rejected"]), "; ".join(report["rejected"]))
    if report["added"]:
        logger.info("skills.triggers added %d custom phrase(s)", len(report["added"]))
    return report
