"""Spoken 'let me think' fillers that bridge dead air on a SLOW LLM answer.

A local model on a hard question — or Claude Code, which returns nothing until
it's wholly done — can leave 8-30 s of silence after you finish speaking, which
reads as "it froze". This layer fills that gap the way a person would: a quick
acknowledgement for a big question, a "let me think" for anything still silent
after a beat, and one "still working on it" if it drags.

It is the LLM path ONLY, by construction: the voice loop arms it on the
router's ``on_llm_start`` hook, which the fast path never reaches (see
``core/router.py``). So a weather lookup that takes three seconds never gets a
filler — only a genuine model call does.

Phrases are DATA (``lang/filler_phrases/<code>.yaml``), one file per language,
loaded for the ACTIVE languages only (constrained two-language mode) and spoken
in the language of the turn — a missing bank degrades to the primary + English.
This module only DECIDES what/when; the voice loop does the speaking and the
timing, so this stays pure and unit-testable without audio.
"""

from __future__ import annotations

import logging
import random
import re

from core.config import PROJECT_ROOT, FillerConfig

logger = logging.getLogger(__name__)

_FILLER_DIR = PROJECT_ROOT / "lang" / "filler_phrases"
_DEFAULT_ACTIVE = ["en", "mk"]

#: Words that mark a substantive question even when it isn't long — so "why?"
#: or "объасни" earns the quick acknowledgement, not the full 8 s wait.
_BIG_MARKERS = re.compile(
    r"\b(?:explain|why|how\s+do|how\s+does|how\s+would|compare|difference|"
    r"summari[sz]e|analy[sz]e|what\s+do\s+you\s+think|tell\s+me\s+about|"
    r"pros\s+and\s+cons|walk\s+me\s+through|in\s+detail|elaborate|"
    r"об[јj]асни|зошто|спореди|анализира[јj]|раскажи\s+ми|детално)\b",
    re.IGNORECASE)


def _read_bank(code: str) -> tuple[list[str], list[str]] | None:
    """One language's (opening, waiting) phrase lists, or None if absent."""
    path = _FILLER_DIR / f"{code}.yaml"
    if not path.exists():
        return None
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    opening = [str(p).strip() for p in data.get("opening", []) if str(p).strip()]
    waiting = [str(p).strip() for p in data.get("waiting", []) if str(p).strip()]
    return opening, waiting


class Filler:
    """Decides whether a question is 'big', how long to wait, and which phrase
    to speak — all pure. ``rng`` is injectable so tests are deterministic."""

    def __init__(self, config: FillerConfig, active: list[str] | None = None,
                 primary: str = "en", rng: random.Random | None = None) -> None:
        self._cfg = config
        self._active = [c.strip().lower() for c in active] if active else list(_DEFAULT_ACTIVE)
        self._primary = (primary or "en").strip().lower()
        self._rng = rng or random.Random()
        # Load banks for the active languages + primary + English (the floor).
        self._banks: dict[str, tuple[list[str], list[str]]] = {}
        for code in dict.fromkeys([*self._active, self._primary, "en"]):
            bank = _read_bank(code)
            if bank is not None and (bank[0] or bank[1]):
                self._banks[code] = bank
            elif code in self._active:
                logger.warning("filler: no phrases for active language %r", code)

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.enabled) and bool(self._banks)

    # -- timing ---------------------------------------------------------------

    def is_big_question(self, text: str) -> bool:
        if not text:
            return False
        if _BIG_MARKERS.search(text):
            return True
        return len(text.split()) >= self._cfg.big_question_words

    def opening_delay(self, text: str) -> float:
        """Seconds to wait before the first filler: short for a big question."""
        return (self._cfg.big_delay_s if self.is_big_question(text)
                else self._cfg.delay_s)

    @property
    def followup_delay(self) -> float:
        return self._cfg.followup_s

    # -- phrases --------------------------------------------------------------

    def _bank_for(self, language: str | None) -> tuple[list[str], list[str]]:
        lang = (str(language).strip().lower()[:2] if language else self._primary)
        if lang not in self._active or lang not in self._banks:
            lang = self._primary if self._primary in self._banks else "en"
        return self._banks.get(lang) or self._banks.get("en") or ([], [])

    def opening(self, language: str | None = None) -> str:
        opening, _ = self._bank_for(language)
        return self._rng.choice(opening) if opening else ""

    def waiting(self, language: str | None = None) -> str:
        _, waiting = self._bank_for(language)
        return self._rng.choice(waiting) if waiting else ""
