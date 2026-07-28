"""Personality layer: one persona, two delivery points (M7).

* **LLM path** — :meth:`Persona.prompt_fragment` returns a 2–3 sentence manner
  instruction that ``llm/prompts.py`` folds into the system prompt (kept tiny:
  ``num_ctx`` is 4096).
* **Fast path** — :meth:`Persona.decorate` occasionally appends one curated
  quip to a skill's literal reply, with probability ``wit_level``.

The charm is a decorator layer, NOT baked into skill strings: skills stay
literal and testable, the persona is swappable from config, and the safety
rules live in exactly one place — a quip is never added to errors, safety
confirmations, or destructive actions (the router only decorates successful,
non-gated fast-path outcomes; see ``Router._run_skill``).

Quips are bilingual: with ``quips_language: match`` a Macedonian utterance
gets a Macedonian quip (detected by Cyrillic), everything else English.
"""

from __future__ import annotations

import random
import re

from core.config import PersonalityConfig

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")

#: Which fast-path skills may be decorated, and with which quip table.
#: Deliberately small (the council's scope): timers, volume, apps, power.
CATEGORY_BY_SKILL = {
    "timers": "timers",
    "volume": "volume",
    "apps": "apps",
    "power": "power",
}

#: Short, dry, spoken-friendly. No emoji, no punctuation tricks — TTS reads these.
QUIPS: dict[str, dict[str, list[str]]] = {
    "timers": {
        "en": [
            "The countdown is my problem now.",
            "Consider it tracked.",
            "I'll do the remembering.",
        ],
        "mk": [
            "Јас ќе бројам.",
            "Сметајте дека е запишано.",
            "Одбројувањето е моја грижа сега.",
        ],
    },
    "volume": {
        "en": [
            "Your eardrums, your call.",
            "Adjusted, tastefully.",
            "As loud as you like it.",
        ],
        "mk": [
            "Ваши уши, ваша одлука.",
            "Наместено, со вкус.",
            "Колку сакате, толку.",
        ],
    },
    "apps": {
        "en": [
            "Opening. Try to look busy.",
            "At your service.",
            "Launching, with ceremony.",
        ],
        "mk": [
            "Се отвора. Изгледајте зафатено.",
            "На ваша услуга.",
            "Стартувам, свечено.",
        ],
    },
    "power": {
        "en": [
            "Rest is a feature, not a bug.",
            "Lights out, as requested.",
        ],
        "mk": [
            "Одморот е функција, не грешка.",
            "Гасиме, како што баравте.",
        ],
    },
}

_STYLE_FRAGMENTS = {
    "dry_wit": (
        "Your manner is concise and dryly witty — think a capable butler, not "
        "a cheerful chatbot. A light touch of understatement is welcome; "
        "never at the expense of clarity."
    ),
    "professional": (
        "Your manner is crisp, courteous, and strictly professional. "
        "No jokes, no asides — clear answers, delivered efficiently."
    ),
    "minimal": (
        "Your manner is minimal: answer with the fewest words that fully "
        "answer the question. No pleasantries, no commentary."
    ),
}


class Persona:
    """Loaded persona; ``rng`` is injectable so tests are deterministic."""

    def __init__(self, config: PersonalityConfig,
                 rng: random.Random | None = None,
                 active: list[str] | None = None,
                 primary: str = "en") -> None:
        self._cfg = config
        self._rng = rng or random.Random()
        # Constrained two-language mode (S3): only quip in an ACTIVE language;
        # a language with no quip bank degrades to primary. None = no constraint
        # (legacy behaviour, used by callers that don't pass the language set).
        self._active = [c.strip().lower() for c in active] if active else None
        self._primary = (primary or "en").strip().lower()

    # -- LLM path -------------------------------------------------------------

    def prompt_fragment(self) -> str:
        """The manner sentence(s) for the system prompt (2–3 sentences max)."""
        return _STYLE_FRAGMENTS.get(self._cfg.style, _STYLE_FRAGMENTS["dry_wit"])

    # -- fast path ------------------------------------------------------------

    def decorate(self, speech: str, *, skill_name: str, user_text: str,
                 language: str | None = None) -> str:
        """Return ``speech``, occasionally with one appended quip.

        The router calls this ONLY for successful, non-confirmation fast-path
        outcomes of non-destructive skills — errors and safety prompts never
        reach here, so they stay literal by construction.
        """
        cfg = self._cfg
        if cfg.style == "minimal" or cfg.wit_level <= 0.0 or not speech:
            return speech
        category = CATEGORY_BY_SKILL.get(skill_name)
        if category is None:
            return speech
        if self._rng.random() >= min(cfg.wit_level, 1.0):
            return speech
        banks = QUIPS[category]
        lang = self._quip_language(language, user_text, banks)
        options = banks.get(lang) or banks.get(self._primary) or next(iter(banks.values()))
        return f"{speech} {self._rng.choice(options)}"

    def _quip_language(self, detected: str | None, user_text: str,
                       banks: dict[str, list[str]]) -> str:
        """Which language to quip in: the utterance's language, constrained to
        the active set (a non-active or bank-less language degrades to primary)."""
        cfg = self._cfg
        if cfg.quips_language != "match":
            return cfg.quips_language
        # Prefer the STT-detected code; fall back to a script guess for Cyrillic.
        lang = (str(detected).strip().lower()[:2] if detected else
                ("mk" if _CYRILLIC.search(user_text) else self._primary))
        if self._active is not None and lang not in self._active:
            lang = self._primary          # never quip in a non-active language
        if lang not in banks:
            lang = self._primary if self._primary in banks else next(iter(banks))
        return lang
