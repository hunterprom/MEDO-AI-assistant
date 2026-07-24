"""Safety layer: a directory whitelist and a spoken-confirmation gate.

Two independent protections:

* :class:`PathWhitelist` — file skills may only touch paths inside configured
  directories. Resolved with symlinks followed, so ``~/Documents/../.ssh`` can't
  escape the sandbox.
* :func:`is_affirmative` / :func:`is_negative` — parse a spoken yes/no so the
  router can require confirmation before a destructive action runs.

The actual confirm-then-execute flow lives in :class:`~core.router.Router`; this
module provides the mechanism, not the policy.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from core.config import PROJECT_ROOT, expand_path

logger = logging.getLogger(__name__)

# Yes/no word banks live as DATA (lang/confirm_words/<code>.yaml), one file per
# language, so adding a language is a word file — not a code change. The gate
# loads banks for the ACTIVE languages only (constrained two-language mode):
# with active=[en, mk] a Macedonian "да" is a yes, but switch to [en, es] and
# "да" no longer matches while "sí" does. Words are matched against the WHOLE
# normalized reply — never substrings, so "не знам" ("I don't know") is neither.
_CONFIRM_DIR = PROJECT_ROOT / "lang" / "confirm_words"

#: The shipped default active pair — used only when the gate is queried before
#: anyone calls configure_confirm_words() (e.g. a unit test). Mirrors
#: LanguagesConfig's default; production always configures from the live set.
_DEFAULT_ACTIVE = ["en", "mk"]


def _read_bank(code: str) -> tuple[set[str], set[str]] | None:
    """Load one language's (affirmative, negative) word sets, or None if absent."""
    path = _CONFIRM_DIR / f"{code}.yaml"
    if not path.exists():
        return None
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    aff = {str(w).strip().lower() for w in data.get("affirmative", []) if str(w).strip()}
    neg = {str(w).strip().lower() for w in data.get("negative", []) if str(w).strip()}
    return aff, neg


def load_confirm_words(active: list[str],
                       primary: str | None = None) -> tuple[set[str], set[str]]:
    """Merge yes/no banks for the ACTIVE languages.

    A missing bank for an active language is a warning, not a failure: the gate
    then also loads ``primary`` + English so a spoken yes/no is never
    un-parseable (a confirmation gate that can't hear "no" is a safety hole).
    """
    aff: set[str] = set()
    neg: set[str] = set()
    missing: list[str] = []
    for code in dict.fromkeys(str(c).strip().lower() for c in active if str(c).strip()):
        bank = _read_bank(code)
        if bank is None:
            missing.append(code)
            logger.warning("confirm-words: no bank for active language %r "
                           "(lang/confirm_words/%s.yaml) — falling back", code, code)
        else:
            aff |= bank[0]
            neg |= bank[1]
    if missing or not aff or not neg:
        for code in dict.fromkeys([(primary or "en").strip().lower(), "en"]):
            bank = _read_bank(code)
            if bank is not None:
                aff |= bank[0]
                neg |= bank[1]
    return aff, neg


_bank_aff: set[str] | None = None
_bank_neg: set[str] | None = None


def configure_confirm_words(active: list[str], primary: str | None = None) -> None:
    """Load the yes/no banks for the ACTIVE languages. Call once at startup with
    ``settings.active_languages()`` — every ``is_affirmative``/``is_negative``
    thereafter matches only those languages."""
    global _bank_aff, _bank_neg
    _bank_aff, _bank_neg = load_confirm_words(active, primary)


def _bank() -> tuple[set[str], set[str]]:
    global _bank_aff, _bank_neg
    if _bank_aff is None or _bank_neg is None:
        _bank_aff, _bank_neg = load_confirm_words(_DEFAULT_ACTIVE, "en")
    return _bank_aff, _bank_neg


# Whisper decorates short utterances ("Да.", "не,") — strip punctuation and
# collapse whitespace before matching. Apostrophes survive ("don't").
_PUNCT = re.compile(r"[.!,?…]+")
_SPACES = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _SPACES.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def is_affirmative(text: str) -> bool:
    """True if ``text`` reads as a yes in one of the active languages."""
    return _normalize(text) in _bank()[0]


def is_negative(text: str) -> bool:
    """True if ``text`` reads as a no in one of the active languages."""
    return _normalize(text) in _bank()[1]


class PathWhitelist:
    """Decides whether a filesystem path is inside the allowed directories."""

    def __init__(self, directories: list[str]) -> None:
        self.roots: list[Path] = []
        for directory in directories:
            root = expand_path(directory)
            if root.exists():
                self.roots.append(root)

    def is_allowed(self, path: str | Path) -> bool:
        """True if ``path`` resolves to a location under a whitelisted root."""
        try:
            target = expand_path(path)
        except (OSError, RuntimeError):
            return False
        for root in self.roots:
            if target == root or root in target.parents:
                return True
        return False

    def describe(self) -> str:
        """Human list of allowed directories, for error messages."""
        return ", ".join(str(r) for r in self.roots) or "(none configured)"
