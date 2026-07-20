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

import re
from pathlib import Path

from core.config import expand_path

# Kept small and unambiguous; matched against the WHOLE normalized reply, never
# by substring — "не знам" ("I don't know") contains "не" but must not cancel.
# Unknown replies fall through to the router, which refuses to guess with a
# destructive action pending. Macedonian entries cover both Cyrillic and the
# Latin transliterations Whisper sometimes produces for spoken Macedonian.
_AFFIRMATIVE = {
    # English
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "confirmed",
    "do it", "go ahead", "affirmative", "please do", "yes please",
    # Macedonian (Cyrillic)
    "да", "може", "секако", "ајде", "важи", "потврди", "потврдувам",
    "да те молам",
    # Latin transliterations
    "da", "moze", "ajde", "vazi",
}
_NEGATIVE = {
    # English
    "no", "nope", "nah", "cancel", "stop", "don't", "dont", "abort", "never mind",
    "nevermind", "negative", "no thanks",
    # Macedonian (Cyrillic)
    "не", "откажи", "стоп", "прекини", "немој", "заборави", "не сакам",
    # Latin transliterations
    "ne", "otkazi",
}

# Whisper wraps single-word confirmations in punctuation ("Да.", "не!", "ok,")
# and sometimes doubles spaces. Apostrophes are NOT stripped ("don't").
_PUNCT = re.compile(r"[.!,?…;:]+")
_SPACES = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — Whisper-proofing."""
    return _SPACES.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def is_affirmative(text: str) -> bool:
    """True if ``text`` reads as a yes (English or Macedonian)."""
    return _normalize(text) in _AFFIRMATIVE


def is_negative(text: str) -> bool:
    """True if ``text`` reads as a no (English or Macedonian)."""
    return _normalize(text) in _NEGATIVE


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
