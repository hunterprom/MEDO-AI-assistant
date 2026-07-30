"""Centralized secret access + log redaction (S5).

One place to READ secrets (the :8710 token, cloud API keys, MCP creds) and one
helper to SCRUB them from anything about to be logged. MEDO already avoids
logging secrets; this makes it a reusable guarantee rather than a per-call
discipline. Secrets themselves keep living in git-ignored ``secrets.local.yaml``
(owner-only, chmod 0600) — this module only reads, never widens, their exposure.

At-rest encryption of the most sensitive stored data (voiceprint, flagged memory)
is an OPTIONAL future add (needs a crypto dep); left out of the default install on
purpose. See docs/Security.md.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

#: Settings attributes that hold a secret value, by dotted name.
_SECRET_ATTRS = {
    "remote.token": lambda s: getattr(s.remote, "token", ""),
    "llm.api_key": lambda s: getattr(s.llm, "api_key", ""),
    "llm.anthropic_api_key": lambda s: getattr(s.llm, "anthropic_api_key", ""),
}

_MASK = "***"


class Secrets:
    """Read-only accessor over the process's secret values."""

    def __init__(self, settings: Any) -> None:
        self._s = settings

    def get(self, name: str) -> str:
        """Return a secret by dotted name (settings first, then
        secrets.local.yaml), or '' if absent."""
        getter = _SECRET_ATTRS.get(name)
        if getter is not None:
            try:
                return str(getter(self._s) or "")
            except Exception:
                return ""
        # fall back to a raw key in secrets.local.yaml
        try:
            from core.config import SECRETS_PATH, _read_local
            return str(_read_local(SECRETS_PATH).get(name, "") or "")
        except Exception:
            return ""

    def has(self, name: str) -> bool:
        return bool(self.get(name))

    def values(self) -> List[str]:
        """Every live secret value (for redaction). Only non-trivial ones — a
        1-char 'secret' would scrub far too much."""
        out: List[str] = []
        for getter in _SECRET_ATTRS.values():
            try:
                v = str(getter(self._s) or "")
            except Exception:
                v = ""
            if len(v) >= 6:
                out.append(v)
        return out

    def redact(self, text: Optional[str]) -> str:
        """Replace any secret value appearing in ``text`` with ``***``. Use on
        anything headed for a log or the audit trail."""
        if not text:
            return text or ""
        red = str(text)
        for v in self.values():
            if v and v in red:
                red = red.replace(v, _MASK)
        return red


def redact(text: Optional[str], settings: Any) -> str:
    """Module-level convenience: scrub ``settings``' secrets from ``text``."""
    return Secrets(settings).redact(text)
