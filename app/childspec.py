"""What the supervisor needs to know about one child process.

Deliberately data-only + duck-typed so the supervisor is testable with fakes:
``health`` and ``start_if`` are plain callables (probe a port, check a daemon),
and ``command`` is already resolved for the current mode (dev venv vs frozen
bundle) by whoever builds the spec — the supervisor never branches on that.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable, List


def is_frozen() -> bool:
    """True when running inside a PyInstaller/Nuitka bundle (vs. dev checkout)."""
    return bool(getattr(sys, "frozen", False))


def _always_true() -> bool:
    return True


@dataclass
class ChildSpec:
    """One supervised process.

    ``friendly_name`` is what the user sees in a plain-language error. ``health``
    reports whether the child is actually SERVING (a port answering), used both
    to decide readiness (open the browser) and to skip launching something that's
    already up (a shared Ollama). ``start_if`` is an extra gate — Ollama is only
    started when it isn't already reachable, and is then the only thing we own.
    """

    name: str
    command: List[str]
    friendly_name: str = ""
    health: Callable[[], bool] = field(default=_always_true)
    start_if: Callable[[], bool] = field(default=_always_true)
    #: How many restarts within ``restart_window_s`` before we give up and show a
    #: friendly error instead of crash-looping.
    max_restarts: int = 3
    restart_window_s: float = 60.0
    #: Exponential backoff base: waits base, 2·base, 4·base… between restarts.
    backoff_base_s: float = 1.0

    def label(self) -> str:
        return self.friendly_name or self.name
