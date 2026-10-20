"""Which app the user is currently working in — lightweight session context.

So "open the tools menu" or "find me tools" can resolve to the app in focus (or
the one just named) instead of asking "which app?". Two signals feed it: the
live foreground window (resolved in the skills, against the learned apps) and
this holder's memory of the last app named/used. Shared across the app-learning
skills and updated as they run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SessionAppContext:
    """Remembers the last app the user named or was resolved to, this session."""

    last_app: str = ""

    def note(self, app: str) -> None:
        app = (app or "").strip()
        if app:
            self.last_app = app
