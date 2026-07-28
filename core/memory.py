"""Storage & memory.

* :class:`NoteStore` — persistent voice notes (SQLite). [M2]
* :class:`ReminderStore` — persistent reminders that survive restarts. [M4]
* :class:`ConversationMemory` — an in-memory rolling window of recent turns fed
  to the LLM so follow-ups ("and tomorrow?") resolve from context. [M4]
"""

from __future__ import annotations

import sqlite3
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.config import PROJECT_ROOT


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class Note:
    id: int
    text: str
    created_at: str


class NoteStore:
    """A minimal notes table with add / list / delete / search."""

    def __init__(self, db_path: str | Path) -> None:
        path = Path(db_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: skills run in worker threads via to_thread.
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS notes ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " text TEXT NOT NULL,"
            " created_at TEXT NOT NULL)"
        )
        self._db.commit()

    def add(self, text: str) -> Note:
        created = _now_iso()
        cur = self._db.execute(
            "INSERT INTO notes (text, created_at) VALUES (?, ?)", (text, created)
        )
        self._db.commit()
        return Note(id=int(cur.lastrowid), text=text, created_at=created)

    def list(self, limit: int = 20) -> list[Note]:
        rows = self._db.execute(
            "SELECT id, text, created_at FROM notes ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [Note(**dict(r)) for r in rows]

    def search(self, term: str, limit: int = 20) -> list[Note]:
        # Escape LIKE wildcards so "50%" or "_" match literally instead of
        # matching every note (see core.facts._like_escape).
        from core.facts import _like_escape

        rows = self._db.execute(
            "SELECT id, text, created_at FROM notes WHERE text LIKE ? ESCAPE '\\' "
            "ORDER BY id DESC LIMIT ?",
            (f"%{_like_escape(term)}%", limit),
        ).fetchall()
        return [Note(**dict(r)) for r in rows]

    def delete(self, note_id: int) -> bool:
        cur = self._db.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        self._db.commit()
        return cur.rowcount > 0

    def clear(self) -> int:
        cur = self._db.execute("DELETE FROM notes")
        self._db.commit()
        return cur.rowcount

    def close(self) -> None:
        self._db.close()


@dataclass
class Reminder:
    id: int
    due_at: str      # ISO timestamp
    label: str
    created_at: str


class ReminderStore:
    """Reminders that persist across restarts (rescheduled on startup)."""

    def __init__(self, db_path: str | Path) -> None:
        path = Path(db_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS reminders ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " due_at TEXT NOT NULL,"
            " label TEXT NOT NULL,"
            " created_at TEXT NOT NULL)"
        )
        self._db.commit()

    def add(self, due_at: str, label: str) -> Reminder:
        cur = self._db.execute(
            "INSERT INTO reminders (due_at, label, created_at) VALUES (?, ?, ?)",
            (due_at, label, _now_iso()),
        )
        self._db.commit()
        return Reminder(int(cur.lastrowid), due_at, label, _now_iso())

    def all(self) -> list[Reminder]:
        rows = self._db.execute(
            "SELECT id, due_at, label, created_at FROM reminders ORDER BY due_at"
        ).fetchall()
        return [Reminder(**dict(r)) for r in rows]

    def delete(self, reminder_id: int) -> bool:
        cur = self._db.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        self._db.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self._db.close()


class ConversationMemory:
    """A rolling window of recent (user, assistant) turns for LLM context.

    In-memory by design: context is per-session and short-lived, unlike notes and
    reminders. ``recent_messages`` yields OpenAI-style role dicts to prepend to a
    prompt so follow-ups resolve ("and tomorrow?" after a weather question).
    """

    def __init__(self, max_turns: int = 10) -> None:
        self._turns: deque[tuple[str, str]] = deque(maxlen=max_turns)

    def add_turn(self, user: str, assistant: str) -> None:
        if user and assistant:
            self._turns.append((user, assistant))

    def recent_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for user, assistant in self._turns:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        return messages

    def last_reply(self) -> str | None:
        """MEDO's most recent spoken reply, for a verbatim 'say that again'."""
        return self._turns[-1][1] if self._turns else None

    def last_question(self) -> str | None:
        """The user's most recent utterance — the question that produced
        :meth:`last_reply`, for 'are you sure?' / 'second opinion'."""
        return self._turns[-1][0] if self._turns else None

    def clear(self) -> None:
        self._turns.clear()
