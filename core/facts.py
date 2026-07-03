"""Long-term memory: user facts that survive restarts.

A tiny sqlite-backed store ("remember that I'm allergic to peanuts"). The router
injects the newest facts into the LLM system prompt each turn, and the memory
skills expose remember/recall/forget over both routing paths.

Every method opens a short-lived connection, so instances are safe to create
anywhere (router, skills, tests) without sharing connections across threads —
callers on the event loop should wrap calls in ``asyncio.to_thread``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


class FactsStore:
    """Persisted user facts in the shared assistant database."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.execute(_SCHEMA)
        return conn

    def add(self, fact: str) -> bool:
        """Store a fact. Returns False when an equivalent fact already exists."""
        fact = " ".join(fact.split()).strip(" .")
        if not fact:
            return False
        with self._connect() as conn:
            try:
                conn.execute("INSERT INTO facts (fact) VALUES (?)", (fact,))
                return True
            except sqlite3.IntegrityError:  # UNIQUE COLLATE NOCASE duplicate
                return False

    def recent(self, limit: int = 20) -> list[str]:
        """The newest ``limit`` facts, oldest of those first (stable reading order)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT fact FROM facts ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [r[0] for r in reversed(rows)]

    def forget(self, needle: str) -> int:
        """Delete every fact containing ``needle`` (case-insensitive). Returns count."""
        needle = needle.strip()
        if not needle:
            return 0
        pattern = f"%{needle}%"
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM facts WHERE fact LIKE ? COLLATE NOCASE", (pattern,)
            )
            return cur.rowcount

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
