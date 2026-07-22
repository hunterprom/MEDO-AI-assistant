"""Long-term memory: user facts that survive restarts.

A tiny sqlite-backed store ("remember that I'm allergic to peanuts"). The router
injects facts into the LLM system prompt each turn — semantically RELEVANT ones
when an embedder is available (see :mod:`core.embeddings`), newest-N otherwise —
and the memory skills expose remember/recall/forget over both routing paths.

Every method opens a short-lived connection, so instances are safe to create
anywhere (router, skills, tests) without sharing connections across threads —
callers on the event loop should wrap calls in ``asyncio.to_thread``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path

#: Embeds a batch of texts to vectors; None when the backend is unavailable.
Embedder = Callable[[Sequence[str]], "list | None"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


class FactsStore:
    """Persisted user facts in the shared assistant database."""

    def __init__(self, db_path: str | Path, embedder: Embedder | None = None) -> None:
        self._db_path = str(db_path)
        self._embedder = embedder

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.execute(_SCHEMA)
        # Older databases predate the embedding column; add it in place.
        try:
            conn.execute("ALTER TABLE facts ADD COLUMN embedding BLOB")
        except sqlite3.OperationalError:  # already there
            pass
        return conn

    def add(self, fact: str) -> bool:
        """Store a fact. Returns False when an equivalent fact already exists."""
        fact = " ".join(fact.split()).strip(" .")
        if not fact:
            return False
        with self._connect() as conn:
            try:
                conn.execute("INSERT INTO facts (fact) VALUES (?)", (fact,))
            except sqlite3.IntegrityError:  # UNIQUE COLLATE NOCASE duplicate
                return False
        self._backfill_embeddings()  # best-effort, outside the insert txn
        return True

    def recent(self, limit: int = 20) -> list[str]:
        """The newest ``limit`` facts, oldest of those first (stable reading order)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT fact FROM facts ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [r[0] for r in reversed(rows)]

    def relevant(self, query: str, limit: int = 20) -> list[str]:
        """Facts worth injecting for ``query``: semantic matches + the newest few.

        With an embedder available, ranks stored facts by cosine similarity to
        the query and blends in the newest ones (so brand-new facts surface
        even before they're semantically related to anything). Ordered by id —
        the stable reading order the prompt expects. Falls back to
        :meth:`recent` whenever embeddings can't be computed.
        """
        if self._embedder is None or not query.strip():
            return self.recent(limit)
        self._backfill_embeddings()
        import numpy as np

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, fact, embedding FROM facts ORDER BY id"
            ).fetchall()
        if len(rows) <= limit:
            return [r[1] for r in rows]

        qvec = self._embedder([query])
        embedded = [(r[0], r[1], r[2]) for r in rows if r[2] is not None]
        if not qvec or not embedded:
            return self.recent(limit)

        from core.embeddings import rank_by_similarity

        vectors = [np.frombuffer(r[2], dtype=np.float32) for r in embedded]
        order = rank_by_similarity(qvec[0], vectors)
        # Semantic matches first (they're what the query needs); the newest
        # facts fill the remaining slots so fresh memories still surface.
        chosen: set[int] = set()
        semantic_budget = max(1, limit - 3)
        for idx in order:
            if len(chosen) >= semantic_budget:
                break
            chosen.add(embedded[idx][0])
        for row in reversed(rows):
            if len(chosen) >= limit:
                break
            chosen.add(row[0])
        return [r[1] for r in rows if r[0] in chosen]

    def _backfill_embeddings(self, batch: int = 32) -> None:
        """Embed facts that don't have vectors yet (best-effort, never raises)."""
        if self._embedder is None:
            return
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT id, fact FROM facts WHERE embedding IS NULL LIMIT ?",
                    (int(batch),),
                ).fetchall()
            if not rows:
                return
            vectors = self._embedder([r[1] for r in rows])
            if not vectors:
                return
            with self._connect() as conn:
                for (fact_id, _), vec in zip(rows, vectors, strict=True):
                    conn.execute(
                        "UPDATE facts SET embedding = ? WHERE id = ?",
                        (vec.tobytes(), fact_id),
                    )
        except Exception:  # embeddings are an enhancement, never a failure mode
            import logging

            logging.getLogger(__name__).debug("embedding backfill failed", exc_info=True)

    def list_all(self) -> list[dict]:
        """Every fact as ``{"id", "fact", "created_at"}``, oldest first (for UIs)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, fact, created_at FROM facts ORDER BY id"
            ).fetchall()
        return [{"id": r[0], "fact": r[1], "created_at": r[2]} for r in rows]

    def delete(self, fact_id: int) -> bool:
        """Delete one fact by id (the HUD memory manager). True when it existed."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM facts WHERE id = ?", (int(fact_id),))
            return cur.rowcount > 0

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
