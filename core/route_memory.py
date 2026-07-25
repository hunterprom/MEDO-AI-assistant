"""Adaptive route memory — the semantic tier that LEARNS (M2.5e).

The curated Tier-2 index (:mod:`core.route_index`) reaches a skill by meaning
using phrases MEDO ships with. This module adds the other half: when an
utterance misses the regex fast path AND the curated semantic tier, and the LLM
then resolves it to exactly ONE non-destructive query skill, that resolution is
worth remembering. Store ``(normalized utterance -> skill)`` as an embedded
exemplar and, next time the same or near-same phrasing arrives, the semantic
tier shortcuts straight to that skill — no LLM round-trip, personalized to how
THIS user actually talks.

Same spirit as :mod:`core.route_index`, deliberately kept separate:

* **Reuse the one embedder.** Vectors come from the same local
  ``nomic-embed-text`` closure already on ``Router._embedder``. No new model,
  no new dependency, fully offline. Embedder down => nothing is learned and
  nothing is matched.
* **Embedder id in every row.** A model swap changes the embedding space, so
  rows are filtered by ``embedder`` on load — a new model simply sees an empty
  store rather than matching stale vectors from another space.
* **Learn only what's safe to replay.** The router only ever calls
  :meth:`remember` for skills that are non-actuation, non-confirmation, and
  already curated-tier eligible (``routing_phrases``). A destructive action can
  never become a learned shortcut, and even a replayed exemplar still passes
  through the router's unchanged confirmation / PC-control gates.

Like the curated index this module is INERT with respect to routing: it stores
and matches vectors, nothing else. The router decides whether to consult it
(``router.route_memory_enabled``) and whether a hit acts or only shadow-logs.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS route_memory (
    norm_text  TEXT PRIMARY KEY,          -- normalized utterance (safety._normalize)
    skill      TEXT NOT NULL,             -- the skill the LLM resolved it to
    embedder   TEXT NOT NULL,             -- embedding space this vector lives in
    dim        INTEGER NOT NULL,
    embedding  BLOB NOT NULL,
    hits       INTEGER NOT NULL DEFAULT 1, -- times this phrasing was re-confirmed
    created    REAL NOT NULL,
    last_used  REAL NOT NULL
);
"""


class RouteMemory:
    """A learned-exemplar store consulted alongside the curated route index."""

    def __init__(self, db_path: str | Path, embedder_id: str) -> None:
        self._db_path = str(db_path)
        #: Identifies the embedding SPACE; rows from another embedder are ignored
        #: on load, so a model swap can never serve incompatible vectors.
        self._embedder_id = embedder_id or "none"
        self._entries: list[tuple[str, np.ndarray]] = []
        self._loaded = False

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        return conn

    # -- load -----------------------------------------------------------------

    def _load(self) -> None:
        """Read this embedder's rows into memory for the matcher."""
        entries: list[tuple[str, np.ndarray]] = []
        try:
            with self._connect() as conn:
                for row in conn.execute(
                        "SELECT skill, embedding FROM route_memory WHERE embedder = ?",
                        (self._embedder_id,)).fetchall():
                    vec = np.frombuffer(row["embedding"], dtype=np.float32)
                    entries.append((row["skill"], vec))
        except Exception:  # a locked/broken DB must never break routing
            logger.debug("route memory load failed", exc_info=True)
            entries = []
        self._entries = entries
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()

    # -- learn ----------------------------------------------------------------

    def remember(self, norm_text: str, skill: str, vector: "np.ndarray",
                 max_rows: int | None = None, now: float | None = None) -> None:
        """Record (or reinforce) one learned exemplar. Best-effort, never raises.

        Same phrasing, same skill => bump ``hits`` (the mapping is proven again).
        Same phrasing, DIFFERENT skill => overwrite and reset ``hits`` to 1: the
        latest LLM resolution wins, so a correction supersedes a stale mapping.
        When ``max_rows`` is set and the store overflows, the least-used rows
        (lowest ``hits`` then oldest ``last_used``) are evicted.
        """
        now = time.time() if now is None else now
        v = np.asarray(vector, dtype=np.float32).ravel()
        if v.size == 0:
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO route_memory "
                    "(norm_text, skill, embedder, dim, embedding, hits, created, last_used) "
                    "VALUES (?,?,?,?,?,1,?,?) "
                    "ON CONFLICT(norm_text) DO UPDATE SET "
                    "  skill=excluded.skill, embedder=excluded.embedder, "
                    "  dim=excluded.dim, embedding=excluded.embedding, "
                    "  last_used=excluded.last_used, "
                    # same skill -> keep counting; changed skill -> restart at 1
                    "  hits=CASE WHEN route_memory.skill=excluded.skill "
                    "            THEN route_memory.hits + 1 ELSE 1 END",
                    (norm_text, skill, self._embedder_id, int(v.shape[0]),
                     v.tobytes(), now, now))
                if max_rows is not None:
                    (count,) = conn.execute(
                        "SELECT COUNT(*) FROM route_memory").fetchone()
                    if count > max_rows:
                        conn.execute(
                            "DELETE FROM route_memory WHERE norm_text IN ("
                            "  SELECT norm_text FROM route_memory "
                            "  ORDER BY hits ASC, last_used ASC LIMIT ?)",
                            (count - max_rows,))
        except Exception:
            logger.debug("route memory write failed", exc_info=True)
            return
        # Matches vastly outnumber writes, so invalidate and lazily reload.
        self._loaded = False

    # -- match ----------------------------------------------------------------

    def match(self, query_vector: "np.ndarray | None",
              threshold: float = 0.82) -> "tuple[str, float] | None":
        """Best learned exemplar for a query embedding, or None.

        Pure cosine against every stored vector — NO runner-up margin: a learned
        exemplar is an (almost) exact single user phrasing, so a high
        ``threshold`` is the sole gate. Read-only: ``hits`` are bumped at learn
        time, never here, so a consulted turn writes nothing.
        """
        self._ensure_loaded()
        if not self._entries or query_vector is None:
            return None
        q = np.asarray(query_vector, dtype=np.float32).ravel()
        qn = float(np.linalg.norm(q))
        if qn == 0.0:
            return None
        q = q / qn
        best_skill: str | None = None
        best = -1.0
        for skill, v in self._entries:
            v = v.ravel()
            vn = float(np.linalg.norm(v))
            if vn == 0.0 or v.shape != q.shape:
                continue
            score = float(q @ (v / vn))
            if score > best:
                best, best_skill = score, skill
        if best_skill is not None and best >= threshold:
            return best_skill, best
        return None

    # -- maintenance ----------------------------------------------------------

    def prune(self, live_names: Iterable[str]) -> int:
        """Drop learned rows whose skill no longer exists. Returns count."""
        keep = set(live_names)
        removed = 0
        try:
            with self._connect() as conn:
                rows = [r["norm_text"] for r in conn.execute(
                    "SELECT norm_text, skill FROM route_memory").fetchall()
                    if r["skill"] not in keep]
                for nt in rows:
                    conn.execute("DELETE FROM route_memory WHERE norm_text = ?", (nt,))
                removed = len(rows)
        except Exception:
            logger.debug("route memory prune failed", exc_info=True)
        if removed:
            self._loaded = False
        return removed

    def forget(self, skill: str | None = None) -> int:
        """Forget one skill's exemplars, or ALL of them when ``skill`` is None."""
        removed = 0
        try:
            with self._connect() as conn:
                if skill is None:
                    (removed,) = conn.execute(
                        "SELECT COUNT(*) FROM route_memory").fetchone()
                    conn.execute("DELETE FROM route_memory")
                else:
                    cur = conn.execute(
                        "DELETE FROM route_memory WHERE skill = ?", (skill,))
                    removed = cur.rowcount
        except Exception:
            logger.debug("route memory forget failed", exc_info=True)
        self._loaded = False
        return removed

    def clear(self) -> int:
        """Empty the whole store (revert / tests). Returns rows removed."""
        return self.forget(None)

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._entries)
