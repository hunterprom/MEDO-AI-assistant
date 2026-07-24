"""Skill embedding index for the Tier-2 semantic router (M2.5a).

Tier 2 sits between MEDO's deterministic fast path and the LLM: when no regex
matches, an utterance can still reach the right skill by *meaning*. To do that
cheaply, each skill's routing surface (its example phrases + description, see
``Skill.routing_text``) is embedded ONCE and held in memory; the vectors are
cached in the shared SQLite so a restart is instant and re-embedding happens
only when a skill's text — or the embedder — changes.

Deliberate design choices:

* **Reuse the existing embedder.** The same local Ollama ``nomic-embed-text``
  the facts/RAG stack uses (:func:`core.embeddings.embed_texts`). No new model,
  no new dependency. Embedder down → the index is simply empty and the router
  falls through to the LLM, exactly as today.
* **Cache key = hash(skill id + routing text + embedder id).** Change any of
  the three and that skill re-embeds; leave them and it's a cache hit. The
  embedder id (model name) is in the key so swapping embed models can never
  serve stale vectors from a different space.
* **Open to more than skills.** :meth:`build` takes any *routing source* — an
  object with ``.name`` and ``.routing_text()``. Today those are skills; M11
  plugins and M15+ MEDO Link device capabilities can register through the same
  door without changing this file.

This module is INERT with respect to routing: it builds and caches vectors and
nothing else. The matcher (M2.5b) and the shadow/live wiring (M2.5c/d) are
separate — so nothing here can alter what MEDO does.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: Embeds a batch of strings; None when the backend is unavailable (mirrors
#: core.embeddings.embed_texts / core.docindex.Embedder).
Embedder = Callable[[Sequence[str]], "list[np.ndarray] | None"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skill_route_vectors (
    skill        TEXT PRIMARY KEY,
    key          TEXT NOT NULL,          -- hash(skill + routing_text + embedder)
    routing_text TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    embedding    BLOB NOT NULL
);
"""


def cache_key(skill_id: str, routing_text: str, embedder_id: str) -> str:
    """Stable hash of the three things that make a vector (in)valid.

    NUL-separated so ``("a", "bc")`` and ``("ab", "c")`` can't collide.
    """
    h = hashlib.sha256()
    h.update(skill_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(routing_text.encode("utf-8"))
    h.update(b"\x00")
    h.update(embedder_id.encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True)
class RouteEntry:
    """One skill's cached routing vector, held in memory for the matcher."""

    skill: str
    routing_text: str
    vector: np.ndarray
    phrase_count: int


@dataclass
class BuildRow:
    """One line of the ``--index`` report."""

    skill: str
    phrases: int
    status: str          # "cached" | "fresh" | "skipped"


@dataclass
class BuildReport:
    rows: list[BuildRow] = field(default_factory=list)
    enabled: bool = True         # False when the embedder is unavailable

    @property
    def cached(self) -> int:
        return sum(1 for r in self.rows if r.status == "cached")

    @property
    def fresh(self) -> int:
        return sum(1 for r in self.rows if r.status == "fresh")

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.rows if r.status == "skipped")


class SkillRouteIndex:
    """Builds + caches per-skill routing vectors; holds them for the matcher."""

    def __init__(self, db_path: str | Path, embedder: Embedder | None,
                 embedder_id: str) -> None:
        self._db_path = str(db_path)
        self._embedder = embedder
        #: Identifies the embedding SPACE; part of every cache key so a model
        #: swap invalidates cleanly rather than mixing incompatible vectors.
        self._embedder_id = embedder_id or "none"
        self._entries: list[RouteEntry] = []

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        return conn

    # -- build ----------------------------------------------------------------

    def build(self, sources: Iterable) -> BuildReport:
        """(Re)build the index from routing SOURCES. Sync — call via to_thread.

        A *source* is anything with ``.name`` (str) and ``.routing_text()``
        (str) — skills today, plugins/devices later. Cached vectors whose key
        still matches are reused; only changed/new sources are re-embedded, in
        one batched call. The embedder being down is not an error: those
        sources are reported "skipped" and the index holds whatever it could.
        """
        report = BuildReport()
        sources = list(sources)
        with self._connect() as conn:
            cached = {r["skill"]: r for r in conn.execute(
                "SELECT skill, key, routing_text, dim, embedding "
                "FROM skill_route_vectors").fetchall()}

        entries: list[RouteEntry] = []
        to_embed: list[tuple[str, str, str, int]] = []   # skill, text, key, count
        for src in sources:
            name = src.name
            text = src.routing_text()
            count = len(src.routing_surface()) if hasattr(src, "routing_surface") else 1
            key = cache_key(name, text, self._embedder_id)
            row = cached.get(name)
            if row is not None and row["key"] == key:
                vec = np.frombuffer(row["embedding"], dtype=np.float32)
                entries.append(RouteEntry(name, text, vec, count))
                report.rows.append(BuildRow(name, count, "cached"))
            else:
                to_embed.append((name, text, key, count))

        if to_embed:
            vectors = self._embedder([t for _, t, _, _ in to_embed]) \
                if self._embedder is not None else None
            if vectors and len(vectors) == len(to_embed):
                with self._connect() as conn:
                    for (name, text, key, count), vec in zip(to_embed, vectors, strict=True):
                        v = np.asarray(vec, dtype=np.float32)
                        conn.execute(
                            "INSERT INTO skill_route_vectors "
                            "(skill, key, routing_text, dim, embedding) "
                            "VALUES (?,?,?,?,?) "
                            "ON CONFLICT(skill) DO UPDATE SET "
                            "key=excluded.key, routing_text=excluded.routing_text, "
                            "dim=excluded.dim, embedding=excluded.embedding",
                            (name, key, text, int(v.shape[0]), v.tobytes()))
                        entries.append(RouteEntry(name, text, v, count))
                        report.rows.append(BuildRow(name, count, "fresh"))
            else:
                # Embedder unavailable / wrong count: keep any cache hits, mark
                # the rest skipped, and flag the whole index as not fully ready.
                report.enabled = self._embedder is not None and bool(entries)
                for name, _text, _key, count in to_embed:
                    report.rows.append(BuildRow(name, count, "skipped"))
                if not vectors and self._embedder is not None:
                    logger.info("route index: embedder unavailable — %d skill(s) "
                                "left unindexed", len(to_embed))

        # Preserve the source order in the report; entries follow build order.
        order = {s.name: i for i, s in enumerate(sources)}
        report.rows.sort(key=lambda r: order.get(r.skill, 1 << 30))
        self._entries = entries
        return report

    # -- access ---------------------------------------------------------------

    def entries(self) -> list[RouteEntry]:
        """In-memory routing vectors (empty until :meth:`build`)."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def prune(self, live_names: Iterable[str]) -> int:
        """Drop cached rows for skills that no longer exist. Returns count."""
        keep = set(live_names)
        with self._connect() as conn:
            rows = [r["skill"] for r in conn.execute(
                "SELECT skill FROM skill_route_vectors").fetchall()]
            gone = [s for s in rows if s not in keep]
            for s in gone:
                conn.execute("DELETE FROM skill_route_vectors WHERE skill = ?", (s,))
        return len(gone)
