"""RAG over the user's documents: index, embed, and semantically search files.

Walks the safety-whitelisted directories for text-bearing files (.txt/.md/.pdf),
chunks them, embeds each chunk with the same local Ollama embedder the facts
store uses (:mod:`core.embeddings`), and answers "what do my documents say
about X" by cosine similarity over the chunks. Everything is stored in the
shared sqlite database; indexing is incremental (unchanged files — by mtime —
are skipped, chunks of deleted files are pruned).

Design constraints, deliberately:
- Only whitelisted trees are ever read (same guarantee as the files skill).
- Indexing is best-effort and bounded (file-size cap, total-chunk cap) so a
  huge Documents folder can't eat the disk or stall startup — it runs in a
  background thread.
- No embedder (Ollama down / disabled) -> search returns [] and the skill
  says so; nothing crashes.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

#: Embeds a batch of texts; None when the backend is unavailable.
Embedder = Callable[[Sequence[str]], "list | None"]

INDEXED_EXTS = {".txt", ".md", ".pdf"}
MAX_FILE_KB = 512          # skip anything bigger (logs, dumps)
MAX_PDF_PAGES = 40         # first pages only — enough for letters/papers
MAX_TOTAL_CHUNKS = 6000    # hard cap for the whole index (~18 MB of vectors)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS doc_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    mtime REAL NOT NULL,
    idx INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS doc_chunks_path ON doc_chunks(path);
"""


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Split text into ~size-char chunks with overlap, preferring line breaks."""
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):  # try to break at a newline/space near the end
            for sep in ("\n", ". ", " "):
                cut = text.rfind(sep, start + size // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_text(path: Path) -> str:
    """Best-effort plain text from a file (txt/md read; pdf via pypdf)."""
    try:
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            pages = reader.pages[:MAX_PDF_PAGES]
            return "\n".join((p.extract_text() or "") for p in pages)
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.debug("could not extract %s: %s", path, exc)
        return ""


class DocumentIndex:
    """Incremental chunk+embed index over whitelisted directories."""

    def __init__(self, db_path: str | Path, embedder: Embedder | None,
                 roots: Sequence[Path]) -> None:
        self._db_path = str(db_path)
        self._embedder = embedder
        self._roots = [Path(r) for r in roots]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.executescript(_SCHEMA)
        return conn

    # -- indexing -------------------------------------------------------------

    def _iter_files(self):
        for root in self._roots:
            if not root.exists():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                for name in filenames:
                    p = Path(dirpath) / name
                    if p.suffix.lower() not in INDEXED_EXTS:
                        continue
                    try:
                        if p.stat().st_size > MAX_FILE_KB * 1024:
                            continue
                    except OSError:
                        continue
                    yield p

    def reindex(self) -> dict:
        """Index new/changed files, prune deleted ones. Returns stats. Sync —
        call via ``asyncio.to_thread``; safe to re-run any time."""
        if self._embedder is None:
            return {"indexed": 0, "pruned": 0, "chunks": 0, "enabled": False}
        indexed = pruned = 0
        seen: set[str] = set()
        with self._connect() as conn:
            known = dict(conn.execute(
                "SELECT path, MAX(mtime) FROM doc_chunks GROUP BY path").fetchall())
            total = conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0]
        for path in self._iter_files():
            key = os.path.normpath(str(path))
            seen.add(key)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if known.get(key) == mtime:
                continue  # unchanged
            if total >= MAX_TOTAL_CHUNKS:
                logger.info("document index full (%d chunks); skipping the rest", total)
                break
            chunks = chunk_text(extract_text(path))
            if not chunks:
                continue
            vectors = self._embedder(chunks)
            if not vectors:
                logger.info("embedder unavailable — pausing document indexing")
                break
            with self._connect() as conn:
                conn.execute("DELETE FROM doc_chunks WHERE path = ?", (key,))
                for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                    conn.execute(
                        "INSERT INTO doc_chunks (path, mtime, idx, text, embedding)"
                        " VALUES (?, ?, ?, ?, ?)",
                        (key, mtime, i, chunk, vec.tobytes()),
                    )
            total += len(chunks)
            indexed += 1
        # prune chunks of files that no longer exist under the roots
        with self._connect() as conn:
            for (path,) in conn.execute("SELECT DISTINCT path FROM doc_chunks").fetchall():
                if path not in seen and not Path(path).exists():
                    conn.execute("DELETE FROM doc_chunks WHERE path = ?", (path,))
                    pruned += 1
            chunks_total = conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0]
        if indexed or pruned:
            logger.info("document index: +%d files, -%d pruned, %d chunks total",
                        indexed, pruned, chunks_total)
        return {"indexed": indexed, "pruned": pruned, "chunks": chunks_total,
                "enabled": True}

    # -- search ---------------------------------------------------------------

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Top-``k`` chunks for ``query``: [{"path", "text", "score"}, ...]."""
        if self._embedder is None or not query.strip():
            return []
        qvec = self._embedder([query])
        if not qvec:
            return []
        import numpy as np

        from core.embeddings import rank_by_similarity

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT path, text, embedding FROM doc_chunks"
                " WHERE embedding IS NOT NULL").fetchall()
        if not rows:
            return []
        vectors = [np.frombuffer(r[2], dtype=np.float32) for r in rows]
        order = rank_by_similarity(qvec[0], vectors)
        q = qvec[0] / (np.linalg.norm(qvec[0]) or 1.0)
        out = []
        for idx in order[:k]:
            v = vectors[idx]
            score = float((v / (np.linalg.norm(v) or 1.0)) @ q)
            out.append({"path": rows[idx][0], "text": rows[idx][1], "score": score})
        return out

    def stats(self) -> dict:
        with self._connect() as conn:
            files, chunks = conn.execute(
                "SELECT COUNT(DISTINCT path), COUNT(*) FROM doc_chunks").fetchone()
        return {"files": int(files), "chunks": int(chunks),
                "enabled": self._embedder is not None}
