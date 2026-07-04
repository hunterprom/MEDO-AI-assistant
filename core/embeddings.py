"""Semantic embeddings via the local Ollama server (no extra Python deps).

``nomic-embed-text`` (274 MB, pulled once) turns facts and queries into
768-dim vectors; cosine similarity then ranks remembered facts by MEANING —
"what's my dentist's number" retrieves "my dentist is Dr. Petrov, 070..."
even though no word overlaps. Everything here is best-effort and fast to
fail: when Ollama or the model is missing, callers fall back to recency
(the pre-semantic behavior), never crash.

With well under a thousand personal facts, a numpy dot product over all rows
is microseconds — a vector index would be pure overhead at this scale.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def embed_texts(
    texts: Sequence[str],
    model: str = "nomic-embed-text",
    host: str = "http://localhost:11434",
    timeout: float = 10.0,
) -> list[np.ndarray] | None:
    """Embed ``texts`` via Ollama's /api/embed. None when unavailable."""
    if not texts:
        return []
    try:
        import httpx

        resp = httpx.post(
            f"{host.rstrip('/')}/api/embed",
            json={"model": model, "input": list(texts)},
            timeout=timeout,
        )
        resp.raise_for_status()
        vectors = resp.json().get("embeddings") or []
        if len(vectors) != len(texts):
            return None
        return [np.asarray(v, dtype=np.float32) for v in vectors]
    except Exception as exc:  # server down, model missing, network...
        logger.debug("embedding unavailable: %s", exc)
        return None


def rank_by_similarity(
    query: np.ndarray, candidates: Sequence[np.ndarray]
) -> list[int]:
    """Indices of ``candidates`` sorted most-similar-first to ``query`` (cosine)."""
    if not len(candidates):
        return []
    matrix = np.stack([c / (np.linalg.norm(c) or 1.0) for c in candidates])
    q = query / (np.linalg.norm(query) or 1.0)
    scores = matrix @ q
    return list(np.argsort(-scores))
