"""Local speaker verification — prove the OWNER, fully on-device (S2).

Enrolment records ONLY a voiceprint EMBEDDING (a short float vector), NEVER raw
audio. The embedder is INJECTED — the real one is a local model (resemblyzer /
ECAPA-TDNN, an *optional* dependency); MEDO ships without it, so this module
degrades SAFELY: with no embedder or no enrolment, :meth:`verify` returns False
and the policy engine denies high-impact actions (fail closed), telling the user
to enrol.

Defensive only: it compares a fresh embedding to the stored one by cosine
similarity against a threshold. No cloud, no attack surface, no raw audio at rest
— the voiceprint file is written owner-only, like the secrets file.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

#: audio (whatever the embedder accepts) -> a fixed-length embedding vector.
Embedder = Callable[[object], Sequence[float]]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]; -1.0 for empty/ragged/zero vectors (so a
    degenerate embedding can never clear a positive threshold)."""
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return -1.0
    return dot / (na * nb)


class VoiceprintStore:
    """Persists ONLY the owner's embedding — owner-readable, git-ignored."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self) -> list[float] | None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        vec = data.get("voiceprint") if isinstance(data, dict) else None
        if not vec:
            return None
        try:
            return [float(x) for x in vec]
        except (TypeError, ValueError):
            return None

    def save(self, embedding: Sequence[float]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"voiceprint": [float(x) for x in embedding]})
        # Owner-only, atomic-ish create — mirrors the secrets file hygiene.
        fd = os.open(str(self._path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)
        try:
            os.chmod(self._path, 0o600)
        except OSError:  # best effort on filesystems without POSIX perms
            pass

    def exists(self) -> bool:
        return self.load() is not None

    def clear(self) -> None:
        try:
            self._path.unlink()
        except OSError:
            pass


class SpeakerVerifier:
    """Enrol the owner's voiceprint and verify a fresh sample against it.

    ``embedder`` is optional: without it (the default install), the verifier is
    UNAVAILABLE and every :meth:`verify` returns False — the fail-closed path the
    policy engine relies on when ``owner_voice`` is on but nothing is enrolled.
    """

    def __init__(self, store: VoiceprintStore, embedder: Embedder | None = None,
                 threshold: float = 0.75) -> None:
        self._store = store
        self._embed = embedder
        self._threshold = threshold

    def is_available(self) -> bool:
        return self._embed is not None

    def is_enrolled(self) -> bool:
        return self._store.exists()

    def enroll(self, samples: Sequence[object]) -> bool:
        """Average the embeddings of a few short samples into the stored
        voiceprint. Returns False (nothing stored) if the embedder is missing or
        the samples don't yield consistent vectors."""
        if self._embed is None or not samples:
            return False
        embs: list[list[float]] = []
        for s in samples:
            try:
                vec = list(self._embed(s))
            except Exception:
                logger.warning("enrol embed failed for one sample", exc_info=True)
                continue
            if vec:
                embs.append([float(x) for x in vec])
        if not embs:
            return False
        n = len(embs[0])
        if any(len(e) != n for e in embs):   # ragged -> refuse, don't guess
            return False
        avg = [sum(e[i] for e in embs) / len(embs) for i in range(n)]
        self._store.save(avg)
        return True

    def verify(self, sample: object) -> bool:
        """True only if a real embedder + enrolment exist AND the sample clears
        the threshold. Every other path (no embedder, not enrolled, embed error,
        low similarity) returns False — fail closed."""
        sim = self.similarity(sample)
        return sim is not None and sim >= self._threshold

    def similarity(self, sample: object) -> float | None:
        """Cosine similarity to the enrolled voiceprint, or None when it can't be
        computed (no embedder / not enrolled / embed error)."""
        if self._embed is None:
            return None
        stored = self._store.load()
        if stored is None:
            return None
        try:
            probe = list(self._embed(sample))
        except Exception:
            logger.warning("verify embed failed", exc_info=True)
            return None
        return cosine(probe, stored)
