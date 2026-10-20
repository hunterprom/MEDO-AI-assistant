"""Embedding the council, so a question finds the right expert by meaning.

Selection used to be trigger words: each specialist carried a list of nouns, in
English and Macedonian, and whoever's list the question hit got the question.
That works exactly as far as somebody thought to extend it. "Will this bracket
melt at 200 degrees?" has no word on the materials scientist's list. A
specialist a user added in ``council.extra`` has no list at all and could never
be reached by topic — only by name.

So the council is embedded the same way skills already are: each specialist's
role description becomes a vector once, cached in SQLite, and a question is
matched against all of them by cosine similarity. This is the *same*
:class:`~core.route_index.SkillRouteIndex` the semantic router uses — one
implementation, one cache-invalidation rule, one set of tests — pointed at a
separate table so specialists and skills can't collide on a shared name.

No new model and no new dependency: it is the ``nomic-embed-text`` that facts,
documents and the semantic router already load. When it is unavailable the
index is simply empty and selection falls back to the trigger words, which is
the behaviour that existed before this file.
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.route_index import Embedder, SkillRouteIndex

logger = logging.getLogger(__name__)

#: Separate keyspace from the skills index — see the module docstring.
COUNCIL_TABLE = "council_vectors"


class CouncilIndex:
    """Per-specialist role vectors, built once and cached."""

    def __init__(self, db_path: str | Path, embedder: Embedder | None,
                 embedder_id: str) -> None:
        self._index = SkillRouteIndex(db_path, embedder, embedder_id,
                                      table=COUNCIL_TABLE)
        self._embedder = embedder

    def build(self, council) -> int:
        """Embed every specialist (cached ones are reused). Returns the count.

        Synchronous — call it via ``asyncio.to_thread``; embedding is one HTTP
        round-trip to Ollama and the event loop is driving a voice pipeline.
        """
        report = self._index.build(list(council))
        if report.skipped:
            logger.info("council index: %d specialist(s) left unembedded "
                        "(embed model cold or down)", report.skipped)
        self._index.prune([m.key for m in council])
        return len(self._index)

    def scores(self, question: str) -> dict[str, float]:
        """Cosine similarity of every indexed specialist to ``question``.

        Empty when the index is empty or the question can't be embedded — the
        caller then falls back to the trigger words, so an unavailable embedder
        costs accuracy, never an answer.
        """
        if not len(self._index) or self._embedder is None or not question.strip():
            return {}
        try:
            vectors = self._embedder([question])
        except Exception:  # noqa: BLE001 - an embedder failure is a fallback
            logger.debug("council index: embedding the question failed",
                         exc_info=True)
            return {}
        if not vectors:
            return {}
        return dict(self._index._scored(vectors[0]))

    def __len__(self) -> int:
        return len(self._index)


def build_council_index(settings=None):
    """Build the index from live config. Returns ``(index, count)``.

    Used by startup warm-up and by ``python -m core.council_index``.
    """
    from core.config import load_settings
    from core.council import enabled_council, load_council

    settings = settings or load_settings()
    embedder = None
    if settings.memory.embed_model:
        from core.embeddings import embed_texts

        model, host = settings.memory.embed_model, settings.llm.host

        def embedder(texts):  # noqa: E731 - tiny closure over config
            return embed_texts(texts, model, host)

    index = CouncilIndex(settings.memory.db_path, embedder,
                         settings.memory.embed_model or "none")
    council = enabled_council(load_council(settings.council.extra),
                              settings.council.disabled)
    return index, index.build(council)


def _main() -> int:  # pragma: no cover - operator utility
    """``python -m core.council_index`` — build the index and report on it."""
    index, count = build_council_index()
    if not count:
        print("No specialists embedded — is the embed model pulled and Ollama up?")
        return 1
    print(f"Embedded {count} specialist(s).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
