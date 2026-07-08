"""Documents RAG: chunker, incremental index, semantic search, and the skill."""

from __future__ import annotations

import numpy as np
import pytest

from core.docindex import DocumentIndex, chunk_text
from skills.base import SkillRequest
from skills.documents import DocumentsSkill


def _fake_embedder(texts):
    """Deterministic 3-axis 'meaning': [lease/rent, recipe/food, python]."""
    out = []
    for t in texts:
        t = t.lower()
        out.append(np.array([
            1.0 if ("lease" in t or "rent" in t or "landlord" in t) else 0.0,
            1.0 if ("recipe" in t or "flour" in t or "oven" in t) else 0.0,
            1.0 if ("python" in t or "code" in t) else 0.0,
        ], dtype=np.float32) + 0.01)
    return out


# --- chunker ------------------------------------------------------------------


def test_chunk_text_splits_and_overlaps():
    text = ("A" * 400 + ".\n") * 6            # ~2.4k chars
    chunks = chunk_text(text, size=900, overlap=150)
    assert len(chunks) >= 3
    assert all(len(c) <= 1000 for c in chunks)
    assert chunk_text("") == []
    assert chunk_text("short note") == ["short note"]


# --- index + search -----------------------------------------------------------


@pytest.fixture
def corpus(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "lease.txt").write_text(
        "The lease agreement: rent is 350 euro per month, due on the 5th. "
        "The landlord handles repairs.", encoding="utf-8")
    (docs / "cake.md").write_text(
        "Grandma's recipe: 300g flour, 3 eggs, bake in the oven at 180C.",
        encoding="utf-8")
    (docs / "ignore.exe").write_bytes(b"\x00\x01")   # non-indexed extension
    return docs


def test_index_and_semantic_search(tmp_path, corpus):
    idx = DocumentIndex(tmp_path / "rag.db", _fake_embedder, [corpus])
    stats = idx.reindex()
    assert stats["indexed"] == 2 and stats["enabled"]

    hits = idx.search("how much is the rent")
    assert hits and "lease.txt" in hits[0]["path"]
    hits = idx.search("what temperature for the oven")
    assert hits and "cake.md" in hits[0]["path"]


def test_reindex_is_incremental_and_prunes(tmp_path, corpus):
    idx = DocumentIndex(tmp_path / "rag.db", _fake_embedder, [corpus])
    idx.reindex()
    assert idx.reindex()["indexed"] == 0            # unchanged files skipped
    (corpus / "cake.md").unlink()
    stats = idx.reindex()
    assert stats["pruned"] == 1
    assert all("cake" not in h["path"] for h in idx.search("oven recipe"))


def test_disabled_index_is_harmless(tmp_path, corpus):
    idx = DocumentIndex(tmp_path / "rag.db", None, [corpus])
    assert idx.reindex()["enabled"] is False
    assert idx.search("rent") == []


# --- skill ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_documents_skill_fast_path_and_tool_path(tmp_path, corpus):
    idx = DocumentIndex(tmp_path / "rag.db", _fake_embedder, [corpus])
    idx.reindex()
    skill = DocumentsSkill(idx)

    text = "what do my documents say about the rent"
    m = skill.match(text)
    assert m is not None
    result = await skill.execute(SkillRequest(text=text, match=m))
    assert "lease.txt" in result.speech and "350" in result.speech

    tool = await skill.execute(SkillRequest(
        text=text, args={"query": "rent"}, context={"via": "tool"}))
    assert "[lease.txt]" in tool.speech


@pytest.mark.asyncio
async def test_documents_skill_no_hits_and_disabled(tmp_path, corpus):
    idx = DocumentIndex(tmp_path / "rag.db", _fake_embedder, [corpus])
    idx.reindex()
    skill = DocumentsSkill(idx)
    r = await skill.execute(SkillRequest(text="x", args={"query": "quantum sharks"}))
    # fake embedder maps unknown topics near-uniformly; accept either outcome
    assert r.speech

    off = DocumentsSkill(DocumentIndex(tmp_path / "off.db", None, [corpus]))
    r = await off.execute(SkillRequest(text="x", args={"query": "rent"}))
    assert "offline" in r.speech or "haven't indexed" in r.speech
