"""Long-term facts: store semantics + the remember/recall/forget skills."""

from __future__ import annotations

import pytest

from core.facts import FactsStore
from skills.base import SkillRequest
from skills.memory_skill import ForgetFactSkill, RecallFactsSkill, RememberFactSkill


@pytest.fixture
def store(tmp_path):
    return FactsStore(tmp_path / "facts.sqlite")


def test_roundtrip_and_nocase_dedupe(store):
    assert store.add("I like coffee") is True
    assert store.add("i like COFFEE") is False  # UNIQUE COLLATE NOCASE
    assert store.recent() == ["I like coffee"]


def test_normalization(store):
    assert store.add("  my   dog is called   Rex . ") is True
    assert store.recent() == ["my dog is called Rex"]
    assert store.add("") is False


def test_forget_substring_case_insensitive(store):
    store.add("My sister is Ana")
    store.add("My favourite drink is coffee")
    assert store.forget("SISTER") == 1
    assert store.count() == 1


def test_recent_returns_newest_capped_oldest_first(store):
    for i in range(25):
        store.add(f"fact number {i}")
    got = store.recent(20)
    assert len(got) == 20
    assert got[0] == "fact number 5"  # oldest of the newest 20 reads first
    assert got[-1] == "fact number 24"


def _req(skill, text):
    """Build a SkillRequest the way the router does: with the regex match."""
    return SkillRequest(text=text, match=skill.match(text))


async def test_remember_and_recall_skills(store):
    remember = RememberFactSkill(store)
    recall = RecallFactsSkill(store, limit=20)

    r = await remember.execute(_req(remember, "remember that my sister is Ana"))
    assert "remember" in r.speech.lower()
    assert store.recent() == ["my sister is Ana"]

    r = await remember.execute(_req(remember, "remember that my sister is Ana"))
    assert "already" in r.speech.lower()

    r = await recall.execute(_req(recall, "what do you remember about me"))
    assert "my sister is Ana" in r.speech


async def test_forget_skill_refuses_vague_pronouns(store):
    store.add("something precious")
    forget = ForgetFactSkill(store)
    r = await forget.execute(_req(forget, "forget it"))
    assert r.success is False
    assert store.count() == 1  # nothing deleted on a vague request

    r = await forget.execute(_req(forget, "forget about something precious"))
    assert r.success and store.count() == 0


async def test_remember_pattern_is_start_anchored():
    # "what do you remember" must not match RememberFactSkill's pattern.
    skill = RememberFactSkill(FactsStore(":memory:"))
    assert skill.match("what do you remember about me") is None
    assert skill.match("remember that I ski") is not None


# --- semantic recall (fake embedder: keyword axes, no Ollama needed) ---------

import numpy as np


def _fake_embedder(texts):
    """Deterministic 3-axis 'meaning': [dental, color, vehicle]."""
    out = []
    for t in texts:
        t = t.lower()
        out.append(np.array([
            1.0 if ("dentist" in t or "tooth" in t or "dental" in t) else 0.0,
            1.0 if ("color" in t or "blue" in t) else 0.0,
            1.0 if ("car" in t or "drive" in t) else 0.0,
        ], dtype=np.float32) + 0.01)
    return out


def test_relevant_ranks_semantically(tmp_path):
    store = FactsStore(tmp_path / "f.db", _fake_embedder)
    store.add("my dentist is Dr. Petrov, phone 070 123 456")
    store.add("my favorite color is blue")
    store.add("I drive a red Toyota")
    store.add("I was born in Skopje")
    store.add("my sister's name is Ana")

    got = store.relevant("when is my tooth appointment", limit=2)
    assert any("dentist" in f for f in got)          # semantic match wins
    # falls back cleanly to recency without an embedder
    plain = FactsStore(tmp_path / "f.db").relevant("tooth", limit=2)
    assert plain == FactsStore(tmp_path / "f.db").recent(2)


def test_relevant_blends_newest_and_survives_dead_embedder(tmp_path):
    store = FactsStore(tmp_path / "g.db", _fake_embedder)
    for i in range(6):
        store.add(f"fact number {i} about the color blue")
    store.add("my dentist is Dr. Novak")
    got = store.relevant("dental checkup", limit=4)
    assert any("dentist" in f for f in got)
    assert "my dentist is Dr. Novak" in got          # also among the newest 3

    dead = FactsStore(tmp_path / "g.db", lambda texts: None)  # Ollama down
    assert dead.relevant("dental checkup", limit=4) == dead.recent(4)


def test_add_backfills_embeddings(tmp_path):
    import sqlite3

    store = FactsStore(tmp_path / "h.db", _fake_embedder)
    store.add("I like espresso")
    with sqlite3.connect(tmp_path / "h.db") as conn:
        blob = conn.execute("SELECT embedding FROM facts").fetchone()[0]
    assert blob is not None and len(np.frombuffer(blob, dtype=np.float32)) == 3


def test_list_all_and_delete_by_id(tmp_path):
    store = FactsStore(tmp_path / "m.db")
    store.add("first fact")
    store.add("second fact")
    rows = store.list_all()
    assert [r["fact"] for r in rows] == ["first fact", "second fact"]
    assert all("id" in r and "created_at" in r for r in rows)
    assert store.delete(rows[0]["id"]) is True
    assert store.delete(9999) is False
    assert [r["fact"] for r in store.list_all()] == ["second fact"]
