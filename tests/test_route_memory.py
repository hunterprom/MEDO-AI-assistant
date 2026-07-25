"""Adaptive route memory store (M2.5e): learn, match, reinforce, evict.

Pure and offline — vectors are fabricated by hand (like _index in
test_semantic_router.py), no embedder or model. The router WIRING that decides
when to learn/consult is covered in test_semantic_router.py; here we prove the
store's own contract: a high-threshold cosine match with no runner-up margin,
hit reinforcement, latest-resolution-wins on a changed skill, embedder-space
isolation, the row cap, and prune/clear.
"""

from __future__ import annotations

import sqlite3

import numpy as np

from core.route_memory import RouteMemory


def _mem(tmp_path, embedder_id="modelA"):
    return RouteMemory(str(tmp_path / "mem.db"), embedder_id)


def _v(*xs):
    return np.array(xs, dtype=np.float32)


def _row(tmp_path, norm_text):
    conn = sqlite3.connect(str(tmp_path / "mem.db"))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT * FROM route_memory WHERE norm_text = ?", (norm_text,)).fetchone()
    finally:
        conn.close()


def test_remember_then_match_threshold(tmp_path):
    m = _mem(tmp_path)
    m.remember("is it toasty out", "weather", _v(1, 0, 0), now=1.0)
    hit = m.match(_v(0.99, 0.02, 0.0), 0.82)
    assert hit is not None and hit[0] == "weather" and hit[1] >= 0.82
    # A weak, off-axis query stays below the high gate → no shortcut.
    assert m.match(_v(0.6, 0.6, 0.0), 0.82) is None
    assert m.match(_v(0, 0, 0), 0.82) is None      # zero-norm query
    assert m.match(None, 0.82) is None


def test_same_phrasing_bumps_hits_no_duplicate(tmp_path):
    m = _mem(tmp_path)
    m.remember("do the thing", "weather", _v(1, 0, 0), now=1.0)
    m.remember("do the thing", "weather", _v(1, 0, 0), now=2.0)
    assert len(m) == 1
    row = _row(tmp_path, "do the thing")
    assert row["hits"] == 2 and row["skill"] == "weather" and row["last_used"] == 2.0


def test_changed_skill_supersedes_and_resets_hits(tmp_path):
    m = _mem(tmp_path)
    m.remember("do the thing", "weather", _v(1, 0, 0), now=1.0)
    m.remember("do the thing", "weather", _v(1, 0, 0), now=2.0)   # hits -> 2
    m.remember("do the thing", "news", _v(0, 1, 0), now=3.0)      # corrected
    assert len(m) == 1
    row = _row(tmp_path, "do the thing")
    assert row["skill"] == "news" and row["hits"] == 1            # latest wins, reset
    assert m.match(_v(0, 0.99, 0), 0.82)[0] == "news"


def test_embedder_id_isolates_vector_space(tmp_path):
    RouteMemory(str(tmp_path / "mem.db"), "modelA").remember(
        "phrase", "weather", _v(1, 0, 0), now=1.0)
    other = RouteMemory(str(tmp_path / "mem.db"), "modelB")   # different space
    assert len(other) == 0
    assert other.match(_v(1, 0, 0), 0.82) is None


def test_row_cap_evicts_least_used(tmp_path):
    m = _mem(tmp_path)
    m.remember("a", "weather", _v(1, 0, 0), now=1.0)           # hits 1
    m.remember("a", "weather", _v(1, 0, 0), now=2.0)           # hits 2 (kept)
    m.remember("b", "news", _v(0, 1, 0), now=3.0)             # hits 1, newer
    # Third distinct row over cap=2 evicts the lowest (hits, last_used) = "b".
    m.remember("c", "weather", _v(1, 0, 0), now=4.0, max_rows=2)
    assert len(m) == 2
    assert _row(tmp_path, "a") is not None                     # highest hits, kept
    assert _row(tmp_path, "b") is None                         # evicted


def test_prune_drops_dead_skills(tmp_path):
    m = _mem(tmp_path)
    m.remember("w", "weather", _v(1, 0, 0), now=1.0)
    m.remember("n", "news", _v(0, 1, 0), now=1.0)
    assert m.prune(["weather"]) == 1
    assert len(m) == 1 and m.match(_v(1, 0, 0), 0.82)[0] == "weather"


def test_forget_and_clear(tmp_path):
    m = _mem(tmp_path)
    m.remember("w", "weather", _v(1, 0, 0), now=1.0)
    m.remember("n", "news", _v(0, 1, 0), now=1.0)
    assert m.forget("weather") == 1 and len(m) == 1
    assert m.clear() == 1 and len(m) == 0
