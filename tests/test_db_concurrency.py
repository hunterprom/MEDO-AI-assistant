"""P1-3: the databases hold up when everything touches them at once.

MEDO writes to SQLite from a lot of directions simultaneously — the router
records a metrics row while the HUD polls /status, a routine fires on a timer,
the document indexer walks a folder in a worker thread, and a skill saves a
fact. In SQLite's default configuration that is a queue: the rollback journal
locks readers out for the duration of a write, and a reader that loses the race
gets ``database is locked`` immediately rather than waiting.

These tests are the evidence for the fix (WAL + a busy timeout + an in-process
write lock, all in :mod:`core.db`). They deliberately use REAL stores against a
REAL file — an in-memory database has none of the locking behaviour under test,
so mocking here would test nothing.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from core import db
from core.facts import FactsStore
from core.memory import NoteStore, ReminderStore
from core.metrics import MetricsStore
from core.projects import ProjectStore


# --- the pragmas are actually applied ----------------------------------------


def test_wal_is_enabled(tmp_path):
    conn = db.connect(tmp_path / "x.db")
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


def test_a_busy_timeout_is_set(tmp_path):
    conn = db.connect(tmp_path / "x.db", timeout=7.0)
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 7000
    finally:
        conn.close()


def test_synchronous_is_normal_not_full(tmp_path):
    """Safe under WAL, and an order of magnitude fewer fsyncs."""
    conn = db.connect(tmp_path / "x.db")
    try:
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1   # NORMAL
    finally:
        conn.close()


def test_wal_survives_reopening(tmp_path):
    """journal_mode lives in the file header — a second process inherits it."""
    path = tmp_path / "x.db"
    db.connect(path).close()
    conn = sqlite3.connect(str(path))
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


def test_the_same_file_shares_one_lock(tmp_path):
    """Two spellings of one path must not get two different locks."""
    path = tmp_path / "x.db"
    assert db.lock_for(path) is db.lock_for(str(path))
    assert db.lock_for(path) is db.lock_for(tmp_path / "." / "x.db")


def test_different_files_get_different_locks(tmp_path):
    assert db.lock_for(tmp_path / "a.db") is not db.lock_for(tmp_path / "b.db")


def test_the_write_lock_is_reentrant(tmp_path):
    """A write helper that calls another one must not deadlock on itself."""
    path = tmp_path / "x.db"
    with db.writing(path) as conn:
        conn.execute("CREATE TABLE t (v INTEGER)")
        with db.writing(path) as inner:
            inner.execute("INSERT INTO t VALUES (1)")
    conn = db.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1
    finally:
        conn.close()


def test_a_failed_write_rolls_back(tmp_path):
    path = tmp_path / "x.db"
    with db.writing(path) as conn:
        conn.execute("CREATE TABLE t (v INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO t VALUES (1)")
    with pytest.raises(sqlite3.IntegrityError):
        with db.writing(path) as conn:
            conn.execute("INSERT INTO t VALUES (2)")
            conn.execute("INSERT INTO t VALUES (1)")     # duplicate key
    conn = db.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1
    finally:
        conn.close()


def test_a_pragma_a_filesystem_refuses_is_not_fatal(tmp_path, monkeypatch):
    """WAL is unavailable on some network shares; a slow DB beats no DB."""

    class _RefusesWal(sqlite3.Connection):
        def execute(self, sql, *args):
            if "journal_mode" in sql.lower():
                raise sqlite3.OperationalError("disk I/O error")
            return super().execute(sql, *args)

    real_connect = sqlite3.connect

    def connect(path, **kw):
        return real_connect(path, factory=_RefusesWal, **kw)

    monkeypatch.setattr(db, "_wal_done", set())
    monkeypatch.setattr(db.sqlite3, "connect", connect)
    conn = db.connect(tmp_path / "x.db")
    conn.close()          # returned a usable connection instead of raising


# --- concurrent readers and writers, against real stores ---------------------


def test_concurrent_writers_all_land(tmp_path):
    """Sixteen threads, a hundred writes each. Nothing lost, nothing raised."""
    store = MetricsStore(tmp_path / "medo.db")
    errors: list[BaseException] = []

    def writer(worker: int) -> None:
        try:
            for i in range(100):
                store.record("FAST", f"skill{worker}", float(i))
        except BaseException as exc:      # noqa: BLE001 - report, don't swallow
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(writer, range(16)))

    assert errors == []
    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 1600


def test_readers_are_not_blocked_out_by_a_writer(tmp_path):
    """The WAL headline: reads keep succeeding while a write is in flight."""
    store = MetricsStore(tmp_path / "medo.db")
    store.record("FAST", "seed", 1.0)
    stop = threading.Event()
    reads = {"n": 0}
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                store.report()
                reads["n"] += 1
        except BaseException as exc:      # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        for i in range(300):
            store.record("LLM", "busy", float(i))
    finally:
        stop.set()
        thread.join(timeout=10)

    assert errors == []
    assert reads["n"] > 0


def test_several_stores_on_one_file_do_not_collide(tmp_path):
    """Facts, notes, reminders, projects and metrics share medo.db in practice."""
    path = tmp_path / "medo.db"
    facts = FactsStore(path, None)
    notes = NoteStore(path)
    reminders = ReminderStore(path)
    projects = ProjectStore(path)
    metrics = MetricsStore(path)
    errors: list[BaseException] = []

    def hammer(kind: str) -> None:
        try:
            for i in range(60):
                if kind == "facts":
                    facts.add(f"fact {i}")
                elif kind == "notes":
                    notes.add(f"note {i}")
                elif kind == "reminders":
                    reminders.add("2030-01-01T00:00:00", f"remind {i}")
                elif kind == "projects":
                    projects.add_task("roof", f"task {i}")
                else:
                    metrics.record("FAST", "s", float(i))
        except BaseException as exc:      # noqa: BLE001
            errors.append(exc)

    kinds = ["facts", "notes", "reminders", "projects", "metrics"] * 2
    with ThreadPoolExecutor(max_workers=len(kinds)) as pool:
        list(pool.map(hammer, kinds))

    assert errors == []
    assert facts.count() == 60                    # add() de-duplicates
    assert len(notes.list(limit=500)) == 120
    assert len(reminders.all()) == 120
    assert len(metrics._latencies()) == 120


def test_a_long_write_does_not_starve_a_reader(tmp_path):
    """A reader must not wait on a big transaction — it reads the last snapshot."""
    path = tmp_path / "medo.db"
    store = MetricsStore(path)
    store.record("FAST", "seed", 1.0)
    started = threading.Event()
    finish = threading.Event()

    def slow_writer() -> None:
        with db.writing(path) as conn:
            conn.execute("INSERT INTO metrics (path, skill, latency_ms) "
                         "VALUES ('LLM', 'slow', 1.0)")
            started.set()
            finish.wait(timeout=5)

    thread = threading.Thread(target=slow_writer, daemon=True)
    thread.start()
    try:
        assert started.wait(timeout=5)
        began = time.monotonic()
        rows = store._latencies()             # a read, mid-transaction
        elapsed = time.monotonic() - began
        assert rows == [1.0]                  # the committed snapshot
        assert elapsed < 2.0                  # did not sit on the busy timeout
    finally:
        finish.set()
        thread.join(timeout=10)


def test_the_route_memory_survives_concurrent_learning(tmp_path):
    """Route memory is written from the routing hot path, from many turns."""
    import numpy as np

    from core.route_memory import RouteMemory

    memory = RouteMemory(tmp_path / "medo.db", "test-embedder")
    errors: list[BaseException] = []

    def learn(worker: int) -> None:
        try:
            for i in range(40):
                memory.remember(f"phrase {worker} {i}", "weather",
                                np.ones(8, dtype=np.float32))
        except BaseException as exc:      # noqa: BLE001
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(learn, range(8)))
    assert errors == []


def test_checkpointing_keeps_the_data(tmp_path):
    path = tmp_path / "medo.db"
    store = MetricsStore(path)
    for i in range(50):
        store.record("FAST", "s", float(i))
    db.checkpoint(path)
    assert len(store._latencies()) == 50
