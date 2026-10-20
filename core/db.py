"""One place that opens a SQLite database, so every store behaves the same.

MEDO keeps several small databases (facts + metrics + route memory in the
assistant DB, the document index, projects) and touches them from a lot of
directions at once: the router writes a metrics row while the HUD polls
``/status``, a routine fires on a timer, the document indexer walks a folder in
a thread, and the voice loop is doing all of this on one event loop.

SQLite handles that badly in its default configuration and well in this one:

* **WAL** — the rollback journal serializes readers against the writer, so a
  half-second index write blocked every read behind it and the HUD stuttered.
  Write-ahead logging lets readers carry on against the last committed
  snapshot while a write is in flight. This is the single biggest change here.
* **busy_timeout** — with WAL there is still exactly one writer at a time.
  Without a busy timeout the loser of that race gets an immediate
  ``database is locked``; with one it waits its turn. Python's ``timeout=``
  argument sets this too, but only for the connection it is passed, and it is
  routinely forgotten — doing it here means it can't be.
* **synchronous=NORMAL** — safe under WAL (a crash can lose the tail of the
  last transaction, never the database), and an order of magnitude fewer
  fsyncs than FULL. These are telemetry and notes, not ledgers.

On top of that, :func:`writing` serializes writers *inside this process* on a
per-file lock. WAL's own locking would make them queue anyway; doing it in
Python means they queue on a cheap in-process mutex instead of spinning on a
file lock and burning the busy timeout. Cross-process writers (the vision
sidecar) still fall back to the busy timeout, which is what it is for.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 10.0

#: One reentrant lock per database FILE, shared by every store that opens it.
#: Reentrant because a store's write helper may call another one that also
#: takes the lock (``record`` inside a ``prune``), and a plain Lock would
#: deadlock on itself the first time that happened.
_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()

#: Files whose WAL pragma has already been applied. journal_mode is persisted in
#: the database header, so this is purely to skip a redundant round-trip.
_wal_done: set[str] = set()


def _key(path: str | Path) -> str:
    """Canonical identity for a database file (two spellings, one lock)."""
    try:
        return str(Path(path).resolve())
    except OSError:                      # unresolvable path — fall back to text
        return str(path)


def lock_for(path: str | Path) -> threading.RLock:
    """The process-wide write lock for one database file."""
    key = _key(path)
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = _locks[key] = threading.RLock()
        return lock


def connect(path: str | Path, *, timeout: float = DEFAULT_TIMEOUT_S,
            check_same_thread: bool = True) -> sqlite3.Connection:
    """Open ``path`` with MEDO's standard pragmas applied.

    Never raises for a pragma that a filesystem refuses (WAL is unavailable on
    some network shares): the connection is still returned, just without the
    concurrency win. A database you can use slowly beats one you can't open.
    """
    conn = sqlite3.connect(str(path), timeout=timeout,
                           check_same_thread=check_same_thread)
    key = _key(path)
    with contextlib.suppress(sqlite3.Error):
        conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
    if key not in _wal_done:
        try:
            mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()
            if mode and str(mode[0]).lower() == "wal":
                _wal_done.add(key)
            else:
                logger.debug("WAL unavailable for %s (mode=%s)", path,
                             mode[0] if mode else "?")
        except sqlite3.Error as exc:
            logger.debug("could not enable WAL on %s: %s", path, exc)
    with contextlib.suppress(sqlite3.Error):
        conn.execute("PRAGMA synchronous=NORMAL")
    return conn


@contextlib.contextmanager
def writing(path: str | Path, *, timeout: float = DEFAULT_TIMEOUT_S,
            check_same_thread: bool = True) -> Iterator[sqlite3.Connection]:
    """A connection held under this file's write lock, committed on exit.

    Use for anything that writes; :func:`connect` alone is right for reads,
    which WAL lets run concurrently with the writer and with each other.
    """
    with lock_for(path):
        conn = connect(path, timeout=timeout, check_same_thread=check_same_thread)
        try:
            with conn:                       # commits, or rolls back on error
                yield conn
        finally:
            conn.close()


def checkpoint(path: str | Path) -> None:
    """Fold the WAL back into the main file (best-effort).

    Worth calling at a clean shutdown so the ``-wal`` sidecar doesn't grow
    unboundedly across a long-running install; never worth failing over.
    """
    with contextlib.suppress(Exception), lock_for(path):
        conn = connect(path)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
