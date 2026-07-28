"""Lightweight project + task tracking, so MEDO can help organize work.

A tiny sqlite-backed store — "start a project called kitchen remodel", "add a
task to it: get quotes", "what's left on the kitchen remodel", "mark get quotes
done". Persisted in the shared assistant database alongside notes and facts.

Every method opens a short-lived connection (like FactsStore/NoteStore), so
instances are safe to build anywhere; callers on the event loop wrap calls in
``asyncio.to_thread``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE COLLATE NOCASE,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        archived INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS project_tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        text TEXT NOT NULL,
        done INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        done_at TEXT,
        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
    )
    """,
)


def _clean(text: str) -> str:
    return " ".join((text or "").split()).strip(" .")


class ProjectStore:
    """Projects and their tasks, in the shared assistant database."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.execute("PRAGMA foreign_keys = ON")
        for stmt in _SCHEMA:
            conn.execute(stmt)
        return conn

    # -- projects -----------------------------------------------------------

    def create_project(self, name: str) -> bool:
        """Create a project. False if the name is empty or already exists."""
        name = _clean(name)
        if not name:
            return False
        with self._connect() as conn:
            try:
                conn.execute("INSERT INTO projects (name) VALUES (?)", (name,))
            except sqlite3.IntegrityError:      # UNIQUE COLLATE NOCASE duplicate
                return False
        return True

    def _project_id(self, conn: sqlite3.Connection, name: str) -> int | None:
        row = conn.execute(
            "SELECT id FROM projects WHERE name = ? COLLATE NOCASE", (name,)
        ).fetchone()
        return row[0] if row else None

    def project_exists(self, name: str) -> bool:
        with self._connect() as conn:
            return self._project_id(conn, _clean(name)) is not None

    def list_projects(self) -> list[dict]:
        """Every active project with its open/done task counts, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.name,
                       SUM(CASE WHEN t.done = 0 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN t.done = 1 THEN 1 ELSE 0 END)
                FROM projects p
                LEFT JOIN project_tasks t ON t.project_id = p.id
                WHERE p.archived = 0
                GROUP BY p.id
                ORDER BY p.id DESC
                """
            ).fetchall()
        return [{"name": r[0], "open": r[1] or 0, "done": r[2] or 0} for r in rows]

    def remove_project(self, name: str) -> bool:
        with self._connect() as conn:
            pid = self._project_id(conn, _clean(name))
            if pid is None:
                return False
            conn.execute("DELETE FROM project_tasks WHERE project_id = ?", (pid,))
            conn.execute("DELETE FROM projects WHERE id = ?", (pid,))
        return True

    # -- tasks --------------------------------------------------------------

    def add_task(self, project: str, text: str, create: bool = True) -> bool:
        """Add a task to ``project`` (created on demand). False on bad input or a
        missing project when ``create`` is False."""
        project, text = _clean(project), _clean(text)
        if not project or not text:
            return False
        with self._connect() as conn:
            pid = self._project_id(conn, project)
            if pid is None:
                if not create:
                    return False
                cur = conn.execute("INSERT INTO projects (name) VALUES (?)", (project,))
                pid = cur.lastrowid
            conn.execute(
                "INSERT INTO project_tasks (project_id, text) VALUES (?, ?)",
                (pid, text))
        return True

    def tasks(self, project: str, include_done: bool = True) -> list[dict] | None:
        """Tasks for ``project`` (open first, then by age), or None if no such
        project."""
        with self._connect() as conn:
            pid = self._project_id(conn, _clean(project))
            if pid is None:
                return None
            q = "SELECT id, text, done FROM project_tasks WHERE project_id = ?"
            if not include_done:
                q += " AND done = 0"
            q += " ORDER BY done ASC, id ASC"
            rows = conn.execute(q, (pid,)).fetchall()
        return [{"id": r[0], "text": r[1], "done": bool(r[2])} for r in rows]

    def complete_task(self, project: str, needle: str) -> str | None:
        """Mark the first OPEN task in ``project`` matching ``needle`` done.

        Returns the task's text, or None if nothing matched.
        """
        needle = _clean(needle)
        with self._connect() as conn:
            pid = self._project_id(conn, _clean(project))
            if pid is None:
                return None
            rows = conn.execute(
                "SELECT id, text FROM project_tasks "
                "WHERE project_id = ? AND done = 0 ORDER BY id ASC", (pid,)
            ).fetchall()
            hit = next((r for r in rows if needle.lower() in r[1].lower()), None)
            if hit is None:
                return None
            conn.execute(
                "UPDATE project_tasks SET done = 1, done_at = datetime('now') "
                "WHERE id = ?", (hit[0],))
        return hit[1]

    def summary(self, project: str) -> dict | None:
        """Counts + the next open tasks for ``project``, or None if missing."""
        rows = self.tasks(project)
        if rows is None:
            return None
        done = sum(1 for t in rows if t["done"])
        open_tasks = [t["text"] for t in rows if not t["done"]]
        return {
            "name": _clean(project), "total": len(rows), "done": done,
            "open": len(open_tasks), "next_tasks": open_tasks[:5],
        }
