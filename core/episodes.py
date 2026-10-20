"""What we talked about yesterday — conversation memory that outlives a session.

MEDO's conversation memory is a ten-turn deque that dies with the process.
Within a session that is exactly right; across sessions it means every
conversation is the first one. "What did we discuss yesterday?" had no answer,
and neither did "you already told me that".

The obvious fix — keep every turn forever — is the wrong one twice over. It is
a privacy liability (a verbatim transcript of everything ever said in the room)
and it is useless for retrieval, because the thing worth recalling is the
*shape* of a conversation, not its stenography. So at the end of a session the
local model writes a few sentences about what happened, that summary is
embedded, and the turns themselves are not kept.

Three consequences worth stating plainly:

* **Recall is semantic.** "What did we decide about the roof?" finds the
  episode about guttering without either phrase matching, because the summaries
  are embedded with the same ``nomic-embed-text`` the rest of MEDO uses.
* **It degrades to nothing.** No model, no summary, no episode — recall then
  returns what it has, which may be nothing. Never an error, never a crash at
  shutdown, because shutdown is the worst possible place to raise.
* **It forgets on a schedule.** Episodes older than the retention window are
  pruned, unless pinned. A memory that only ever grows is a memory nobody can
  audit.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from core import db

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    ended_at    TEXT NOT NULL,
    summary     TEXT NOT NULL,
    turns       INTEGER NOT NULL DEFAULT 0,
    language    TEXT,
    pinned      INTEGER NOT NULL DEFAULT 0,
    embedder    TEXT,
    embedding   BLOB
);
CREATE INDEX IF NOT EXISTS idx_episodes_ended ON episodes(ended_at DESC);
"""

#: The instruction handed to the local model. Deliberately demanding about
#: FACTS: a summary that invents a decision nobody made is worse than no memory
#: at all, because it will be recalled later with total confidence.
SUMMARY_PROMPT = (
    "You are writing a short memory of a conversation, for an assistant to "
    "recall days later. Write three or four plain sentences covering: what the "
    "person wanted, what was decided or discovered, and anything left "
    "unfinished. Name specifics — people, places, numbers, file names — because "
    "those are what makes a memory findable later.\n\n"
    "Rules: only state what the transcript actually shows. Do not speculate "
    "about intent, do not invent decisions, and do not add advice. If the "
    "conversation was trivial (a clock check, a timer), say so in one sentence "
    "instead of padding it. Write in the language the conversation was in. No "
    "markdown, no bullet points, no preamble."
)

#: A session shorter than this is not a conversation, it is a command. Summarising
#: "what time is it" costs a model call and produces a memory nobody will ever
#: want, so those sessions are skipped entirely.
MIN_TURNS = 2

#: Cosine floor below which an episode is not an ANSWER to a topic question.
#:
#: Without it, "what did we say about quantum physics" returned the most recent
#: episode — "The day before yesterday: we talked about the roof" — which reads
#: as MEDO confirming a conversation that never happened. Falling back to
#: recency is right for "what did we talk about"; for a NAMED topic it
#: fabricates. 0.35 sits below a genuine topical match on nomic-embed-text
#: (0.5-0.7) and above an unrelated pair (~0.3).
RELEVANCE_FLOOR = 0.35


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class Episode:
    """One remembered conversation."""

    id: int
    started_at: str
    ended_at: str
    summary: str
    turns: int = 0
    language: str | None = None
    pinned: bool = False

    #: Weekday names. ``strftime('%A')`` is LOCALE-dependent and would return
    #: whatever the OS is set to, which is neither reliably English nor ever
    #: Macedonian — so both are spelled out here.
    _WEEKDAYS = {
        "en": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
               "Saturday", "Sunday"),
        "mk": ("понеделник", "вторник", "среда", "четврток", "петок",
               "сабота", "недела"),
    }

    def when(self, now: datetime | None = None, language: str = "en") -> str:
        """"yesterday", "on Tuesday", "three weeks ago" — spoken, not ISO.

        Bilingual, because the caller splices this straight into a sentence: an
        English "on saturday" in front of a Macedonian summary is the mixed-
        language reply this project keeps having to fix.
        """
        now = now or datetime.now().astimezone()
        mk = language == "mk"
        try:
            ended = datetime.fromisoformat(self.ended_at)
        except ValueError:
            return "во некој момент" if mk else "at some point"
        if ended.tzinfo is None:
            ended = ended.replace(tzinfo=now.tzinfo)
        days = (now.date() - ended.date()).days
        weekday = self._WEEKDAYS["mk" if mk else "en"][ended.weekday()]
        if days <= 0:
            return "порано денес" if mk else "earlier today"
        if days == 1:
            return "вчера" if mk else "yesterday"
        if days == 2:
            return "завчера" if mk else "the day before yesterday"
        if days < 7:
            return f"во {weekday}" if mk else f"on {weekday}"
        if days < 14:
            return "минатата недела" if mk else "last week"
        if days < 60:
            weeks = days // 7
            if mk:
                return f"пред {weeks} недели"
            return f"{weeks} weeks ago"
        months = max(1, days // 30)
        if mk:
            return f"пред {months} месеци" if months > 1 else "пред еден месец"
        return f"{months} months ago"


def transcript(turns: Sequence[tuple[str, str]], limit: int = 60) -> str:
    """The conversation as plain text for the summariser.

    Bounded from the END: a long session's most recent turns are the ones a
    summary should be about, and the whole thing would not fit in the local
    brain's context window anyway.
    """
    kept = list(turns)[-limit:]
    lines = []
    for user, assistant in kept:
        lines.append(f"Person: {user.strip()}")
        lines.append(f"Assistant: {assistant.strip()}")
    return "\n".join(lines)


class EpisodeStore:
    """Summarised past conversations, searchable by meaning."""

    def __init__(self, db_path: str | Path, embedder=None,
                 embedder_id: str = "none") -> None:
        self._db_path = str(db_path)
        self._embedder = embedder
        self._embedder_id = embedder_id or "none"

    def _connect(self) -> sqlite3.Connection:
        conn = db.connect(self._db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        return conn

    # -- writing ---------------------------------------------------------------

    def add(self, summary: str, *, started_at: str | None = None,
            ended_at: str | None = None, turns: int = 0,
            language: str | None = None, pinned: bool = False) -> int | None:
        """Store one episode, embedding its summary. Returns the row id."""
        summary = (summary or "").strip()
        if not summary:
            return None
        vector = self._embed(summary)
        now = _now_iso()
        try:
            with db.writing(self._db_path, timeout=10.0) as conn:
                conn.executescript(_SCHEMA)
                cur = conn.execute(
                    "INSERT INTO episodes (started_at, ended_at, summary, turns, "
                    "language, pinned, embedder, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (started_at or now, ended_at or now, summary, int(turns),
                     language, 1 if pinned else 0,
                     self._embedder_id if vector is not None else None,
                     vector.tobytes() if vector is not None else None))
                return int(cur.lastrowid)
        except Exception:  # noqa: BLE001 - a memory that fails to save must not
            logger.warning("could not store the session summary", exc_info=True)
            return None

    def _embed(self, text: str):
        if self._embedder is None:
            return None
        try:
            vectors = self._embedder([text])
        except Exception:  # noqa: BLE001
            logger.debug("episode embedding failed", exc_info=True)
            return None
        if not vectors:
            return None
        return np.asarray(vectors[0], dtype=np.float32)

    def pin(self, episode_id: int, pinned: bool = True) -> bool:
        """Protect an episode from the retention sweep (or release it)."""
        try:
            with db.writing(self._db_path, timeout=10.0) as conn:
                conn.executescript(_SCHEMA)
                cur = conn.execute("UPDATE episodes SET pinned = ? WHERE id = ?",
                                   (1 if pinned else 0, int(episode_id)))
                return (cur.rowcount or 0) > 0
        except Exception:  # noqa: BLE001
            return False

    def delete(self, episode_id: int) -> bool:
        try:
            with db.writing(self._db_path, timeout=10.0) as conn:
                conn.executescript(_SCHEMA)
                cur = conn.execute("DELETE FROM episodes WHERE id = ?",
                                   (int(episode_id),))
                return (cur.rowcount or 0) > 0
        except Exception:  # noqa: BLE001
            return False

    def prune(self, keep_days: int, *, keep_pinned: bool = True) -> int:
        """Forget episodes older than ``keep_days``. Returns how many went.

        Pinned episodes survive by default — that is what pinning is for. A
        ``keep_days`` of 0 or less disables the sweep entirely, which is a
        supported choice and a deliberate one: it means the summaries of every
        conversation ever held in this room stay on disk.
        """
        if keep_days <= 0:
            return 0
        cutoff = (datetime.now().astimezone()
                  - timedelta(days=int(keep_days))).isoformat(timespec="seconds")
        sql = "DELETE FROM episodes WHERE ended_at < ?"
        if keep_pinned:
            sql += " AND pinned = 0"
        try:
            with db.writing(self._db_path, timeout=10.0) as conn:
                conn.executescript(_SCHEMA)
                return conn.execute(sql, (cutoff,)).rowcount or 0
        except Exception:  # noqa: BLE001
            return 0

    # -- reading ---------------------------------------------------------------

    def _rows(self, where: str = "", args: tuple = (),
              limit: int = 200) -> list[sqlite3.Row]:
        try:
            with self._connect() as conn:
                return conn.execute(
                    "SELECT id, started_at, ended_at, summary, turns, language, "
                    f"pinned, embedder, embedding FROM episodes {where} "  # noqa: S608
                    "ORDER BY ended_at DESC LIMIT ?", (*args, int(limit))).fetchall()
        except Exception:  # noqa: BLE001
            return []

    @staticmethod
    def _episode(row: sqlite3.Row) -> Episode:
        return Episode(id=row["id"], started_at=row["started_at"],
                       ended_at=row["ended_at"], summary=row["summary"],
                       turns=row["turns"], language=row["language"],
                       pinned=bool(row["pinned"]))

    def recent(self, limit: int = 10) -> list[Episode]:
        return [self._episode(r) for r in self._rows(limit=limit)]

    def count(self) -> int:
        try:
            with self._connect() as conn:
                return int(conn.execute(
                    "SELECT COUNT(*) FROM episodes").fetchone()[0])
        except Exception:  # noqa: BLE001
            return 0

    def since(self, start: datetime, end: datetime | None = None,
              limit: int = 20) -> list[Episode]:
        """Episodes that ended in a time window — "yesterday", "last week"."""
        args = [start.isoformat(timespec="seconds")]
        where = "WHERE ended_at >= ?"
        if end is not None:
            where += " AND ended_at < ?"
            args.append(end.isoformat(timespec="seconds"))
        return [self._episode(r) for r in self._rows(where, tuple(args), limit)]

    def search(self, query: str, limit: int = 3,
               window: tuple[datetime, datetime | None] | None = None,
               require_relevance: bool = False) -> list[Episode]:
        """The most relevant episodes for ``query``, best first.

        Semantic when an embedder is available, falling back to a substring
        match and then to plain recency. The fallbacks matter: recall that
        answers "nothing" because Ollama is cold is indistinguishable, to the
        person asking, from an assistant that has forgotten them.

        ``require_relevance`` switches the last fallback off. Set it when the
        user NAMED a topic: "what did we say about the roof" must answer
        nothing rather than hand back the most recent conversation, which the
        listener will hear as "yes, we discussed the roof". Broad questions
        ("what did we talk about") leave it False, where recency is the right
        answer rather than a guess.
        """
        where, args = "", ()
        if window is not None:
            start, end = window
            where = "WHERE ended_at >= ?"
            args = (start.isoformat(timespec="seconds"),)
            if end is not None:
                where += " AND ended_at < ?"
                args = (*args, end.isoformat(timespec="seconds"))
        rows = self._rows(where, args)
        if not rows:
            return []
        query_vector = self._embed(query) if query.strip() else None
        if query_vector is not None:
            scored = self._by_similarity(rows, query_vector)
            if scored:
                if require_relevance:
                    scored = [pair for pair in scored
                              if pair[1] >= RELEVANCE_FLOOR]
                    # The embeddings WERE usable and nothing was close enough.
                    # That is a real answer — "no, we didn't discuss that" — and
                    # returning it beats inventing one.
                    return [self._episode(r) for r, _ in scored[:limit]]
                return [self._episode(r) for r, _ in scored[:limit]]
        needle = query.strip().lower()
        if needle:
            hits = [r for r in rows if needle in (r["summary"] or "").lower()]
            if hits:
                return [self._episode(r) for r in hits[:limit]]
        if require_relevance:
            return []
        return [self._episode(r) for r in rows[:limit]]

    def _by_similarity(self, rows, query_vector):
        """(row, score) for rows with a usable vector, most similar first."""
        q = np.asarray(query_vector, dtype=np.float32).ravel()
        qn = float(np.linalg.norm(q))
        if not np.isfinite(qn) or qn == 0.0:
            return []
        q = q / qn
        scored = []
        for row in rows:
            blob, embedder = row["embedding"], row["embedder"]
            # A vector from a DIFFERENT embed model lives in a different space;
            # comparing them produces confident nonsense, so those rows are
            # skipped rather than mixed in.
            if not blob or embedder != self._embedder_id:
                continue
            v = np.frombuffer(blob, dtype=np.float32)
            vn = float(np.linalg.norm(v))
            if not np.isfinite(vn) or vn == 0.0 or v.shape != q.shape:
                continue
            score = float(q @ (v / vn))
            if np.isfinite(score):
                scored.append((row, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored


async def summarize_session(turns: Sequence[tuple[str, str]], ask,
                            *, min_turns: int = MIN_TURNS) -> str:
    """Ask the local model for a memory of this conversation. "" when skipped.

    ``ask`` is ``async (system: str, user: str) -> str`` — the same shape the
    council uses. Never raises: this runs at shutdown, where an exception would
    replace a clean exit with a traceback over a feature nobody asked for.
    """
    if ask is None or len(turns) < min_turns:
        return ""
    text = transcript(turns)
    if not text.strip():
        return ""
    try:
        summary = await ask(SUMMARY_PROMPT, text)
    except Exception:  # noqa: BLE001
        logger.warning("could not summarise the session", exc_info=True)
        return ""
    return " ".join((summary or "").split()).strip()
