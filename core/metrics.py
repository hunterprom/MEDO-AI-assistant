"""Latency instrumentation.

Each turn records per-stage timings; :class:`LatencyLog` keeps them and computes
averages against the budgets in the README. Kept separate from the UI so it can be
unit-tested and queried (``/latency`` in the REPL).

:class:`MetricsStore` persists one row per routed request (path, skill, latency)
into the existing assistant sqlite DB, and renders the README's Performance
table from whatever has accumulated:

    python -m core.metrics --report
"""

from __future__ import annotations

import contextlib
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from core import db
from core.facts import _like_escape


@dataclass
class TurnTimings:
    """Milliseconds spent in each stage of one interaction (None = not run)."""

    path: str = ""                       # "FAST" or "LLM"
    wake_to_listen_ms: float | None = None
    stt_ms: float | None = None
    route_ms: float | None = None        # LLM path: includes token generation
    tts_ms: float | None = None          # time to synthesize the reply (pre-play)
    #: End of the user's speech to the FIRST audio out of the speakers. The
    #: number sentence-streaming exists to reduce, and the one a person actually
    #: experiences as "how long before it answers" — ``route_ms`` measures the
    #: whole generation, which streaming deliberately decoupled from the wait.
    #: None on a turn that never streamed (fast path, tool answers).
    first_audio_ms: float | None = None

    def spoken_ms(self) -> float:
        """Rough time from end-of-speech to first audio out (STT + route + TTS)."""
        return sum(v for v in (self.stt_ms, self.route_ms, self.tts_ms) if v)


#: Budgets from the project spec (README latency table), in milliseconds.
BUDGETS = {
    "wake_to_listen_ms": 500.0,
    "fast_route_ms": 1000.0,
    "llm_spoken_ms": 4000.0,
}


class LatencyLog:
    """Collects :class:`TurnTimings` and reports averages by stage."""

    def __init__(self) -> None:
        self.turns: list[TurnTimings] = []

    def record(self, timings: TurnTimings) -> None:
        self.turns.append(timings)

    def _avg(self, attr: str, path: str | None = None) -> float | None:
        vals = [
            getattr(t, attr)
            for t in self.turns
            if getattr(t, attr) is not None and (path is None or t.path == path)
        ]
        return sum(vals) / len(vals) if vals else None

    def summary(self) -> dict[str, float | None]:
        """Average per-stage latencies (ms) across recorded turns."""
        return {
            "count": float(len(self.turns)),
            "wake_to_listen_ms": self._avg("wake_to_listen_ms"),
            "fast_route_ms": self._avg("route_ms", path="FAST"),
            "stt_ms": self._avg("stt_ms"),
            "tts_ms": self._avg("tts_ms"),
            "llm_route_ms": self._avg("route_ms", path="LLM"),
            "first_audio_ms": self._avg("first_audio_ms"),
        }


# --- persisted routing metrics (README Performance table) --------------------

_METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL DEFAULT (datetime('now')),
    path TEXT NOT NULL,
    skill TEXT,
    latency_ms REAL NOT NULL
)
"""

#: Time to FIRST AUDIO, per turn. Separate from the metrics table because it is
#: a different question: ``latency_ms`` is how long the answer took to produce,
#: this is how long the person waited before hearing anything. Sentence
#: streaming widened the gap between those two on purpose, so measuring only the
#: first stopped describing the experience.
_FIRST_AUDIO_SCHEMA = """
CREATE TABLE IF NOT EXISTS first_audio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL DEFAULT (datetime('now')),
    path TEXT NOT NULL,
    streamed INTEGER NOT NULL DEFAULT 0,
    first_audio_ms REAL NOT NULL,
    total_ms REAL
)
"""

#: Refused requests to the companion API. Recorded so a token being ground
#: against from the LAN is *visible* rather than merely blocked — the block is
#: in remote/server.py, this is the evidence.
#:
#: ``caller`` is the throttling identity from :func:`remote.ratelimit.caller_key`
#: — a hash prefix for a token, an address for an anonymous peer. The token
#: itself is deliberately absent: a rejected credential is still a credential,
#: and writing it to a database that gets copied into bug reports would turn a
#: failed attack into a successful one.
_AUTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS auth_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL DEFAULT (datetime('now')),
    peer TEXT,
    path TEXT,
    method TEXT,
    reason TEXT NOT NULL,
    caller TEXT
)
"""

#: Every fast-path (regex) match, and whether the user disowned it seconds
#: later. This is the ground truth for "does the FAST tier misfire?", which
#: nothing else in MEDO could answer: a wrong regex route produces a confident,
#: fast, wrong answer that looks exactly like a right one in the latency table.
_ROUTE_SCHEMA = """
CREATE TABLE IF NOT EXISTS route_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL DEFAULT (datetime('now')),
    tier TEXT NOT NULL,
    skill TEXT,
    text TEXT,
    confidence REAL,
    corrected INTEGER NOT NULL DEFAULT 0,
    correction TEXT,
    reply TEXT,
    latency_ms REAL
)
"""

#: Columns added after the table shipped. SQLite has no "ADD COLUMN IF NOT
#: EXISTS", and an existing install must not have to delete its history to get
#: the dashboard, so each is added in place and the duplicate error swallowed.
_ROUTE_MIGRATIONS = (
    "ALTER TABLE route_events ADD COLUMN reply TEXT",
    "ALTER TABLE route_events ADD COLUMN latency_ms REAL",
)

#: Council picks where a second opinion disagreed with the deterministic word
#: list. Feeds the decision to move selection onto embeddings, and then proves
#: whether the move helped.
_COUNCIL_SCHEMA = """
CREATE TABLE IF NOT EXISTS council_selections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL DEFAULT (datetime('now')),
    question TEXT,
    method TEXT NOT NULL,
    chosen TEXT,
    alternative TEXT,
    agreed INTEGER NOT NULL DEFAULT 1
)
"""

#: Every time MEDO spoke without being asked. Recorded because the whole risk of
#: a proactive layer is that it says too much: this is how you find out, rather
#: than waiting for someone to get annoyed enough to switch it off.
_PROACTIVE_SCHEMA = """
CREATE TABLE IF NOT EXISTS proactive_fires (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL DEFAULT (datetime('now')),
    trigger TEXT NOT NULL
)
"""

_ALL_SCHEMAS = (_METRICS_SCHEMA, _AUTH_SCHEMA, _ROUTE_SCHEMA, _COUNCIL_SCHEMA,
                _FIRST_AUDIO_SCHEMA, _PROACTIVE_SCHEMA)


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile of an ascending list (q in 0..100)."""
    if not sorted_vals:
        return 0.0
    rank = max(1, math.ceil(q / 100 * len(sorted_vals)))
    return sorted_vals[min(rank, len(sorted_vals)) - 1]


class MetricsStore:
    """One sqlite row per routed request, in the shared assistant DB.

    Same idiom as :class:`~core.facts.FactsStore`: short-lived connection per
    operation, safe to construct anywhere. ``record`` is best-effort — a
    locked/corrupt DB must never cost a routed reply.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = db.connect(self._db_path, timeout=5.0)
        for schema in _ALL_SCHEMAS:
            conn.execute(schema)
        self._migrate(conn)
        return conn

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add columns that post-date the table. Idempotent and quiet."""
        for statement in _ROUTE_MIGRATIONS:
            try:
                conn.execute(statement)
            except sqlite3.OperationalError:
                pass                      # already there

    @contextlib.contextmanager
    def _writing(self):
        """A committed, write-locked connection with every table ensured."""
        with db.writing(self._db_path, timeout=5.0) as conn:
            for schema in _ALL_SCHEMAS:
                conn.execute(schema)
            self._migrate(conn)
            yield conn

    def record(self, path: str, skill: str | None, latency_ms: float) -> None:
        try:
            with self._writing() as conn:
                conn.execute(
                    "INSERT INTO metrics (path, skill, latency_ms) VALUES (?, ?, ?)",
                    (path, skill, float(latency_ms)),
                )
        except Exception:  # noqa: BLE001 - telemetry must never break routing
            pass

    def record_first_audio(self, path: str, first_audio_ms: float, *,
                           streamed: bool = False,
                           total_ms: float | None = None) -> None:
        """Record how long this turn took to make its first sound."""
        try:
            with self._writing() as conn:
                conn.execute(
                    "INSERT INTO first_audio (path, streamed, first_audio_ms, "
                    "total_ms) VALUES (?, ?, ?, ?)",
                    (str(path), 1 if streamed else 0, float(first_audio_ms),
                     None if total_ms is None else float(total_ms)))
        except Exception:  # noqa: BLE001 - telemetry must never break a reply
            pass

    def first_audio_report(self) -> str:
        """Time-to-first-audio, streamed vs not — what streaming actually bought."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT streamed, COUNT(*), AVG(first_audio_ms), "
                    "AVG(total_ms) FROM first_audio GROUP BY streamed").fetchall()
        except Exception as exc:  # noqa: BLE001
            return f"could not read first-audio metrics: {exc}"
        if not rows:
            return "No first-audio timings recorded yet — talk to MEDO in voice mode."
        out = ["| Reply | Turns | Mean time to first audio | Mean total |",
               "|-------|------:|-------------------------:|-----------:|"]
        for streamed, n, first, total in rows:
            label = "streamed" if streamed else "whole"
            out.append(f"| {label} | {n} | {first:,.0f} ms "
                       f"| {'-' if total is None else f'{total:,.0f} ms'} |")
        return "\n".join(out)

    def record_proactive(self, trigger: str) -> None:
        """Record one unprompted announcement."""
        try:
            with self._writing() as conn:
                conn.execute("INSERT INTO proactive_fires (trigger) VALUES (?)",
                             (str(trigger),))
        except Exception:  # noqa: BLE001
            pass

    def proactive_report(self) -> str:
        """How often MEDO spoke first, by trigger."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT trigger, COUNT(*) AS n, MAX(ts) FROM proactive_fires "
                    "GROUP BY trigger ORDER BY n DESC").fetchall()
        except Exception as exc:  # noqa: BLE001
            return f"could not read proactive fires: {exc}"
        if not rows:
            return "MEDO has not spoken unprompted."
        out = ["| Trigger | Fires | Last |", "|---------|------:|------|"]
        out += [f"| {trigger} | {n} | {last} |" for trigger, n, last in rows]
        return "\n".join(out)

    # --- refused API requests (remote/server.py) -----------------------------

    def record_auth_event(self, reason: str, *, peer: str | None = None,
                          path: str | None = None, method: str | None = None,
                          caller: str | None = None) -> None:
        """Record one refused companion-API request. Never raises.

        ``reason`` is a short stable slug (``bad_token``, ``no_token``,
        ``rate_limited``, ``cross_origin``) so the report can group by it.
        """
        try:
            with self._writing() as conn:
                conn.execute(
                    "INSERT INTO auth_events (peer, path, method, reason, caller) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (peer, path, method, str(reason), caller),
                )
        except Exception:  # noqa: BLE001 - security telemetry, not a hard dep
            pass

    def auth_failures(self, limit: int = 50) -> list[dict]:
        """Most recent refusals, newest first (HUD + ``--auth`` report)."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT ts, peer, path, method, reason, caller FROM auth_events "
                    "ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        except Exception:  # noqa: BLE001
            return []
        keys = ("ts", "peer", "path", "method", "reason", "caller")
        return [dict(zip(keys, row)) for row in rows]

    def auth_report(self) -> str:
        """Refused API requests, grouped — ``python -m core.metrics --auth``."""
        try:
            with self._connect() as conn:
                by_reason = conn.execute(
                    "SELECT reason, COUNT(*) AS n FROM auth_events "
                    "GROUP BY reason ORDER BY n DESC").fetchall()
                by_peer = conn.execute(
                    "SELECT peer, COUNT(*) AS n, MAX(ts) FROM auth_events "
                    "GROUP BY peer ORDER BY n DESC LIMIT 10").fetchall()
        except Exception as exc:  # noqa: BLE001
            return f"could not read auth events: {exc}"
        if not by_reason:
            return "No refused API requests recorded — nothing has been turned away."
        out = ["| Reason | Count |", "|--------|------:|"]
        out += [f"| {reason} | {n} |" for reason, n in by_reason]
        out += ["", "| Peer | Refusals | Last seen |", "|------|---------:|-----------|"]
        out += [f"| {peer or '?'} | {n} | {last} |" for peer, n, last in by_peer]
        return "\n".join(out)

    # --- routing telemetry (fast-path misfires, council disagreements) -------

    def record_route_event(self, tier: str, skill: str | None, text: str,
                           confidence: float | None = None,
                           reply: str | None = None,
                           latency_ms: float | None = None) -> int | None:
        """Log one routing decision; returns its row id (for a later correction).

        The utterance is stored because a misfire report that says only
        "the timer skill fired wrongly 4 times" can't be acted on — the whole
        point is to see *which phrasings* trip it. This is the same local
        database the conversation already lives in; nothing leaves the machine.
        """
        try:
            with self._writing() as conn:
                cur = conn.execute(
                    "INSERT INTO route_events (tier, skill, text, confidence, "
                    "reply, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
                    (str(tier), skill, (text or "")[:500],
                     None if confidence is None else float(confidence),
                     None if reply is None else reply[:2000],
                     None if latency_ms is None else float(latency_ms)),
                )
                return int(cur.lastrowid)
        except Exception:  # noqa: BLE001 - telemetry must never break routing
            return None

    def mark_route_corrected(self, event_id: int, correction: str) -> None:
        """Flag a logged route as one the user immediately took back."""
        try:
            with self._writing() as conn:
                conn.execute(
                    "UPDATE route_events SET corrected = 1, correction = ? "
                    "WHERE id = ?", ((correction or "")[:500], int(event_id)))
        except Exception:  # noqa: BLE001
            pass

    def history(self, query: str = "", tier: str = "", limit: int = 100,
                offset: int = 0) -> list[dict]:
        """Past turns, newest first — the dashboard's only data source.

        ``query`` matches the utterance OR the reply, case-insensitively;
        ``tier`` narrows to one routing tier. Both are optional and both are
        bound parameters, never interpolated.
        """
        where, args = [], []
        if query.strip():
            where.append(r"(LOWER(text) LIKE ? ESCAPE '\' "
                         r"OR LOWER(reply) LIKE ? ESCAPE '\')")
            needle = f"%{_like_escape(query.strip().lower())}%"
            args += [needle, needle]
        if tier.strip():
            where.append("tier = ?")
            args.append(tier.strip().upper())
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT id, ts, tier, skill, text, reply, confidence, "
                    f"corrected, correction, latency_ms FROM route_events {clause} "  # noqa: S608
                    "ORDER BY id DESC LIMIT ? OFFSET ?",
                    (*args, int(limit), int(offset))).fetchall()
        except Exception:  # noqa: BLE001
            return []
        keys = ("id", "ts", "tier", "skill", "text", "reply", "confidence",
                "corrected", "correction", "latency_ms")
        return [dict(zip(keys, row)) for row in rows]

    def history_tiers(self) -> list[str]:
        """Tiers that actually appear in the history, for the filter control."""
        try:
            with self._connect() as conn:
                return [r[0] for r in conn.execute(
                    "SELECT DISTINCT tier FROM route_events ORDER BY tier"
                ).fetchall()]
        except Exception:  # noqa: BLE001
            return []

    def record_council_selection(self, question: str, method: str,
                                 chosen: str | None, alternative: str | None,
                                 agreed: bool) -> None:
        """Log who the council picked, and who a second method would have picked."""
        try:
            with self._writing() as conn:
                conn.execute(
                    "INSERT INTO council_selections "
                    "(question, method, chosen, alternative, agreed) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ((question or "")[:500], str(method), chosen, alternative,
                     1 if agreed else 0),
                )
        except Exception:  # noqa: BLE001
            pass

    def routing_report(self) -> str:
        """Per-skill fast-path misfire rate + council disagreement rate."""
        try:
            with self._connect() as conn:
                # FAST/SEMANTIC only: an LLM turn has no pattern to blame, so
                # a misfire RATE for it would mean something quite different and
                # would dilute the number this report exists to show. The rows
                # are still stored — the dashboard shows every tier.
                rows = conn.execute(
                    "SELECT tier, skill, COUNT(*) AS n, SUM(corrected) AS bad "
                    "FROM route_events WHERE tier IN ('FAST', 'SEMANTIC') "
                    "GROUP BY tier, skill "
                    "ORDER BY bad DESC, n DESC LIMIT 20").fetchall()
                worst = conn.execute(
                    "SELECT skill, text, correction FROM route_events "
                    "WHERE corrected = 1 ORDER BY id DESC LIMIT 10").fetchall()
                council = conn.execute(
                    "SELECT method, COUNT(*) AS n, SUM(1 - agreed) AS disagreed "
                    "FROM council_selections GROUP BY method").fetchall()
        except Exception as exc:  # noqa: BLE001
            return f"could not read routing telemetry: {exc}"
        if not rows and not council:
            return ("No routing telemetry recorded yet — set "
                    "logging.route_telemetry: true and talk to MEDO.")
        out: list[str] = []
        if rows:
            out += ["| Tier | Skill | Routes | Corrected | Rate |",
                    "|------|-------|-------:|----------:|-----:|"]
            for tier, skill, n, bad in rows:
                bad = int(bad or 0)
                out.append(f"| {tier} | {skill or '?'} | {n} | {bad} "
                           f"| {bad / n * 100:.1f}% |")
        if worst:
            out += ["", "Most recent corrections:", ""]
            out += [f"- `{text}` -> **{skill}**, corrected by: `{corr}`"
                    for skill, text, corr in worst]
        if council:
            out += ["", "| Council method | Picks | Disagreements | Rate |",
                    "|----------------|------:|--------------:|-----:|"]
            for method, n, disagreed in council:
                disagreed = int(disagreed or 0)
                out.append(f"| {method} | {n} | {disagreed} "
                           f"| {disagreed / n * 100:.1f}% |")
        return "\n".join(out)

    def prune(self, keep_days: int = 90) -> int:
        """Drop telemetry older than ``keep_days``. Returns rows deleted.

        Route events carry the text of what was said, so they are the one table
        here with a privacy cost to unbounded growth — this is what bounds it.
        """
        deleted = 0
        try:
            with self._writing() as conn:
                for table in ("metrics", "auth_events", "route_events",
                              "council_selections", "first_audio",
                              "proactive_fires"):
                    cur = conn.execute(
                        f"DELETE FROM {table} "  # noqa: S608 - fixed table names
                        f"WHERE ts < datetime('now', ?)", (f"-{int(keep_days)} days",))
                    deleted += cur.rowcount or 0
        except Exception:  # noqa: BLE001
            return deleted
        return deleted

    def _latencies(self, where: str = "", args: tuple = ()) -> list[float]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT latency_ms FROM metrics {where} ORDER BY latency_ms", args
            ).fetchall()
        return [r[0] for r in rows]

    def report(self) -> str:
        """The README Performance section as markdown (see --report)."""
        with self._connect() as conn:
            per_path = conn.execute(
                "SELECT path, COUNT(*) FROM metrics GROUP BY path"
            ).fetchall()
            top_skills = conn.execute(
                "SELECT skill, COUNT(*) AS n FROM metrics WHERE skill IS NOT NULL "
                "GROUP BY skill ORDER BY n DESC, skill LIMIT 10"
            ).fetchall()
        total = sum(n for _, n in per_path)
        if total == 0:
            return ("No routing metrics recorded yet — talk to MEDO first "
                    "(logging.routing_stats: true), then re-run this report.")

        counts = dict(per_path)
        lines = [
            "| Path | Requests | Share | p50 | p95 |",
            "|------|---------:|------:|--------:|--------:|",
        ]
        for path in ("FAST", "LLM"):
            n = counts.get(path, 0)
            if n == 0:
                continue
            lat = self._latencies("WHERE path = ?", (path,))
            lines.append(
                f"| {path} | {n} | {n / total * 100:.1f}% "
                f"| {_percentile(lat, 50):,.0f} ms | {_percentile(lat, 95):,.0f} ms |"
            )
        out = [f"{total} routed request{'s' if total != 1 else ''} measured.", "", *lines]

        if top_skills:
            out += ["", "Top skills by usage:", "",
                    "| Skill | Requests | p50 |", "|-------|---------:|--------:|"]
            for skill, n in top_skills:
                lat = self._latencies("WHERE skill = ?", (skill,))
                out.append(f"| {skill} | {n} | {_percentile(lat, 50):,.0f} ms |")
        return "\n".join(out)


def main() -> None:
    """``python -m core.metrics --report`` — print the Performance table."""
    import argparse

    from core.config import load_settings

    parser = argparse.ArgumentParser(description="MEDO routing metrics")
    parser.add_argument("--report", action="store_true",
                        help="print the README Performance table (markdown)")
    parser.add_argument("--auth", action="store_true",
                        help="print refused companion-API requests, grouped")
    parser.add_argument("--routing", action="store_true",
                        help="print fast-path misfires + council disagreements")
    parser.add_argument("--prune", type=int, metavar="DAYS", default=None,
                        help="delete telemetry older than DAYS and exit")
    parser.add_argument("--first-audio", action="store_true",
                        help="print time-to-first-audio, streamed vs whole")
    parser.add_argument("--proactive", action="store_true",
                        help="print how often MEDO spoke without being asked")
    parser.add_argument("--db", default=None,
                        help="database path (default: memory.db_path from config.yaml)")
    args = parser.parse_args()
    if not (args.report or args.auth or args.routing or args.first_audio
            or args.proactive or args.prune is not None):
        parser.print_help()
        return
    db_path = args.db or load_settings().memory.db_path
    store = MetricsStore(db_path)
    if args.prune is not None:
        print(f"pruned {store.prune(args.prune)} row(s) older than {args.prune} days")
        return
    if args.report:
        print(f"<!-- generated by: python -m core.metrics --report ({db_path}) -->")
        print(store.report())
    if args.auth:
        print(f"\n<!-- python -m core.metrics --auth ({db_path}) -->")
        print(store.auth_report())
    if args.routing:
        print(f"\n<!-- python -m core.metrics --routing ({db_path}) -->")
        print(store.routing_report())
    if args.first_audio:
        print(f"\n<!-- python -m core.metrics --first-audio ({db_path}) -->")
        print(store.first_audio_report())


if __name__ == "__main__":
    main()
