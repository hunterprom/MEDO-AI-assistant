"""Latency instrumentation.

Each turn records per-stage timings; :class:`LatencyLog` keeps them and computes
averages against the budgets in the README. Kept separate from the UI so it can be
unit-tested and queried (``/latency`` in the REPL).

:class:`MetricsStore` additionally persists one row per routed request into the
shared sqlite database (``jarvis.db`` — no new files), so routing stats survive
restarts. ``python -m core.metrics --report`` renders them as the markdown
table in the README's Performance section.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TurnTimings:
    """Milliseconds spent in each stage of one interaction (None = not run)."""

    path: str = ""                       # "FAST" or "LLM"
    wake_to_listen_ms: float | None = None
    stt_ms: float | None = None
    route_ms: float | None = None        # LLM path: includes token generation
    tts_ms: float | None = None          # time to synthesize the reply (pre-play)

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
        }


# --- persisted routing stats (the README Performance table) -----------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    path TEXT NOT NULL,
    skill TEXT,
    latency_ms REAL NOT NULL
);
"""


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (0 < pct <= 100). Empty input -> 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * pct // 100))  # ceil without math import
    return ordered[int(rank) - 1]


class MetricsStore:
    """Append-only per-request routing log in the shared sqlite database.

    Same short-lived-connection pattern as the facts store (no cross-thread
    sharing). A failed write is logged and swallowed — metrics must never
    break routing.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.executescript(_SCHEMA)
        return conn

    def record(self, path: str, skill: str | None, latency_ms: float) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO metrics (ts, path, skill, latency_ms)"
                    " VALUES (?, ?, ?, ?)",
                    (time.time(), path, skill, float(latency_ms)),
                )
        except Exception:
            logger.debug("could not record metrics row", exc_info=True)

    def _rows(self) -> list[tuple[str, str | None, float]]:
        try:
            with self._connect() as conn:
                return conn.execute(
                    "SELECT path, skill, latency_ms FROM metrics"
                ).fetchall()
        except Exception:
            return []

    def report(self) -> str:
        """The README Performance table as markdown (or a friendly empty note)."""
        rows = self._rows()
        if not rows:
            # ASCII only: this prints straight to a cp1252 Windows console.
            return ("No routing stats recorded yet - talk to MEDO for a while, "
                    "then re-run `python -m core.metrics --report`.")
        total = len(rows)
        lines = ["| Path | Requests | Share | p50 | p95 |",
                 "|------|---------:|------:|----:|----:|"]
        for path in ("FAST", "LLM"):
            lat = [r[2] for r in rows if r[0] == path]
            if not lat:
                continue
            lines.append(
                f"| {path} | {len(lat)} | {len(lat) / total * 100:.0f}% "
                f"| {percentile(lat, 50):.0f} ms | {percentile(lat, 95):.0f} ms |"
            )
        by_skill: dict[str, list[float]] = {}
        for path, skill, latency in rows:
            if skill:
                by_skill.setdefault(skill, []).append(latency)
        if by_skill:
            top = sorted(by_skill.items(), key=lambda kv: len(kv[1]), reverse=True)
            lines += ["", "| Skill (top 10) | Requests | p50 |",
                      "|----------------|---------:|----:|"]
            for skill, lat in top[:10]:
                lines.append(
                    f"| {skill} | {len(lat)} | {percentile(lat, 50):.0f} ms |"
                )
        return "\n".join(lines)


def _main() -> None:  # pragma: no cover - thin CLI over report()
    import argparse

    parser = argparse.ArgumentParser(description="MEDO routing/latency stats")
    parser.add_argument("--report", action="store_true",
                        help="print the markdown Performance table")
    parser.add_argument("--db", default=None,
                        help="sqlite path (default: memory.db_path from config)")
    args = parser.parse_args()
    if not args.report:
        parser.print_help()
        return
    db = args.db
    if db is None:
        from core.config import load_settings

        db = load_settings().memory.db_path
    print(MetricsStore(db).report())


if __name__ == "__main__":  # pragma: no cover
    _main()
