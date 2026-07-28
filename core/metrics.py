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

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path


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
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.execute(_METRICS_SCHEMA)
        return conn

    def record(self, path: str, skill: str | None, latency_ms: float) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO metrics (path, skill, latency_ms) VALUES (?, ?, ?)",
                    (path, skill, float(latency_ms)),
                )
        except Exception:  # noqa: BLE001 - telemetry must never break routing
            pass

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
    parser.add_argument("--db", default=None,
                        help="database path (default: memory.db_path from config.yaml)")
    args = parser.parse_args()
    if not args.report:
        parser.print_help()
        return
    db_path = args.db or load_settings().memory.db_path
    print(f"<!-- generated by: python -m core.metrics --report ({db_path}) -->")
    print(MetricsStore(db_path).report())


if __name__ == "__main__":
    main()
