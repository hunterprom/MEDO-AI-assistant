"""Latency instrumentation.

Each turn records per-stage timings; :class:`LatencyLog` keeps them and computes
averages against the budgets in the README. Kept separate from the UI so it can be
unit-tested and queried (``/latency`` in the REPL).
"""

from __future__ import annotations

from dataclasses import dataclass


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
