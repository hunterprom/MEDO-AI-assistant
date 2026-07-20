"""M5: latency instrumentation aggregation and the event/state machine."""

from __future__ import annotations

import pytest

from core.events import AssistantState, EventBus, EventType, StateMachine
from core.metrics import BUDGETS, LatencyLog, MetricsStore, TurnTimings, _percentile


def test_latency_summary_averages_by_stage_and_path():
    log = LatencyLog()
    log.record(TurnTimings(path="FAST", route_ms=2))
    log.record(TurnTimings(path="FAST", route_ms=4))
    log.record(TurnTimings(path="LLM", stt_ms=800, route_ms=25000, tts_ms=200))

    s = log.summary()
    assert s["count"] == 3
    assert s["fast_route_ms"] == 3            # (2 + 4) / 2, LLM excluded
    assert s["llm_route_ms"] == 25000
    assert s["stt_ms"] == 800                 # only the LLM turn had STT
    assert s["wake_to_listen_ms"] is None     # never recorded


def test_budgets_present():
    assert BUDGETS["fast_route_ms"] == 1000.0
    assert BUDGETS["llm_spoken_ms"] == 4000.0


# --- persisted routing metrics + the --report table --------------------------


def _seeded_store(tmp_path) -> MetricsStore:
    store = MetricsStore(tmp_path / "metrics-test.db")
    for ms in (5, 8, 11, 40):                       # 4 fast datetime hits
        store.record("FAST", "datetime", ms)
    store.record("FAST", "volume", 3)
    store.record("LLM", None, 2500)                 # plain chat, no tool
    store.record("LLM", "weather", 4200)
    return store


def test_percentiles_nearest_rank():
    assert _percentile([], 50) == 0.0
    assert _percentile([1, 2, 3, 4], 50) == 2
    assert _percentile([1, 2, 3, 4], 95) == 4
    assert _percentile([7], 95) == 7


def test_report_from_seeded_db(tmp_path):
    report = _seeded_store(tmp_path).report()
    assert "7 routed requests measured." in report
    assert "| FAST | 5 | 71.4% |" in report
    assert "| LLM | 2 | 28.6% |" in report
    # FAST p50 over [3,5,8,11,40] = 8; p95 = 40 (nearest rank)
    assert "| 8 ms | 40 ms |" in report
    # top skills: datetime first (4 hits), then the single-hit ones
    lines = report.splitlines()
    assert "| datetime | 4 | 8 ms |" in lines
    assert lines.index("| datetime | 4 | 8 ms |") < lines.index("| volume | 1 | 3 ms |")


def test_report_empty_db_says_so(tmp_path):
    report = MetricsStore(tmp_path / "empty.db").report()
    assert "No routing metrics recorded yet" in report


def test_record_never_raises_on_bad_db(tmp_path):
    bad = tmp_path / "not-a-dir" / "x.db"           # parent missing -> connect fails
    MetricsStore(bad).record("FAST", "datetime", 1.0)  # must not raise


@pytest.mark.asyncio
async def test_router_persists_metrics(tmp_path, monkeypatch):
    """One routed fast-path request lands as one metrics row."""
    import re

    from core.config import load_settings
    from core.router import Router
    from llm.client import OllamaClient
    from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult

    class Hello(Skill):
        name = "hello"
        description = "test"
        patterns = [re.compile(r"\bhello\b", re.IGNORECASE)]

        async def execute(self, request: SkillRequest) -> SkillResult:
            return SkillResult("hi")

        def tool_schema(self):
            return {}

    monkeypatch.setenv("MEDO_MEMORY__DB_PATH", str(tmp_path / "router-metrics.db"))
    settings = load_settings()
    registry = SkillRegistry()
    registry.register(Hello())
    router = Router(settings, registry, OllamaClient(settings.llm), EventBus())
    router.model = None
    await router.route("hello there")

    report = router.metrics.report()
    assert "1 routed request measured." in report
    assert "| hello | 1 |" in report


@pytest.mark.asyncio
async def test_state_machine_emits_transitions():
    bus = EventBus()
    seen: list[tuple] = []
    bus.subscribe(EventType.STATE_CHANGED, lambda e: seen.append(e.payload))
    sm = StateMachine(bus)

    await sm.transition(AssistantState.LISTENING)
    await sm.transition(AssistantState.LISTENING)   # no-op, same state
    await sm.transition(AssistantState.THINKING)

    assert seen == [
        (AssistantState.IDLE, AssistantState.LISTENING),
        (AssistantState.LISTENING, AssistantState.THINKING),
    ]
