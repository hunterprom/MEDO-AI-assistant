"""M5: latency instrumentation aggregation and the event/state machine."""

from __future__ import annotations

import pytest

from core.events import AssistantState, EventBus, EventType, StateMachine
from core.metrics import BUDGETS, LatencyLog, MetricsStore, TurnTimings, percentile


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


# --- persisted routing stats + the --report table ---------------------------


def test_percentile_nearest_rank():
    assert percentile([], 50) == 0.0
    assert percentile([7.0], 50) == 7.0
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([5, 1, 3], 100) == 5  # unsorted input handled


def test_metrics_store_report_from_seeded_db(tmp_path):
    store = MetricsStore(tmp_path / "metrics.db")
    for latency in (2.0, 4.0, 6.0):
        store.record("FAST", "datetime", latency)
    store.record("FAST", "volume", 3.0)
    store.record("LLM", None, 900.0)

    report = store.report()
    assert "| FAST | 4 | 80% | 3 ms | 6 ms |" in report  # nearest-rank p50/p95
    assert "| LLM | 1 | 20% | 900 ms | 900 ms |" in report
    assert "| datetime | 3 |" in report                  # top skill first
    assert "| volume | 1 |" in report


def test_metrics_store_empty_db_message(tmp_path):
    report = MetricsStore(tmp_path / "empty.db").report()
    assert "No routing stats recorded yet" in report


@pytest.mark.asyncio
async def test_router_persists_metrics_row(tmp_path):
    from core.config import load_settings
    from core.router import Router
    from llm.client import OllamaClient
    from skills.base import SkillRegistry
    from skills.datetime_skill import DateTimeSkill

    settings = load_settings()
    store = MetricsStore(tmp_path / "m.db")
    registry = SkillRegistry()
    registry.register(DateTimeSkill())
    router = Router(settings, registry, OllamaClient(settings.llm), EventBus(),
                    metrics=store)
    router.model = None
    await router.route("what time is it")

    rows = store._rows()
    assert len(rows) == 1
    path, skill, latency = rows[0]
    assert path == "FAST" and skill == "datetime" and latency >= 0


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
