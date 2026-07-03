"""M5: latency instrumentation aggregation and the event/state machine."""

from __future__ import annotations

import pytest

from core.events import AssistantState, EventBus, EventType, StateMachine
from core.metrics import BUDGETS, LatencyLog, TurnTimings


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
