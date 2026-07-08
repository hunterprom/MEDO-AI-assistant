"""Proactive routines: next-fire math (pure) + firing through a fake router."""

from __future__ import annotations

from datetime import datetime

import pytest

from core.config import RoutineItem
from core.routines import RoutineScheduler, seconds_until

WED = datetime(2026, 7, 1, 7, 0)   # 2026-07-01 is a Wednesday, 07:00


def test_seconds_until_later_today():
    assert seconds_until("08:00", [], WED) == 3600


def test_seconds_until_wraps_to_tomorrow():
    assert seconds_until("06:00", [], WED) == 23 * 3600


def test_seconds_until_respects_days():
    # Only Friday allowed: Wed 07:00 -> Fri 08:00 = 2 days + 1 hour.
    assert seconds_until("08:00", ["fri"], WED) == 49 * 3600
    # Full names / mixed case are tolerated.
    assert seconds_until("08:00", ["Friday"], WED) == 49 * 3600


@pytest.mark.asyncio
async def test_fire_routes_and_announces_and_survives_failures():
    routed, announced = [], []

    class _Router:
        async def route(self, text, context=None, on_delta=None):
            routed.append(text)
            if "broken" in text:
                raise RuntimeError("boom")
            class R: speech = f"answer to {text}"
            return R()

    async def announcer(text):
        announced.append(text)

    routine = RoutineItem(name="morning briefing",
                          ask=["what's the weather", "broken step", "the news"])
    await RoutineScheduler([routine], _Router(), announcer).fire(routine)

    assert routed == ["what's the weather", "broken step", "the news"]
    assert announced[0] == "Morning briefing."
    assert "answer to what's the weather" in announced
    assert "answer to the news" in announced      # the broken step didn't stop it


def test_disabled_and_empty_routines_are_dropped():
    r1 = RoutineItem(name="off", ask=["x"], enabled=False)
    r2 = RoutineItem(name="empty", ask=[])
    sched = RoutineScheduler([r1, r2], router=None, announcer=None)
    assert sched._routines == []
