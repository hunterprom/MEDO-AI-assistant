"""Proactive routines: scheduled briefings MEDO delivers without being asked.

Configured in ``config.yaml``::

    routines:
      - name: "morning briefing"
        at: "08:00"
        days: []                      # empty = every day; or [mon, tue, ...]
        ask:
          - "what's the weather"
          - "what's the news"

At each fire time every ``ask`` utterance runs through the SAME intent router
as voice/typed input, and the answers are announced — spoken aloud in voice
mode, printed otherwise, and visible in the HUD via the normal ROUTED events.
A failing utterance is logged and skipped; the scheduler itself never dies.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

DAY_KEYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def seconds_until(at: str, days: list[str], now: datetime) -> float:
    """Seconds from ``now`` to the next HH:MM occurrence on an allowed day.

    ``days`` uses three-letter names (``mon`` … ``sun``); empty means daily.
    Pure — unit-tested without any sleeping.
    """
    hh, mm = at.strip().split(":")
    allowed = {d.strip().lower()[:3] for d in days if d.strip()} or set(DAY_KEYS)
    for ahead in range(8):  # today + a full week always contains a hit
        candidate = (now + timedelta(days=ahead)).replace(
            hour=int(hh), minute=int(mm), second=0, microsecond=0
        )
        if candidate > now and DAY_KEYS[candidate.weekday()] in allowed:
            return (candidate - now).total_seconds()
    raise ValueError(f"no valid day for routine time {at!r} days {days!r}")


class RoutineScheduler:
    """Sleeps until the next routine, runs its questions, announces answers."""

    def __init__(self, routines, router, announcer) -> None:
        self._routines = [r for r in routines if r.enabled and r.ask]
        self._router = router
        self._announcer = announcer

    async def run_forever(self) -> None:
        if not self._routines:
            return
        logger.info("routines armed: %s",
                    ", ".join(f"{r.name} @ {r.at}" for r in self._routines))
        while True:
            now = datetime.now()
            delay, routine = min(
                (seconds_until(r.at, r.days, now), r) for r in self._routines
            )
            await asyncio.sleep(delay + 1.0)
            await self.fire(routine)

    async def fire(self, routine) -> None:
        """Run one routine now (also used by tests and a manual trigger)."""
        logger.info("routine %r firing", routine.name)
        try:
            await self._announcer(f"{routine.name.capitalize()}.")
        except Exception:
            logger.exception("routine announcement failed")
        for utterance in routine.ask:
            try:
                result = await self._router.route(utterance, context={"source": "routine"})
                await self._announcer(result.speech)
            except Exception:  # one bad question must not kill the briefing
                logger.exception("routine step %r failed", utterance)
