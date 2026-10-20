"""MEDO speaking first — and the rules that make that tolerable.

Everything MEDO does is a reply. Useful things it could say without being asked
— the laptop is at eight percent, the build you started twenty minutes ago has
finished, there is a thing at four — it cannot, because there is no path from
"something happened" to "say something".

This is that path, and it is mostly restraint. An assistant that speaks
unprompted is charming for a day and intolerable by the end of the week, so the
rules come first and the triggers second:

* **Never during a conversation.** A proactive line while the user is mid-
  sentence, or while MEDO is answering, is an interruption. The state machine
  already knows; this waits for IDLE.
* **Never in the quiet hours.** A do-not-disturb window, checked in local time,
  and it handles the wrap-around case (22:00 to 08:00 is one window, not two).
* **Never twice.** Every trigger has a cooldown, and a level-based one (the
  battery) only fires when it CROSSES its threshold — not on every poll while
  it sits below.
* **Never a surprise you can't stop.** The whole layer is off by default.

Triggers are data (:class:`Trigger` subclasses), the service is a loop, and the
decision to speak is a pure function of state + clock. That last part is what
makes the rules testable without waiting for a Tuesday evening.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time

from core.events import AssistantState, Event, EventBus, EventType

logger = logging.getLogger(__name__)

#: How often the service looks at its triggers. Proactive news is never urgent
#: to the second, and a tighter loop is only a battery cost.
POLL_INTERVAL_S = 30.0


def parse_hhmm(value: str) -> dt_time | None:
    """``"22:30"`` -> a time, or None when it isn't one."""
    try:
        hour, minute = value.strip().split(":")
        return dt_time(int(hour), int(minute))
    except (ValueError, AttributeError):
        return None


def in_quiet_hours(now: datetime, start: str, end: str) -> bool:
    """Is ``now`` inside the do-not-disturb window?

    Handles the wrap-around, which is the case that matters: "22:00 to 08:00" is
    one window crossing midnight, and the naive ``start <= t <= end`` reads it
    as an empty one — silently disabling the only setting anyone will actually
    set.
    """
    begin, finish = parse_hhmm(start), parse_hhmm(end)
    if begin is None or finish is None or begin == finish:
        return False
    current = now.time()
    if begin < finish:
        return begin <= current < finish
    return current >= begin or current < finish        # crosses midnight


@dataclass
class Trigger:
    """Something worth saying, and the rule for when to say it again.

    Subclasses implement :meth:`check`, which returns the line to speak or None.
    It must be cheap and must not block: it runs on the event loop every poll.
    """

    name: str
    #: Minimum seconds between two firings of THIS trigger.
    cooldown_s: float = 3600.0
    #: When this trigger last fired, on the MONOTONIC clock. None means never.
    #:
    #: Not 0.0. ``time.monotonic()`` counts from an arbitrary origin that on
    #: Windows and Linux is the boot, so a zero sentinel does not read as
    #: "never fired" — it reads as "fired at boot", and every trigger then
    #: stays silent until the machine has been up for a full cooldown. That is
    #: an hour of no battery warnings after exactly the reboot a flat battery
    #: causes, and half an hour for the battery trigger's own 1800 s. Caught by
    #: the suite only because the machine happened to be 45 minutes into an
    #: uptime; on a long-running box the tests pass and the feature is broken
    #: where it matters.
    _last_fired: float | None = field(default=None, repr=False)

    def ready(self, now: float | None = None) -> bool:
        if self._last_fired is None:
            return True
        now = time.monotonic() if now is None else now
        return (now - self._last_fired) >= self.cooldown_s

    def mark_fired(self, now: float | None = None) -> None:
        self._last_fired = time.monotonic() if now is None else now

    def check(self) -> str | None:                     # pragma: no cover - base
        raise NotImplementedError


@dataclass
class BatteryTrigger(Trigger):
    """"The battery is at eight percent" — once per discharge, not per poll."""

    name: str = "battery"
    threshold: int = 15
    cooldown_s: float = 1800.0
    #: The reading last seen, so this fires on the CROSSING rather than on every
    #: poll while the battery sits below the threshold. A warning repeated every
    #: thirty seconds is how a helpful feature becomes one people disable.
    _was_above: bool = field(default=True, repr=False)
    reader: object = None                              # injected in tests

    def _percent(self) -> tuple[int, bool] | None:
        """(percent, plugged_in), or None when there is no battery."""
        if self.reader is not None:
            return self.reader()
        try:
            import psutil

            battery = psutil.sensors_battery()
        except Exception:  # noqa: BLE001 - no battery, no psutil, a desktop
            return None
        if battery is None:
            return None
        return int(battery.percent), bool(battery.power_plugged)

    def check(self) -> str | None:
        reading = self._percent()
        if reading is None:
            return None
        percent, plugged = reading
        above = percent > self.threshold or plugged
        crossed = self._was_above and not above
        self._was_above = above
        if not crossed:
            return None
        return (f"The battery is down to {percent} percent — worth plugging in.")


@dataclass
class TimeTrigger(Trigger):
    """A line at a time of day, at most once per day."""

    name: str = "time"
    at: str = "09:00"
    message: str = ""
    cooldown_s: float = 23 * 3600.0
    clock: object = None                               # injected in tests

    def check(self) -> str | None:
        target = parse_hhmm(self.at)
        if target is None or not self.message:
            return None
        now = (self.clock() if self.clock else datetime.now()).time()
        # A window, not an instant: the loop polls every thirty seconds and must
        # not miss the minute it was aiming for.
        if now.hour != target.hour or now.minute != target.minute:
            return None
        return self.message


@dataclass
class EventTrigger(Trigger):
    """Fires when something publishes on the bus — a long task finishing.

    The producer calls :meth:`fire`; the service picks it up on its next pass.
    Queued rather than spoken immediately because "immediately" is often the
    middle of a sentence.
    """

    name: str = "event"
    cooldown_s: float = 0.0
    _queued: list = field(default_factory=list, repr=False)

    def fire(self, message: str) -> None:
        if message and message.strip():
            self._queued.append(message.strip())

    def check(self) -> str | None:
        return self._queued.pop(0) if self._queued else None


class ProactiveService:
    """Watches the triggers and speaks, when the rules allow it.

    ``announce`` is ``async (text: str) -> None`` — the same announcer routines
    and timers already use, so a proactive line is spoken in voice mode, printed
    otherwise, and visible in the HUD either way.
    """

    def __init__(self, config, state_machine, announce, bus: EventBus | None = None,
                 triggers: list[Trigger] | None = None, metrics=None) -> None:
        self._config = config
        self._sm = state_machine
        self._announce = announce
        self._bus = bus
        self._metrics = metrics
        self.triggers: list[Trigger] = list(triggers or [])
        #: Set while a turn is in flight, so a line queued mid-conversation waits
        #: for the end of it rather than being dropped.
        self._pending: list[str] = []
        if bus is not None:
            bus.subscribe(EventType.STATE_CHANGED, self._on_state)

    # -- the rules -------------------------------------------------------------

    def busy(self) -> bool:
        """Is the user mid-conversation? Then MEDO does not start another one."""
        state = getattr(self._sm, "state", AssistantState.IDLE)
        return state is not AssistantState.IDLE

    def quiet(self, now: datetime | None = None) -> bool:
        return in_quiet_hours(now or datetime.now(),
                              self._config.quiet_from, self._config.quiet_until)

    def may_speak(self, now: datetime | None = None) -> bool:
        """Every rule, in one place, as a pure function of state and clock."""
        return bool(self._config.enabled) and not self.busy() and not self.quiet(now)

    def _on_state(self, event: Event) -> None:
        """Flush anything that was held back while the user was talking."""
        _old, new = event.payload
        if new is AssistantState.IDLE and self._pending:
            asyncio.create_task(self._flush())

    async def _flush(self, now: datetime | None = None) -> None:
        """Say what was held back, while the rules still allow it.

        Takes the clock rather than reading it, so the caller's notion of "now"
        is the one that decides — and so a held message flushed at the end of a
        turn is still checked against the quiet window, not just against
        whether the user has stopped talking.
        """
        while self._pending and self.may_speak(now):
            await self._say(self._pending.pop(0))

    async def _say(self, message: str) -> None:
        try:
            await self._announce(message)
        except Exception:  # noqa: BLE001 - a nudge must not break the session
            logger.warning("proactive announcement failed", exc_info=True)

    # -- the loop --------------------------------------------------------------

    def poll(self, now: datetime | None = None) -> list[str]:
        """Collect what the triggers want to say. Pure enough to unit-test.

        Triggers are ALWAYS checked, even in quiet hours — a level trigger has
        to see the crossing to know it happened. What the rules gate is
        speaking, not observing.
        """
        messages: list[str] = []
        for trigger in self.triggers:
            try:
                message = trigger.check()
            except Exception:  # noqa: BLE001 - one bad trigger, not a dead loop
                logger.warning("proactive trigger %r failed", trigger.name,
                               exc_info=True)
                continue
            if not message or not trigger.ready():
                continue
            trigger.mark_fired()
            messages.append(message)
            self._record(trigger.name)
        return messages

    def _record(self, trigger_name: str) -> None:
        if self._metrics is None:
            return
        with contextlib.suppress(Exception):
            self._metrics.record_proactive(trigger_name)

    async def tick(self, now: datetime | None = None) -> int:
        """One pass: check triggers, speak or hold. Returns how many were spoken."""
        messages = self.poll(now)
        if not messages:
            # Something held back earlier may be sayable now.
            if self._pending and self.may_speak(now):
                await self._flush(now)
            return 0
        if not self.may_speak(now):
            # Quiet hours DROP; a conversation DEFERS. A reminder delivered at
            # 3am is worse than one missed, but one delayed until you stop
            # talking is exactly right.
            if self.busy():
                self._pending.extend(messages)
                logger.info("proactive: holding %d message(s) until the "
                            "conversation ends", len(messages))
            else:
                logger.info("proactive: %d message(s) suppressed by quiet hours",
                            len(messages))
            return 0
        for message in messages:
            await self._say(message)
        return len(messages)

    async def run_forever(self) -> None:
        if not self._config.enabled or not self.triggers:
            return
        logger.info("proactive layer armed: %s",
                    ", ".join(t.name for t in self.triggers))
        while True:
            await asyncio.sleep(POLL_INTERVAL_S)
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the loop outlives its triggers
                logger.exception("proactive tick failed")


def build_triggers(config) -> list[Trigger]:
    """The triggers described by config. Empty when the layer is off."""
    if not config.enabled:
        return []
    triggers: list[Trigger] = []
    if config.battery_enabled:
        triggers.append(BatteryTrigger(name="battery",
                                       threshold=config.battery_threshold))
    for item in config.at or []:
        if not isinstance(item, dict):
            continue
        at, message = str(item.get("at", "")), str(item.get("say", ""))
        if parse_hhmm(at) and message:
            triggers.append(TimeTrigger(name=f"at-{at}", at=at, message=message))
    return triggers
