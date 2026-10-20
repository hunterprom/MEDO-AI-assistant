"""P2-4: MEDO speaking first, and the three rules that keep that bearable.

Almost all of this file is about restraint rather than capability. Making an
assistant say something on a timer is easy; making it shut up during a
conversation, at two in the morning, and after it has already said the same
thing once, is the entire feature.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from core.config import load_settings
from core.events import AssistantState, Event, EventBus, EventType, StateMachine
from core.metrics import MetricsStore
from core.proactive import (
    BatteryTrigger,
    EventTrigger,
    ProactiveService,
    TimeTrigger,
    build_triggers,
    in_quiet_hours,
    parse_hhmm,
)


def _config(**over):
    settings = load_settings()
    cfg = settings.proactive
    cfg.enabled = True
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


class _Spy:
    def __init__(self) -> None:
        self.said: list[str] = []

    async def __call__(self, text: str) -> None:
        self.said.append(text)


def _service(triggers, *, state=AssistantState.IDLE, bus=None, metrics=None,
             **cfg):
    # An EMPTY quiet window unless a test asks for one, so these tests assert on
    # the rule they are about and not on what time it happens to be when the
    # suite runs. (An earlier version passed only outside 22:00-07:00.)
    cfg.setdefault("quiet_from", "")
    cfg.setdefault("quiet_until", "")
    bus = bus or EventBus()
    sm = StateMachine(bus)
    sm.state = state
    spy = _Spy()
    return ProactiveService(_config(**cfg), sm, spy, bus, triggers, metrics), spy, sm


# --- quiet hours --------------------------------------------------------------


def test_a_time_parses():
    assert parse_hhmm("22:30").hour == 22
    assert parse_hhmm("nonsense") is None


def test_a_window_crossing_midnight_is_one_window():
    """The naive start<=t<=end reading makes 22:00-07:00 empty, silently
    disabling the only setting anyone actually sets."""
    at = lambda h, m=0: datetime(2026, 8, 25, h, m)   # noqa: E731
    assert in_quiet_hours(at(23), "22:00", "07:00")
    assert in_quiet_hours(at(3), "22:00", "07:00")
    assert not in_quiet_hours(at(12), "22:00", "07:00")
    assert not in_quiet_hours(at(7), "22:00", "07:00")     # end is exclusive


def test_a_same_day_window_works_too():
    at = lambda h: datetime(2026, 8, 25, h)          # noqa: E731
    assert in_quiet_hours(at(10), "09:00", "17:00")
    assert not in_quiet_hours(at(20), "09:00", "17:00")


def test_an_empty_window_never_silences():
    assert not in_quiet_hours(datetime(2026, 8, 25, 3), "", "")
    assert not in_quiet_hours(datetime(2026, 8, 25, 3), "08:00", "08:00")


# --- the three rules ----------------------------------------------------------


@pytest.mark.asyncio
async def test_it_speaks_when_everything_is_clear():
    trigger = EventTrigger(name="task")
    trigger.fire("The build finished.")
    service, spy, _sm = _service([trigger])
    assert await service.tick(datetime(2026, 8, 25, 14, 0)) == 1
    assert spy.said == ["The build finished."]


@pytest.mark.asyncio
async def test_it_stays_quiet_during_a_conversation():
    trigger = EventTrigger(name="task")
    trigger.fire("The build finished.")
    service, spy, _sm = _service([trigger], state=AssistantState.LISTENING)
    assert await service.tick(datetime(2026, 8, 25, 14, 0)) == 0
    assert spy.said == []


@pytest.mark.asyncio
async def test_a_message_held_during_a_turn_is_said_afterwards():
    """Deferred, not dropped: the build really did finish."""
    trigger = EventTrigger(name="task")
    trigger.fire("The build finished.")
    service, spy, sm = _service([trigger], state=AssistantState.SPEAKING)
    await service.tick(datetime(2026, 8, 25, 14, 0))
    assert spy.said == []
    sm.state = AssistantState.IDLE
    await service.tick(datetime(2026, 8, 25, 14, 0))
    assert spy.said == ["The build finished."]


@pytest.mark.asyncio
async def test_it_stays_quiet_in_the_small_hours():
    trigger = EventTrigger(name="task")
    trigger.fire("The build finished.")
    service, spy, _sm = _service([trigger], quiet_from="22:00", quiet_until="07:00")
    assert await service.tick(datetime(2026, 8, 25, 3, 0)) == 0
    assert spy.said == []


@pytest.mark.asyncio
async def test_a_quiet_hours_message_is_dropped_not_queued():
    """A 3am reminder delivered at 7am is noise; the moment for it has passed."""
    trigger = EventTrigger(name="task")
    trigger.fire("The build finished.")
    service, spy, _sm = _service([trigger], quiet_from="22:00", quiet_until="07:00")
    await service.tick(datetime(2026, 8, 25, 3, 0))
    assert await service.tick(datetime(2026, 8, 25, 14, 0)) == 0
    assert spy.said == []


@pytest.mark.asyncio
async def test_the_whole_layer_is_off_by_default():
    settings = load_settings()
    assert settings.proactive.enabled is False
    trigger = EventTrigger(name="task")
    trigger.fire("hello")
    service, spy, _sm = _service([trigger], enabled=False)
    assert await service.tick(datetime(2026, 8, 25, 14, 0)) == 0
    assert spy.said == []


@pytest.mark.asyncio
async def test_a_cooldown_stops_it_repeating_itself():
    trigger = EventTrigger(name="task", cooldown_s=3600.0)
    trigger.fire("The build finished.")
    trigger.fire("The build finished.")
    service, spy, _sm = _service([trigger])
    noon = datetime(2026, 8, 25, 12, 0)
    assert await service.tick(noon) == 1
    assert await service.tick(noon) == 0
    assert len(spy.said) == 1


# --- the triggers -------------------------------------------------------------


def test_the_battery_trigger_fires_on_the_crossing():
    readings = iter([(80, False), (12, False), (10, False), (8, False)])
    trigger = BatteryTrigger(threshold=15, reader=lambda: next(readings))
    assert trigger.check() is None            # 80%: fine
    assert "12 percent" in trigger.check()    # crossed
    assert trigger.check() is None            # still low: already said
    assert trigger.check() is None


def test_the_battery_trigger_rearms_after_charging():
    readings = iter([(10, False), (90, True), (9, False)])
    trigger = BatteryTrigger(threshold=15, reader=lambda: next(readings))
    assert trigger.check() is not None
    assert trigger.check() is None            # plugged in: back above
    assert trigger.check() is not None        # unplugged and low again


def test_the_battery_trigger_is_silent_while_plugged_in():
    trigger = BatteryTrigger(threshold=15, reader=lambda: (5, True))
    assert trigger.check() is None


def test_a_machine_with_no_battery_is_not_an_error():
    trigger = BatteryTrigger(threshold=15, reader=lambda: None)
    assert trigger.check() is None


def test_the_time_trigger_fires_in_its_minute():
    at_five = datetime(2026, 8, 25, 17, 30)
    trigger = TimeTrigger(name="t", at="17:30", message="Time to stop.",
                          clock=lambda: at_five)
    assert trigger.check() == "Time to stop."
    trigger.clock = lambda: datetime(2026, 8, 25, 17, 31)
    assert trigger.check() is None


def test_a_time_trigger_without_a_message_says_nothing():
    trigger = TimeTrigger(name="t", at="17:30", message="",
                          clock=lambda: datetime(2026, 8, 25, 17, 30))
    assert trigger.check() is None


def test_an_event_trigger_queues_in_order():
    trigger = EventTrigger(name="task", cooldown_s=0.0)
    trigger.fire("one")
    trigger.fire("two")
    assert (trigger.check(), trigger.check(), trigger.check()) == ("one", "two", None)


def test_an_event_trigger_ignores_an_empty_message():
    trigger = EventTrigger(name="task")
    trigger.fire("   ")
    assert trigger.check() is None


@pytest.mark.asyncio
async def test_a_raising_trigger_does_not_stop_the_others():
    class _Broken(EventTrigger):
        def check(self):
            raise RuntimeError("sensor exploded")

    good = EventTrigger(name="good", cooldown_s=0.0)
    good.fire("still here")
    service, spy, _sm = _service([_Broken(name="broken"), good])
    assert await service.tick(datetime(2026, 8, 25, 14, 0)) == 1
    assert spy.said == ["still here"]


@pytest.mark.asyncio
async def test_a_failing_announcer_does_not_break_the_loop():
    async def broken(text):
        raise RuntimeError("no speakers")

    bus = EventBus()
    sm = StateMachine(bus)
    trigger = EventTrigger(name="task", cooldown_s=0.0)
    trigger.fire("hello")
    service = ProactiveService(_config(), sm, broken, bus, [trigger])
    assert await service.tick(datetime(2026, 8, 25, 14, 0)) == 1   # survived


# --- observability ------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_unprompted_line_is_recorded(tmp_path):
    """The risk of this layer is that it talks too much; this is how you know."""
    store = MetricsStore(tmp_path / "m.db")
    trigger = EventTrigger(name="task", cooldown_s=0.0)
    trigger.fire("The build finished.")
    service, _spy, _sm = _service([trigger], metrics=store)
    await service.tick(datetime(2026, 8, 25, 14, 0))
    assert "task" in store.proactive_report()


def test_the_report_says_so_when_it_has_been_quiet(tmp_path):
    assert "not spoken unprompted" in MetricsStore(tmp_path / "m.db").proactive_report()


# --- building from config -----------------------------------------------------


def test_nothing_is_built_when_the_layer_is_off():
    cfg = load_settings().proactive
    cfg.enabled = False
    assert build_triggers(cfg) == []


def test_the_battery_trigger_is_built_from_config():
    triggers = build_triggers(_config(battery_enabled=True, battery_threshold=20))
    assert any(isinstance(t, BatteryTrigger) and t.threshold == 20
               for t in triggers)


def test_time_triggers_are_built_from_config():
    triggers = build_triggers(_config(
        battery_enabled=False,
        at=[{"at": "17:30", "say": "Time to stop."},
            {"at": "nonsense", "say": "never"},
            {"at": "09:00", "say": ""}]))
    assert [t.at for t in triggers] == ["17:30"]


@pytest.mark.asyncio
async def test_the_bus_flushes_held_messages_when_the_turn_ends():
    """The state machine is the signal; nothing has to poll for the gap."""
    bus = EventBus()
    trigger = EventTrigger(name="task", cooldown_s=0.0)
    trigger.fire("The build finished.")
    service, spy, sm = _service([trigger], bus=bus, state=AssistantState.SPEAKING)
    await service.tick(datetime(2026, 8, 25, 14, 0))
    assert spy.said == []
    sm.state = AssistantState.IDLE
    await bus.emit(Event(EventType.STATE_CHANGED,
                         (AssistantState.SPEAKING, AssistantState.IDLE)))
    import asyncio

    await asyncio.sleep(0)          # let the flush task run
    await asyncio.sleep(0)
    assert spy.said == ["The build finished."]


def test_a_trigger_is_ready_on_a_machine_that_just_booted(monkeypatch):
    """`_last_fired = 0.0` meant "fired at boot", not "never fired".

    ``time.monotonic()`` counts from the boot on Windows and Linux, so a
    freshly constructed trigger with a zero sentinel was not ready until the
    machine had been up for its whole cooldown — an hour of silence from the
    proactive layer after every restart. This pins the sentinel rather than the
    arithmetic, so it fails on any uptime instead of only on a short one.
    """
    import core.proactive as proactive

    monkeypatch.setattr(proactive.time, "monotonic", lambda: 12.0)  # just booted
    trigger = EventTrigger(name="task", cooldown_s=3600.0)
    assert trigger.ready() is True
    trigger.mark_fired()
    assert trigger.ready() is False
