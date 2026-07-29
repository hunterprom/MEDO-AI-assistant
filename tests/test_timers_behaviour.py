"""Timers end to end: arm, list, cancel, fire — without waiting for real time.

Timers are the one skill that keeps state and a background task, so the things
worth pinning are the ones that outlive a single call: does cancelling really
cancel, does a labelled reminder reach the store, does firing announce once and
then clean up after itself.
"""

from __future__ import annotations

import asyncio

import pytest

from skills.base import SkillRequest
from skills.timers import TimerSkill


class _Announcer:
    """Stands in for the spoken announcement a timer makes when it fires."""

    def __init__(self):
        self.said: list[str] = []

    async def __call__(self, text: str) -> None:
        self.said.append(text)


@pytest.fixture
def announcer():
    return _Announcer()


@pytest.fixture
def skill(announcer):
    return TimerSkill(announcer)


async def _say(skill, text):
    match = skill.match(text)
    assert match is not None, f"no pattern matched: {text!r}"
    return await skill.execute(SkillRequest(text=text, match=match))


# --- arming --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setting_a_timer_arms_it(skill):
    result = await _say(skill, "set a timer for 5 minutes")
    assert result.success and result.data["seconds"] == 300
    assert len(skill._timers) == 1


@pytest.mark.asyncio
async def test_a_timer_with_no_duration_asks_instead_of_arming(skill):
    result = await _say(skill, "set a timer")
    assert result.success is False and skill._timers == {}


@pytest.mark.asyncio
async def test_a_labelled_reminder_keeps_its_label(skill):
    result = await _say(skill, "remind me in 10 minutes to check the oven")
    assert "check the oven" in result.speech


@pytest.mark.asyncio
async def test_set_an_alarm_matches_and_arms(skill):
    # "an"/"the" article — not just "a" — must still route to the timer skill.
    result = await _say(skill, "set an alarm for 10 minutes")
    assert result.success and result.data["seconds"] == 600


@pytest.mark.parametrize("seconds,expected", [
    (30, "30 seconds"),
    (60, "1 minute"),
    (90, "1 minute and 30 seconds"),
    (3600, "1 hour"),
    (3661, "1 hour and 1 minute and 1 second"),
    (0, "0 seconds"),
])
def test_durations_are_spoken_naturally(seconds, expected):
    assert TimerSkill._pretty(seconds) == expected


# --- listing and cancelling ----------------------------------------------------


@pytest.mark.asyncio
async def test_listing_with_nothing_armed(skill):
    result = await _say(skill, "list my timers")
    assert "no active timers" in result.speech.lower()


@pytest.mark.asyncio
async def test_listing_reports_what_is_armed(skill):
    await _say(skill, "set a timer for 5 minutes")
    await _say(skill, "remind me in 10 minutes to stretch")
    result = await _say(skill, "list my timers")
    assert "2 active" in result.speech and "stretch" in result.speech


@pytest.mark.asyncio
async def test_cancelling_stops_the_background_tasks(skill):
    await _say(skill, "set a timer for 5 minutes")
    await _say(skill, "set a timer for 9 minutes")
    tasks = [t.task for t in skill._timers.values()]
    result = await _say(skill, "cancel all timers")

    assert "2 timers" in result.speech and skill._timers == {}
    await asyncio.sleep(0)                       # let the cancellations land
    assert all(t.cancelled() or t.done() for t in tasks), "a task outlived cancel"


@pytest.mark.asyncio
async def test_cancelling_with_nothing_armed_is_not_an_error(skill):
    result = await _say(skill, "cancel all timers")
    assert result.success and "0 timers" in result.speech


@pytest.mark.asyncio
async def test_reminder_label_containing_cancel_is_not_hijacked(skill):
    # "cancel" inside a reminder LABEL must not divert into the cancel branch and
    # wipe every active timer — it should arm the reminder like any other.
    await _say(skill, "set a timer for 5 minutes")
    result = await _say(skill, "remind me to cancel my dentist appointment in 2 hours")
    assert "cancel my dentist appointment" in result.speech
    assert len(skill._timers) == 2


@pytest.mark.asyncio
async def test_reminder_label_containing_list_is_not_hijacked(skill):
    result = await _say(skill, "remind me to list the shopping items in 10 minutes")
    assert "list the shopping items" in result.speech
    assert len(skill._timers) == 1


# --- firing --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_timer_announces_once_and_cleans_up(skill, announcer):
    """Armed for ~0 s so the real task fires without the test waiting."""
    result = await skill.execute(SkillRequest(
        text="set a timer for 1 second",
        match=skill.match("set a timer for 1 second")))
    assert result.success

    for _ in range(40):                          # up to ~2 s, exits as soon as it fires
        if announcer.said:
            break
        await asyncio.sleep(0.05)

    assert len(announcer.said) == 1, f"announced {len(announcer.said)} times"
    assert skill._timers == {}, "a fired timer must not stay in the list"


@pytest.mark.asyncio
async def test_a_cancelled_timer_never_announces(skill, announcer):
    await _say(skill, "set a timer for 1 second")
    await _say(skill, "cancel all timers")
    await asyncio.sleep(1.3)
    assert announcer.said == [], "a cancelled timer still fired"
