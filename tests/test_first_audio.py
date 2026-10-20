"""P2-1: measuring what streaming bought, and stopping the filler talking over it.

Sentence streaming was already in place; two things were missing around it.

**Nothing measured it.** ``route_ms`` is how long the whole reply took to
generate, which streaming deliberately decoupled from the wait — a streamed
answer can start speaking at two seconds and finish generating at twelve. The
number a person experiences is time to FIRST audio, and it was not recorded, so
"is streaming helping?" had no answer.

**The filler could talk over the answer.** It checked whether anything had been
spoken only at its own wake points, so a first sentence queued at 2.1 s could
not stop a filler whose timer expired at 2.0 s. It now waits on the sentence
too, and loses that race by construction.
"""

from __future__ import annotations

import asyncio

import pytest

from core.config import load_settings
from core.metrics import LatencyLog, MetricsStore, TurnTimings


# --- the timing itself --------------------------------------------------------


def test_first_audio_is_recorded_per_turn(tmp_path):
    store = MetricsStore(tmp_path / "m.db")
    store.record_first_audio("LLM", 1800.0, streamed=True, total_ms=9000.0)
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT path, streamed, first_audio_ms, total_ms FROM first_audio"
        ).fetchall()
    assert rows == [("LLM", 1, 1800.0, 9000.0)]


def test_the_report_separates_streamed_from_whole_replies(tmp_path):
    """The comparison that says whether streaming is earning its complexity."""
    store = MetricsStore(tmp_path / "m.db")
    store.record_first_audio("LLM", 1800.0, streamed=True, total_ms=9000.0)
    store.record_first_audio("LLM", 8600.0, streamed=False, total_ms=8600.0)
    report = store.first_audio_report()
    assert "streamed" in report and "whole" in report
    assert "1,800 ms" in report and "8,600 ms" in report


def test_the_report_says_so_when_empty(tmp_path):
    assert "No first-audio" in MetricsStore(tmp_path / "m.db").first_audio_report()


def test_a_broken_store_does_not_raise(tmp_path):
    store = MetricsStore(tmp_path / "nonexistent-dir" / "m.db")
    store.record_first_audio("LLM", 1.0)        # must not raise


def test_turn_timings_carry_first_audio():
    log = LatencyLog()
    log.record(TurnTimings(path="LLM", route_ms=9000.0, first_audio_ms=1800.0))
    log.record(TurnTimings(path="LLM", route_ms=7000.0, first_audio_ms=2200.0))
    assert log.summary()["first_audio_ms"] == 2000.0


def test_first_audio_is_absent_when_a_turn_never_made_a_sound():
    log = LatencyLog()
    log.record(TurnTimings(path="FAST", route_ms=3.0))
    assert log.summary()["first_audio_ms"] is None


# --- the filler stands down the moment a sentence is queued -------------------


class _Fillerish:
    """The wait logic from voice/loop.py, isolated from audio.

    ``_sleep_or_done`` is a closure inside ``_run_turn`` and cannot be reached
    from a test without standing up the whole voice stack. Its RULE is what
    matters and is reproduced exactly here: sleep, but wake early if either the
    reply has landed or the first sentence has been queued.
    """

    def __init__(self) -> None:
        self.route_done = asyncio.Event()
        self.first_sentence = asyncio.Event()

    async def sleep_or_done(self, seconds: float) -> bool:
        waiters = [asyncio.ensure_future(self.route_done.wait()),
                   asyncio.ensure_future(self.first_sentence.wait())]
        try:
            done, _ = await asyncio.wait(waiters, timeout=seconds,
                                         return_when=asyncio.FIRST_COMPLETED)
            return bool(done)
        finally:
            for w in waiters:
                w.cancel()


@pytest.mark.asyncio
async def test_the_filler_waits_when_nothing_has_happened():
    fill = _Fillerish()
    assert await fill.sleep_or_done(0.01) is False      # timed out: speak up


@pytest.mark.asyncio
async def test_the_filler_stands_down_when_the_reply_lands():
    fill = _Fillerish()
    fill.route_done.set()
    assert await fill.sleep_or_done(5.0) is True


@pytest.mark.asyncio
async def test_the_filler_stands_down_when_the_first_sentence_is_queued():
    """The bug: a sentence at 2.1 s could not cancel a filler firing at 2.0 s."""
    fill = _Fillerish()

    async def stream_soon():
        await asyncio.sleep(0.01)
        fill.first_sentence.set()

    task = asyncio.create_task(stream_soon())
    assert await fill.sleep_or_done(5.0) is True
    await task


@pytest.mark.asyncio
async def test_the_filler_does_not_leak_pending_waiters():
    """It runs on every LLM turn; a leaked task per turn is a slow leak."""
    fill = _Fillerish()
    before = len(asyncio.all_tasks())
    for _ in range(20):
        await fill.sleep_or_done(0.001)
    await asyncio.sleep(0)
    assert len(asyncio.all_tasks()) <= before + 1


# --- the retuned default ------------------------------------------------------


def test_the_filler_delay_stays_short():
    """The filler must not become the dead air it exists to fill.

    This delay was briefly raised to 3.5 s on the theory that streaming had
    made a 2 s filler arrive just before the answer. That reasoning was wrong:
    once the filler stands down on the first queued sentence (the tests above),
    it cannot talk over the reply at ANY delay — so a longer one buys nothing
    and costs 1.5 s of extra silence on every slow answer. Reported as "voice
    delay", and it was.
    """
    filler = load_settings().filler
    assert filler.big_delay_s <= 2.5
    assert filler.big_delay_s < filler.delay_s
