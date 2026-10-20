"""S3 — interrupting MEDO actually interrupts it.

Barge-in already stopped PLAYBACK. What S2 added, and what this pins down, is
that it must also stop everything queued behind the playback:

* **In-flight synthesis.** The speaker synthesizes one sentence ahead, and on
  the English/Japanese Kokoro voice that takes ~3.3 s. Noticing the interrupt
  only when that finishes is most of the lag a barge-in used to have.
* **Queued audio.** A sentence that reaches the speakers AFTER the user cut in
  is the single thing that makes an assistant feel like it wasn't listening.
* **The text queue.** ``on_delta`` keeps arriving from the LLM after the
  interrupt; those sentences must not be synthesized either.

The mechanism is an ``asyncio.Event`` alongside the existing bool. The bool is
only ever read at the top of a loop, so a task parked on ``queue.get()`` or on
a 3-second synthesis never sees it — which is precisely the "MEDO carries on
for a moment" feel.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.config import load_settings
from core.metrics import LatencyLog
from voice.loop import VoiceLoop

#: Barge-in must land well inside a sentence, not at the end of one.
BOUND_S = 0.25


def _loop() -> VoiceLoop:
    loop = VoiceLoop(load_settings(), router=MagicMock(), sm=MagicMock(),
                     announcer=MagicMock(), ui=MagicMock(), log=LatencyLog())
    loop._sm.transition = _noop
    return loop


async def _noop(*a, **k):
    return None


class _Speaker:
    """Stands in for synthesis + playback, recording what actually happened."""

    def __init__(self, loop: VoiceLoop, synth_s: float = 0.05,
                 play_s: float = 0.05) -> None:
        self.synthesized: list[str] = []
        self.played: list[str] = []
        self._synth_s = synth_s
        self._play_s = play_s
        loop._synthesize = self._synthesize
        loop._play = self._play

    async def _synthesize(self, text, language):
        self.synthesized.append(text)
        await asyncio.sleep(self._synth_s)
        return np.ones(8, dtype=np.int16), text          # carry the text along

    async def _play(self, wav, sr):
        self.played.append(sr)
        await asyncio.sleep(self._play_s)


async def _feed(queue: asyncio.Queue, *sentences: str, end: bool = True):
    for s in sentences:
        queue.put_nowait(s)
    if end:
        queue.put_nowait(None)


# --- the primitives ----------------------------------------------------------


@pytest.mark.asyncio
async def test_race_returns_the_result_when_nothing_interrupts():
    async def work():
        return "done"

    cut, value = await VoiceLoop._race(work(), asyncio.Event())
    assert (cut, value) == (False, "done")


@pytest.mark.asyncio
async def test_race_gives_up_the_moment_the_event_sets():
    event = asyncio.Event()

    async def slow():
        await asyncio.sleep(10)
        return "too late"

    async def cut_in():
        await asyncio.sleep(0.02)
        event.set()

    t0 = time.perf_counter()
    (cut, value), _ = await asyncio.gather(VoiceLoop._race(slow(), event), cut_in())
    assert cut is True and value is None
    assert time.perf_counter() - t0 < BOUND_S


@pytest.mark.asyncio
async def test_race_does_not_start_work_that_is_already_interrupted():
    event = asyncio.Event()
    event.set()
    started = []

    async def work():
        started.append(1)
        return "x"

    cut, value = await VoiceLoop._race(work(), event)
    assert cut is True and value is None and started == []


def test_drain_empties_a_queue_and_counts_what_it_dropped():
    q: asyncio.Queue = asyncio.Queue()
    for i in range(4):
        q.put_nowait(i)
    assert VoiceLoop._drain(q) == 4
    assert q.empty()
    assert VoiceLoop._drain(q) == 0


def test_interrupt_sets_both_views_of_the_same_fact():
    loop = _loop()
    assert not loop._barged_in and not loop._interrupted.is_set()
    loop.interrupt()
    assert loop._barged_in and loop._interrupted.is_set()
    loop.interrupt()                       # idempotent
    assert loop._barged_in


# --- the speaker -------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_whole_reply_is_spoken_when_nobody_interrupts():
    loop = _loop()
    speaker = _Speaker(loop)
    q: asyncio.Queue = asyncio.Queue()
    await _feed(q, "One.", "Two.", "Three.")
    await loop._stream_speaker(q)
    assert speaker.synthesized == ["One.", "Two.", "Three."]
    assert len(speaker.played) == 3


@pytest.mark.asyncio
async def test_nothing_queued_is_spoken_after_the_user_cuts_in():
    loop = _loop()
    # Playback is long enough that the interrupt lands during sentence one.
    speaker = _Speaker(loop, synth_s=0.01, play_s=0.20)
    q: asyncio.Queue = asyncio.Queue()
    await _feed(q, "One.", "Two.", "Three.", "Four.")

    async def cut_in():
        await asyncio.sleep(0.05)
        loop.interrupt()

    await asyncio.gather(loop._stream_speaker(q), cut_in())
    # Sentence one was already playing and finishes; nothing after it plays.
    assert len(speaker.played) == 1, speaker.played
    assert q.empty(), "the text queue must be flushed too"


@pytest.mark.asyncio
async def test_an_interrupt_cancels_synthesis_that_is_still_running():
    # THE S2 regression: with synthesis a sentence ahead, a barge-in during a
    # 3.3 s Kokoro synthesis used to wait for it to finish.
    loop = _loop()
    speaker = _Speaker(loop, synth_s=5.0, play_s=0.01)
    q: asyncio.Queue = asyncio.Queue()
    await _feed(q, "One.", "Two.")

    async def cut_in():
        await asyncio.sleep(0.05)
        loop.interrupt()

    t0 = time.perf_counter()
    await asyncio.gather(loop._stream_speaker(q), cut_in())
    elapsed = time.perf_counter() - t0
    assert elapsed < BOUND_S, f"took {elapsed:.2f}s — waited for the synthesis"
    assert speaker.played == []


@pytest.mark.asyncio
async def test_an_interrupt_while_waiting_for_the_llm_returns_at_once():
    # The queue is EMPTY and the LLM is still generating. A task parked on
    # queue.get() has to wake on the interrupt, not on the next sentence.
    loop = _loop()
    _Speaker(loop)
    q: asyncio.Queue = asyncio.Queue()        # nothing fed, no sentinel

    async def cut_in():
        await asyncio.sleep(0.05)
        loop.interrupt()

    t0 = time.perf_counter()
    await asyncio.gather(loop._stream_speaker(q), cut_in())
    assert time.perf_counter() - t0 < BOUND_S


@pytest.mark.asyncio
async def test_an_interrupt_before_a_word_is_spoken_speaks_nothing():
    loop = _loop()
    speaker = _Speaker(loop)
    loop.interrupt()
    q: asyncio.Queue = asyncio.Queue()
    await _feed(q, "One.", "Two.")
    await loop._stream_speaker(q)
    assert speaker.played == [] and speaker.synthesized == []


@pytest.mark.asyncio
async def test_the_speaker_finishes_even_with_a_full_queue_and_no_consumer():
    # The sentinel must not deadlock on a bounded queue whose consumer left.
    loop = _loop()
    _Speaker(loop, synth_s=0.01, play_s=0.01)
    q: asyncio.Queue = asyncio.Queue()
    await _feed(q, *[f"S{i}." for i in range(12)])
    await asyncio.wait_for(loop._stream_speaker(q), timeout=5)


# --- echo: MEDO must not interrupt itself ------------------------------------


class _Wake:
    """A wake model that scores whatever it is told to."""

    def __init__(self, scores) -> None:
        self._scores = list(scores)
        self.threshold = 0.5

    def predict(self, frame):
        return self._scores.pop(0) if self._scores else 0.0

    def reset(self):
        pass


def _frames(level: float, count: int) -> list:
    """``count`` mic frames at ``level`` (0.0-1.0), in the units the mic yields.

    `Microphone` opens the stream as **int16**, and `frame_rms` normalises int16
    to 0.0-1.0 by dividing by 32768. Building frames as float amplitudes instead
    made every frame read as ~1e-6 — far below `barge_min_rms` — so the
    energy-barge tests could never fire. One of them failed outright; the ones
    asserting `None` passed for the wrong reason, proving nothing.
    """
    return [np.full(1280, int(level * 32768), dtype=np.int16)
            for _ in range(count)]


def _silence(count: int) -> list:
    return [np.zeros(1280, dtype=np.int16) for _ in range(count)]


def _audio_cfg(**kw):
    cfg = load_settings().audio
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_medos_own_playback_does_not_trigger_a_barge_in():
    # Speaker leakage into the open mic is loud but scores nothing on the wake
    # model. In the default "wake" mode, loudness alone must never interrupt.
    frames = _frames(0.3, 60)
    why = VoiceLoop.watch_for_barge(frames, _Wake([]), _audio_cfg(barge_mode="wake"))
    assert why is None


def test_one_stray_high_scoring_frame_is_not_an_interruption():
    # A blip must not cut the reply off; the phrase has to be sustained.
    frames = _silence(20)
    why = VoiceLoop.watch_for_barge(
        frames, _Wake([0.9, 0.0, 0.0, 0.9]),
        _audio_cfg(barge_mode="wake", barge_wake_frames=3))
    assert why is None


def test_a_sustained_wake_word_over_the_reply_does_interrupt():
    frames = _silence(20)
    why = VoiceLoop.watch_for_barge(
        frames, _Wake([0.9] * 5),
        _audio_cfg(barge_mode="wake", barge_wake_frames=3))
    assert why == "wake word over playback"


def test_voice_mode_ignores_the_replys_own_steady_leakage():
    # In "voice" mode the bar adapts to MEDO's own loudness, so its playback —
    # steady, however loud — never crosses it.
    frames = _frames(0.25, 60)
    why = VoiceLoop.watch_for_barge(frames, _Wake([]), _audio_cfg(barge_mode="voice"))
    assert why is None


def test_voice_mode_still_hears_someone_talking_over_it():
    quiet, shout = _frames(0.05, 12), _frames(0.40, 12)
    why = VoiceLoop.watch_for_barge(quiet + shout, _Wake([]),
                                    _audio_cfg(barge_mode="voice"))
    assert why == "voice over playback"


def test_barge_mode_off_ignores_the_microphone_entirely():
    frames = _frames(0.9, 40)
    why = VoiceLoop.watch_for_barge(frames, _Wake([0.99] * 40),
                                    _audio_cfg(barge_mode="off"))
    assert why is None


def test_the_hud_interrupt_signal_works_in_every_mode():
    signal = __import__("threading").Event()
    signal.set()
    frames = _silence(4)
    why = VoiceLoop.watch_for_barge(frames, _Wake([]),
                                    _audio_cfg(barge_mode="off"), signal)
    assert why == "interrupt signal"
