"""M5 refactor guard: VoiceLoop wires up without any voice hardware/models."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.config import load_settings
from core.metrics import LatencyLog
from voice.loop import VoiceLoop


def _make_loop(**kwargs) -> VoiceLoop:
    return VoiceLoop(
        load_settings(),
        router=MagicMock(),          # never called at construction time
        sm=MagicMock(),
        announcer=MagicMock(),
        ui=MagicMock(),
        log=LatencyLog(),
        **kwargs,
    )


def test_constructs_with_mocks_and_loads_nothing():
    loop = _make_loop()
    # Heavy engines must load in run(), not __init__ — constructing a loop on
    # a machine without audio deps (CI, tests) must always work.
    assert loop._wakeword is None and loop._stt is None
    assert loop._tts is None and loop._edge is None
    assert loop._barged_in is False


def test_accepts_the_manual_wake_event():
    event = threading.Event()
    loop = _make_loop(wake_event=event)
    assert loop._wake_event is event


def test_speak_is_wired_for_announcements():
    # main.py hands VoiceLoop._speak to the Announcer so timers are spoken;
    # the method must exist as a bound coroutine function before run().
    import inspect

    loop = _make_loop()
    assert inspect.iscoroutinefunction(loop._speak)


def test_main_delegates_to_voice_loop():
    # main.py must not grow its own voice loop back.
    import main

    assert not hasattr(main, "run_voice")
    assert main.VoiceLoop is VoiceLoop


def test_custom_wake_model_path_resolves(tmp_path):
    """M6 scaffolding: wakeword.phrase may be a path to a trained model."""
    from voice.wakeword import _resolve_model_path

    onnx = tmp_path / "hey_medo.onnx"
    onnx.write_bytes(b"fake")
    assert _resolve_model_path(str(onnx)) == str(onnx)
    tflite = tmp_path / "hey_medo.tflite"
    tflite.write_bytes(b"fake")
    assert _resolve_model_path(str(tflite)) == str(tflite)


# --- barge-in: "medo" interrupts, noise does not (default = wake) -------------


class _WakeFrame:
    """A mic frame carrying the wake word. A plain object (not an ndarray) so it
    can be tagged; the voice-energy path is never reached on it in wake mode."""

    rms = 0.0


class _FakeWake:
    """Scores high only on a frame that IS the wake word ("medo")."""

    def predict(self, frame):
        return 0.9 if isinstance(frame, _WakeFrame) else 0.0

    def reset(self):
        pass


def _quiet_frame():
    return np.zeros(1280, dtype=np.int16)      # frames are int16 PCM


def _loud_frame():
    # rms 10000/32768 ≈ 0.3 — well above barge_min_rms once baseline settles
    return np.full(1280, 10000, dtype=np.int16)


def _cfg(mode):
    s = load_settings()
    s.audio.barge_mode = mode
    return s.audio


def test_wake_mode_ignores_loud_noise():
    # The default: a random word / noise stream never interrupts.
    frames = [_loud_frame() for _ in range(30)]
    why = VoiceLoop.watch_for_barge(iter(frames), _FakeWake(), _cfg("wake"))
    assert why is None


def test_wake_mode_interrupts_on_the_wake_word():
    frames = [_quiet_frame(), _quiet_frame(), _WakeFrame(), _quiet_frame()]
    why = VoiceLoop.watch_for_barge(iter(frames), _FakeWake(), _cfg("wake"))
    assert why == "wake word over playback"


def test_voice_mode_still_interrupts_on_sustained_noise():
    # Opt back in to the old behaviour: quiet leakage first (baseline settles
    # low), then sustained loud speech trips the energy bar.
    frames = [_quiet_frame() for _ in range(8)] + [_loud_frame() for _ in range(8)]
    why = VoiceLoop.watch_for_barge(iter(frames), _FakeWake(), _cfg("voice"))
    assert why == "voice over playback"


def test_off_mode_ignores_both_wake_and_noise():
    frames = [_WakeFrame()] + [_loud_frame() for _ in range(30)]
    why = VoiceLoop.watch_for_barge(iter(frames), _FakeWake(), _cfg("off"))
    assert why is None


def test_barge_needs_sustained_wake_frames_when_configured():
    # A single wake-scoring frame (a transient) must not interrupt when the
    # wake word requires several consecutive frames.
    wake = _FakeWake()
    wake.trigger_frames = 2
    one = [_quiet_frame(), _WakeFrame(), _quiet_frame(), _quiet_frame()]
    assert VoiceLoop.watch_for_barge(iter(one), wake, _cfg("wake")) is None
    two = [_WakeFrame(), _WakeFrame(), _quiet_frame()]
    assert VoiceLoop.watch_for_barge(iter(two), wake, _cfg("wake")) \
        == "wake word over playback"


def test_explicit_interrupt_signal_wins_in_every_mode():
    for mode in ("wake", "voice", "off"):
        ev = threading.Event()
        ev.set()
        why = VoiceLoop.watch_for_barge(
            iter([_quiet_frame()]), _FakeWake(), _cfg(mode), wake_event=ev)
        assert why == "interrupt signal"
        assert ev.is_set() is False        # consumed


def test_default_barge_mode_is_wake():
    assert load_settings().audio.barge_mode == "wake"


# --- after an interrupt: go to standby, don't sit listening -------------------

def _relisten_loop(*, awaiting=False, continuous=False, listen_after_barge=False):
    loop = _make_loop()
    loop._router.awaiting_confirmation = awaiting     # MagicMock attr -> real bool
    loop._modes.continuous = continuous
    loop._settings.conversation.listen_after_barge = listen_after_barge
    return loop


def test_interrupt_returns_to_standby_by_default():
    # The complaint this fixes: after saying "medo" to cut MEDO off (and NOT
    # saying a command), MEDO must not keep listening to the silence.
    loop = _relisten_loop()
    assert loop._should_relisten(barged_in=True) is False


def test_interrupt_can_keep_listening_when_opted_in():
    loop = _relisten_loop(listen_after_barge=True)
    assert loop._should_relisten(barged_in=True) is True


def test_a_normal_turn_returns_to_standby():
    loop = _relisten_loop()
    assert loop._should_relisten(barged_in=False) is False


def test_confirmation_still_relistens_even_after_a_barge():
    # "are you sure?" always keeps the mic open, regardless of the barge rule.
    loop = _relisten_loop(awaiting=True)
    assert loop._should_relisten(barged_in=True) is True


def test_continuous_mode_always_relistens():
    loop = _relisten_loop(continuous=True)
    assert loop._should_relisten(barged_in=False) is True


def test_default_listen_after_barge_is_off():
    assert load_settings().conversation.listen_after_barge is False


def test_wake_requires_sustained_frames_by_default():
    # The anti-false-wake default: more than one frame must clear the threshold.
    assert load_settings().wakeword.trigger_frames >= 2


def test_missing_custom_model_falls_back_to_bundled(tmp_path, monkeypatch):
    """A configured hey_medo.onnx that isn't trained yet must NOT crash voice
    ('Could not find pretrained model ...') — it falls back to bundled
    hey_jarvis so MEDO keeps listening until the model is trained."""
    import voice.wakeword as ww

    monkeypatch.setattr(
        ww, "_bundled_match",
        lambda name: f"/bundled/{name}_v0.1.onnx" if name == "hey_jarvis" else None,
    )
    resolved = ww._resolve_model_path(str(tmp_path / "hey_medo.onnx"))  # missing
    assert resolved == "/bundled/hey_jarvis_v0.1.onnx"


# --- STT wake-confirm: verify the WORDS, kill loud-noise triggers -------------

from voice.wakeword import wake_phrase_confirmed  # noqa: E402


@pytest.mark.parametrize("text", [
    "hey medo", "hey medo what time is it", "medo", "Hey, Medo!",
    "hey meadow", "hey medoh", "медо",
])
def test_wake_confirm_accepts_a_real_wake(text):
    assert wake_phrase_confirmed(text, "models/wakeword/hey_medo.onnx") is True


@pytest.mark.parametrize("text", [
    "", "   ", "you", "thanks for watching", "let me go", "hello there",
    "the meeting is at ten",
])
def test_wake_confirm_rejects_noise_and_unrelated_speech(text):
    # empty = non-speech (door slam / music / clap); the rest lack the phrase.
    assert wake_phrase_confirmed(text, "models/wakeword/hey_medo.onnx") is False


def test_wake_confirm_uses_the_configured_phrase():
    assert wake_phrase_confirmed("hey jarvis", "hey_jarvis") is True
    assert wake_phrase_confirmed("hey medo", "hey_jarvis") is False


def test_confirm_wake_fails_open_when_stt_errors():
    loop = _make_loop()
    class _Boom:
        def transcribe(self, audio):
            raise RuntimeError("stt down")
    loop._stt = _Boom()
    # a broken transcriber must not make MEDO unwakeable
    assert loop._confirm_wake(np.zeros(1280, dtype=np.int16)) is True


def test_confirm_wake_rejects_when_transcript_lacks_the_phrase():
    loop = _make_loop()
    class _Stt:
        def transcribe(self, audio):
            return "just some background chatter"
    loop._stt = _Stt()
    assert loop._confirm_wake(np.zeros(1280, dtype=np.int16)) is False


def test_default_stt_confirm_is_on():
    assert load_settings().wakeword.stt_confirm is True
