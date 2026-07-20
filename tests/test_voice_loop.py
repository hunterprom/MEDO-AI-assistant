"""M5 refactor guard: VoiceLoop wires up without any voice hardware/models."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

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
