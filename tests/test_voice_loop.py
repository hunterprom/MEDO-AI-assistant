"""M5 refactor: voice/loop.py — construction stays light and mockable.

The heavy audio stack (sounddevice/whisper/piper) is imported lazily inside
``run()``, so constructing a :class:`VoiceLoop` with mocks must work on any
machine, deps installed or not — that's what keeps main.py's wiring testable.
"""

from __future__ import annotations


class _Dummy:
    """Stands in for settings/router/sm/announcer/ui/log/console."""


def test_voice_loop_constructs_with_mocked_wiring():
    from voice.loop import VoiceLoop

    loop = VoiceLoop(
        settings=_Dummy(), router=_Dummy(), sm=_Dummy(), announcer=_Dummy(),
        ui=_Dummy(), log=_Dummy(), console=_Dummy(), wake_event=None,
    )
    # No audio stack touched at construction time.
    assert loop._wake is None and loop._stt is None
    assert loop._tts is None and loop._edge is None
    assert loop._barge_hit is False


def test_voice_loop_phases_and_tuning_preserved():
    from voice.loop import VoiceLoop

    # The named phases of the pipeline all exist...
    for phase in ("_load_models", "_wait_for_wake", "_capture", "_transcribe",
                  "_route_streaming", "_speak", "_play_interruptible", "run"):
        assert callable(getattr(VoiceLoop, phase)), phase
    # ...and the barge-in tuning survived the extraction unchanged.
    assert VoiceLoop.BARGE_WAKE_THRESHOLD == 0.25
    assert VoiceLoop.BARGE_RMS_RATIO == 3.0
    assert VoiceLoop.BARGE_MIN_RMS == 0.02
    assert VoiceLoop.BARGE_HOLD_FRAMES == 4
