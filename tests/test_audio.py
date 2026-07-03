"""Unit tests for the capture-until-silence VAD logic (no audio hardware)."""

from __future__ import annotations

import numpy as np

from voice.audio import FRAME_SAMPLES, frame_rms, record_until_silence


class FakeMic:
    """Replays a scripted list of int16 frames as if it were a live mic."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self.sample_rate = 16000
        self.frame_samples = FRAME_SAMPLES
        self._frames = frames
        self._i = 0

    def read_frame(self) -> np.ndarray:
        # After the script runs out, return silence forever (mic keeps streaming).
        if self._i < len(self._frames):
            frame = self._frames[self._i]
            self._i += 1
            return frame
        return np.zeros(self.frame_samples, dtype=np.int16)


def _loud() -> np.ndarray:
    return np.full(FRAME_SAMPLES, 8000, dtype=np.int16)


def _quiet() -> np.ndarray:
    return np.zeros(FRAME_SAMPLES, dtype=np.int16)


def _noise(amplitude: int) -> np.ndarray:
    """A steady ambient-noise frame at the given int16 amplitude."""
    return np.full(FRAME_SAMPLES, amplitude, dtype=np.int16)


def test_frame_rms_bounds():
    assert frame_rms(_quiet()) == 0.0
    assert 0.0 < frame_rms(_loud()) < 1.0


def test_records_speech_then_stops_on_silence():
    # A quiet ambient lead-in (the noise-floor calibration window), 5 loud
    # frames of speech, then quiet long enough to trigger end-of-speech.
    frames = [_quiet()] * 4 + [_loud()] * 5 + [_quiet()] * 40
    mic = FakeMic(frames)
    audio = record_until_silence(
        mic, silence_threshold=0.01, silence_duration_s=0.5, max_seconds=5
    )
    assert audio.dtype == np.float32
    assert audio.size > 0
    # Captured the speech plus the trailing silence window, not all 49 frames.
    assert audio.size < 49 * FRAME_SAMPLES


def test_returns_empty_when_no_speech():
    mic = FakeMic([_quiet()] * 200)
    audio = record_until_silence(
        mic, silence_threshold=0.01, silence_duration_s=0.5, start_timeout_s=0.5
    )
    assert audio.size == 0


def test_detects_onset_over_loud_ambient_noise():
    # Ambient hiss well above the fixed silence_threshold: with a fixed
    # threshold the noise itself would register as endless "speech". The
    # adaptive floor (ambient * 2.5) must still catch real speech on top of it
    # and stop once the room falls back to just ambient noise.
    ambient = [_noise(1200)] * 4          # rms ~0.037 >> threshold 0.01
    speech = [_loud()] * 5                # rms ~0.244 > ambient * 2.5
    trailing = [_noise(1200)] * 40        # back to just the noise floor
    mic = FakeMic(ambient + speech + trailing)
    audio = record_until_silence(
        mic, silence_threshold=0.01, silence_duration_s=0.5, max_seconds=5
    )
    assert audio.size > 0
    # Recording ended on the ambient "silence", not at the max_seconds cap.
    assert audio.size < len(ambient + speech + trailing) * FRAME_SAMPLES


def test_ambient_noise_alone_never_triggers_onset():
    # A noisy room with nobody speaking: the raised threshold means the hiss
    # is not mistaken for speech, so the recorder times out empty.
    mic = FakeMic([_noise(1200)] * 200)
    audio = record_until_silence(
        mic, silence_threshold=0.01, silence_duration_s=0.5, start_timeout_s=1.0
    )
    assert audio.size == 0
