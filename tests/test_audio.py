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


def test_normalize_peak_boosts_quiet_and_caps_loud():
    from voice.audio import normalize_peak

    quiet = np.array([0.02, -0.05, 0.03], dtype=np.float32)    # webcam-mic level
    boosted = normalize_peak(quiet, target=0.6)
    assert abs(float(np.max(np.abs(boosted))) - 0.6) < 1e-6
    loud = np.array([0.95, -0.9], dtype=np.float32)
    assert abs(float(np.max(np.abs(normalize_peak(loud, target=0.6)))) - 0.6) < 1e-6


def test_normalize_peak_gain_is_capped():
    from voice.audio import normalize_peak

    # A clip that barely crossed the recording gate must not be blasted all the
    # way to target: the gain stops at max_gain so noise stays quiet.
    faint = np.array([0.002, -0.001], dtype=np.float32)
    out = normalize_peak(faint, target=0.6, max_gain=25.0)
    assert abs(float(np.max(np.abs(out))) - 0.05) < 1e-6   # 0.002 * 25


def test_normalize_peak_leaves_silence_and_empty_alone():
    from voice.audio import normalize_peak

    silence = np.zeros(100, dtype=np.float32)
    assert np.array_equal(normalize_peak(silence), silence)  # no noise amplification
    empty = np.zeros(0, dtype=np.float32)
    assert normalize_peak(empty).size == 0


def _fake_inputs(monkeypatch, devices):
    """Patch the usable-device table: [(index, name, hostapi), ...]."""
    import voice.audio as va

    monkeypatch.setattr(va, "_usable_inputs", lambda: devices)


def test_resolve_input_device_priority_chain(monkeypatch):
    import pytest

    from voice.audio import resolve_input_device

    _fake_inputs(monkeypatch, [(5, "Microphone (FHD Webcam)", "MME")])
    # Headphones absent -> the chain falls through to the webcam.
    assert resolve_input_device(["A25", "FHD Webcam"]) == 5
    # Headphones connect -> the chain now prefers them.
    _fake_inputs(monkeypatch, [
        (5, "Microphone (FHD Webcam)", "MME"),
        (7, "Headset (A25 Hands-Free)", "MME"),
    ])
    assert resolve_input_device(["A25", "FHD Webcam"]) == 7
    # Nothing in the chain available -> loud error, not a silent dead default.
    _fake_inputs(monkeypatch, [(3, "Microphone (Other)", "MME")])
    with pytest.raises(RuntimeError):
        resolve_input_device(["A25", "FHD Webcam"])


def test_resolve_input_device_passthrough_and_single(monkeypatch):
    from voice.audio import resolve_input_device

    assert resolve_input_device(None) is None
    assert resolve_input_device(9) == 9
    _fake_inputs(monkeypatch, [(2, "Microphone (NVIDIA Broadcast)", "MME")])
    assert resolve_input_device("nvidia") == 2


def test_lists_every_mic_once_across_host_apis(monkeypatch):
    """Detect ALL mics: collapse per-host-API duplicates to one entry each, and
    surface a mic that only exists on WASAPI (a Bluetooth headset) — the case
    the old MME-only picker dropped."""
    from voice.audio import list_input_devices

    _fake_inputs(monkeypatch, [
        # One physical mic on two host APIs: MME truncates the name to 31 chars,
        # WASAPI gives the full name. They must collapse to a single entry.
        (0, "Microphone (High Definition Aud", "MME"),               # 31-char MME
        (5, "Microphone (High Definition Audio Device)", "Windows WASAPI"),
        # A headset mic that only appears under WASAPI — invisible before.
        (9, "Headset (A25 Stereo)", "Windows WASAPI"),
    ])
    devices = list_input_devices()
    names = [d["name"] for d in devices]
    assert len(devices) == 2                          # collapsed, not 3 rows
    assert any("A25" in n for n in names)             # WASAPI-only mic detected
    hd = next(d for d in devices if "High Definition" in d["name"])
    assert hd["index"] == 0                           # kept the openable MME index
    assert hd["name"] == "Microphone (High Definition Aud"  # truncation-safe name


def test_resolve_prefers_the_openable_host_api(monkeypatch):
    """A mic enumerated on several host APIs resolves to the one we can open at
    16 kHz (MME > DirectSound > WASAPI), not whichever came first."""
    from voice.audio import resolve_input_device

    _fake_inputs(monkeypatch, [
        (7, "Microphone (High Definition Audio Device)", "Windows WASAPI"),
        (3, "Microphone (High Definition Aud", "MME"),
        (5, "Microphone (High Definition Audi", "Windows DirectSound"),
    ])
    assert resolve_input_device("high definition") == 3
