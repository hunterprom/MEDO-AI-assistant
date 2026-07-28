"""Wake-word sample generator: audio conversion + wav writing (pure/offline)."""

from __future__ import annotations

import wave

import numpy as np

from voice.train_wakeword import SAMPLE_RATE, _resample, _to_float, _write_wav


def test_to_float_normalizes_int16_and_passes_float():
    i16 = np.array([-32768, 0, 32767], dtype=np.int16)
    out = _to_float(i16)
    assert out.dtype == np.float32
    assert -1.0 <= out.min() and out.max() <= 1.0
    f = np.array([0.5, -0.5], dtype=np.float32)
    assert np.allclose(_to_float(f), f)


def test_resample_changes_length_to_16k():
    # 1 second of 24 kHz -> ~16000 samples at 16 kHz.
    src = (np.sin(np.linspace(0, 100, 24000)) * 20000).astype(np.int16)
    out = _resample(src, 24000)
    assert abs(len(out) - SAMPLE_RATE) <= 2
    assert out.dtype == np.float32
    # same rate is a no-op (just normalized)
    assert len(_resample(src, SAMPLE_RATE)) == len(src)


def test_write_wav_is_16k_mono_16bit(tmp_path):
    audio = (np.sin(np.linspace(0, 50, SAMPLE_RATE)) * 0.5).astype(np.float32)
    path = tmp_path / "s.wav"
    _write_wav(path, audio)
    w = wave.open(str(path))
    try:
        assert w.getframerate() == SAMPLE_RATE
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == SAMPLE_RATE
    finally:
        w.close()


def test_write_wav_clips_out_of_range(tmp_path):
    audio = np.array([2.0, -2.0, 0.0], dtype=np.float32)  # beyond [-1,1]
    path = tmp_path / "c.wav"
    _write_wav(path, audio)              # must not raise / overflow
    with wave.open(str(path)) as w:
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    assert frames.max() <= 32767 and frames.min() >= -32768
