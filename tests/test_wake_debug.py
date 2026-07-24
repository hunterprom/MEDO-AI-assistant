"""S1 — wake-word activation capture + report (diagnosis before tuning)."""

from __future__ import annotations

import wave
from datetime import datetime

import numpy as np

from voice.wake_debug import (
    WakeCaptureLog,
    format_report,
    load_labels,
    summarize,
    write_wav,
)


def _frame(level=8000, n=1280):
    return np.full(n, level, dtype=np.int16)


# -- capture writes a row + a wav ----------------------------------------------

def test_enabled_capture_writes_a_row_and_a_wav(tmp_path):
    log = WakeCaptureLog(tmp_path, enabled=True, sample_rate=16000)
    when = datetime(2026, 7, 24, 9, 30, 0)
    wav = log.record(0.83, 0.12, _frame(), when=when)
    assert wav is not None and wav.exists()
    rows = log.rows()
    assert len(rows) == 1
    assert rows[0]["score"] == 0.83 and rows[0]["wav"] == wav.name
    assert rows[0]["rms"] == 0.12 and "2026-07-24" in rows[0]["time"]
    # the wav is real, mono, 16-bit, 16 kHz
    with wave.open(str(wav), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        assert w.getframerate() == 16000 and w.getnframes() == 1280


def test_disabled_capture_writes_nothing(tmp_path):
    log = WakeCaptureLog(tmp_path, enabled=False)
    assert log.record(0.9, 0.2, _frame()) is None
    assert log.rows() == []
    assert not any(tmp_path.iterdir())          # no files at all


def test_capture_appends_across_activations(tmp_path):
    log = WakeCaptureLog(tmp_path, enabled=True)
    for i in range(3):
        log.record(0.6 + i * 0.1, 0.1, _frame(),
                   when=datetime(2026, 7, 24, 9, 30, i))
    assert len(log.rows()) == 3


def test_write_wav_roundtrips(tmp_path):
    path = tmp_path / "x.wav"
    write_wav(path, _frame(level=1234, n=800), 16000)
    with wave.open(str(path), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    assert data.tolist() == [1234] * 800


# -- labels --------------------------------------------------------------------

def test_load_labels_reads_true_false(tmp_path):
    (tmp_path / "labels.csv").write_text(
        "wav,label\nwake-a.wav,true\nwake-b.wav,false\nwake-c.wav,real\n",
        encoding="utf-8")
    labels = load_labels(tmp_path)
    assert labels == {"wake-a.wav": True, "wake-b.wav": False, "wake-c.wav": True}


def test_load_labels_missing_file(tmp_path):
    assert load_labels(tmp_path) == {}


# -- summarize / report --------------------------------------------------------

def _rows(scores):
    return [{"wav": f"w{i}.wav", "score": s, "rms": 0.1, "time": "t"}
            for i, s in enumerate(scores)]


def test_summary_without_labels_is_overall_distribution():
    s = summarize(_rows([0.3, 0.5, 0.9]))
    assert s["overall"]["n"] == 3 and s["labelled"] is False
    assert s["overall"]["min"] == 0.3 and s["overall"]["max"] == 0.9


def test_separable_clusters_suggest_a_threshold_between_them():
    rows = _rows([0.2, 0.25, 0.95, 0.98])   # noise low, real high
    labels = {"w0.wav": False, "w1.wav": False, "w2.wav": True, "w3.wav": True}
    s = summarize(rows, labels)
    assert s["separable"] is True
    # threshold sits in the gap (0.25 .. 0.95)
    assert 0.25 < s["suggested_threshold"] < 0.95
    assert "none" in s["false_negative_risk"]


def test_overlapping_clusters_report_false_negative_risk():
    rows = _rows([0.6, 0.7, 0.65, 0.9])     # a noise (0.7) above a real (0.65)
    labels = {"w0.wav": False, "w1.wav": False, "w2.wav": True, "w3.wav": True}
    s = summarize(rows, labels)
    assert s["separable"] is False
    assert "missed" in s["false_negative_risk"] or "1/" in s["false_negative_risk"]


def test_report_renders_without_crashing():
    assert "No captures" in format_report([])
    text = format_report(_rows([0.3, 0.8]))
    assert "2 activation" in text
    labelled = format_report(
        _rows([0.2, 0.95]), {"w0.wav": False, "w1.wav": True})
    assert "suggested threshold" in labelled
