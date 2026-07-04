"""Language routing for TTS: Cyrillic detection + ffmpeg discovery (pure units)."""

from __future__ import annotations

from voice.tts import contains_cyrillic, find_ffmpeg


def test_cyrillic_detection():
    assert contains_cyrillic("Скопје е главниот град на Македонија.")
    assert contains_cyrillic("mixed: часот е 3 pm")   # any Cyrillic wins
    assert not contains_cyrillic("It's 3 PM in Skopje.")
    assert not contains_cyrillic("")
    assert not contains_cyrillic(None)


def test_find_ffmpeg_returns_path_or_none():
    p = find_ffmpeg()
    assert p is None or p.lower().endswith("ffmpeg.exe") or "ffmpeg" in p.lower()
