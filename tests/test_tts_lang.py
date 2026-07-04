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


def test_drain_sentences_streaming():
    from voice.tts import drain_sentences

    # Nothing until a sentence completes.
    sents, rest = drain_sentences("The weather in Skopje is")
    assert sents == [] and rest.startswith("The weather")
    # A completed sentence is emitted; the tail stays buffered.
    sents, rest = drain_sentences("The weather in Skopje is sunny today. Tomorrow will")
    assert sents == ["The weather in Skopje is sunny today."]
    assert rest == "Tomorrow will"
    # Short fragments ("Dr.") merge with the next sentence instead of being spoken alone.
    sents, rest = drain_sentences("Dr. Petrov is your dentist, sir. And that")
    assert sents == ["Dr. Petrov is your dentist, sir."]
    assert rest == "And that"
    # Multiple sentences drain in order.
    sents, rest = drain_sentences("First complete sentence here. Second complete sentence there! Third")
    assert sents == ["First complete sentence here.", "Second complete sentence there!"]
    assert rest == "Third"
