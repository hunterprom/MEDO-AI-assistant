"""Unit tests for the pure STT helpers (no Whisper model is ever loaded)."""

from __future__ import annotations

import numpy as np

from voice.stt import Transcriber, build_hotwords, filter_transcript


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


# --- filter_transcript: hallucination defense -------------------------------


def test_filter_transcript_keeps_good_segments():
    segments = [
        ("What time is it?", 0.01, -0.20),
        ("Open Chrome.", 0.05, -0.30),
    ]
    assert filter_transcript(segments) == "What time is it? Open Chrome."


def test_filter_transcript_drops_high_no_speech_low_logprob():
    segments = [
        ("What time is it?", 0.01, -0.20),
        ("Thanks for watching!", 0.90, -1.20),  # classic silence hallucination
    ]
    assert filter_transcript(segments) == "What time is it?"


def test_filter_transcript_requires_both_signals_to_drop():
    # Either score alone is common on genuine short utterances — keep those.
    segments = [
        ("Yes.", 0.90, -0.20),   # high no_speech but confident decode
        ("Stop.", 0.10, -1.50),  # weak decode but clearly speech
    ]
    assert filter_transcript(segments) == "Yes. Stop."


def test_filter_transcript_junk_only_returns_empty():
    assert filter_transcript([("Thank you.", 0.30, -0.40)]) == ""
    assert filter_transcript([("Thanks for watching!", 0.30, -0.40)]) == ""
    assert filter_transcript([("you", 0.30, -0.40)]) == ""
    assert filter_transcript([("Bye.", 0.30, -0.40)]) == ""
    assert filter_transcript([(".", 0.30, -0.40)]) == ""


def test_filter_transcript_junk_check_is_whole_transcript_only():
    # "thank you" inside a real sentence must not nuke the transcript.
    segments = [("Thank you, now open Chrome.", 0.05, -0.30)]
    assert filter_transcript(segments) == "Thank you, now open Chrome."


def test_filter_transcript_empty_input():
    assert filter_transcript([]) == ""


def test_filter_transcript_all_segments_dropped():
    segments = [("...", 0.99, -2.0), ("Thank you.", 0.80, -1.5)]
    assert filter_transcript(segments) == ""


# --- RMS-target normalization ------------------------------------------------


def test_normalize_boosts_quiet_audio_toward_target():
    quiet = (0.005 * np.sin(np.linspace(0, 200 * np.pi, 16000))).astype(np.float32)
    out = Transcriber._normalize(quiet)
    assert out.dtype == np.float32
    assert _rms(out) > _rms(quiet)
    assert _rms(out) <= 0.06 + 1e-6  # never overshoots the target
    assert float(np.max(np.abs(out))) <= 1.0


def test_normalize_gain_is_capped():
    very_quiet = (0.001 * np.sin(np.linspace(0, 200 * np.pi, 16000))).astype(np.float32)
    out = Transcriber._normalize(very_quiet)
    # target/rms would be ~85x; the cap holds it to at most 10x.
    assert _rms(out) <= 10.0 * _rms(very_quiet) + 1e-9


def test_normalize_leaves_loud_audio_alone():
    loud = (0.5 * np.sin(np.linspace(0, 200 * np.pi, 16000))).astype(np.float32)
    out = Transcriber._normalize(loud)
    np.testing.assert_array_equal(out, loud)


def test_normalize_leaves_silence_untouched():
    silence = np.zeros(16000, dtype=np.float32)
    out = Transcriber._normalize(silence)
    np.testing.assert_array_equal(out, silence)
    empty = np.zeros(0, dtype=np.float32)
    assert Transcriber._normalize(empty).size == 0


def test_normalize_never_clips():
    # Quiet on average but with one full-scale spike: gain must be limited so
    # the spike stays inside [-1, 1] instead of clipping.
    spiky = np.full(16000, 0.005, dtype=np.float32)
    spiky[8000] = 0.9
    out = Transcriber._normalize(spiky)
    assert float(np.max(np.abs(out))) <= 1.0


# --- hotwords extraction ------------------------------------------------------


def test_build_hotwords_extracts_command_phrases():
    prompt = (
        "Commands for a voice assistant: what time is it, open chrome, volume up."
    )
    assert build_hotwords(prompt) == "what time is it, open chrome, volume up"


def test_build_hotwords_without_colon_uses_whole_prompt():
    assert build_hotwords("open chrome, volume up") == "open chrome, volume up"


# --- bilingual auto-detect clamp (pick_forced_language) ----------------------


def test_clamp_keeps_allowed_detections():
    from voice.stt import pick_forced_language

    assert pick_forced_language("en", ["en", "mk"]) is None
    assert pick_forced_language("mk", ["en", "mk"]) is None


def test_clamp_forces_first_non_english_on_misdetection():
    # Macedonian heard as Bulgarian/Serbian/Russian -> re-decode as mk.
    from voice.stt import pick_forced_language

    for wrong in ("bg", "sr", "sl", "ru", "hr"):
        assert pick_forced_language(wrong, ["en", "mk"]) == "mk"


def test_clamp_disabled_with_empty_allowed_list():
    from voice.stt import pick_forced_language

    assert pick_forced_language("bg", []) is None


def test_clamp_all_english_falls_back_to_first():
    from voice.stt import pick_forced_language

    assert pick_forced_language("de", ["en"]) == "en"


def test_wake_display_phrase():
    from voice.wakeword import display_phrase
    assert display_phrase("models/wakeword/hey_medo.onnx") == "hey medo"
    assert display_phrase("hey_jarvis") == "hey jarvis"
    assert display_phrase("/abs/path/custom_word.tflite") == "custom word"
