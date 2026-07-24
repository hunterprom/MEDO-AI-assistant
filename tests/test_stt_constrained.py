"""Constrained two-language mode — S2: detection restricted to the active pair.

The point of the feature is that a THIRD language is never chosen. These pin
both constraint paths (native-restrict via detect_language, and the snap
fallback) with an injected fake model, plus the pure ``choose_language`` core.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.config import LanguagesConfig, STTConfig
from voice.stt import choose_language


# --- pure core ---------------------------------------------------------------

def test_choose_language_picks_highest_active():
    # bg scores highest overall but isn't active — mk (0.3) beats en (0.1).
    probs = [("bg", 0.6), ("mk", 0.3), ("en", 0.1)]
    assert choose_language(probs, ["en", "mk"], "en") == "mk"


def test_choose_language_english_wins_when_higher():
    assert choose_language([("en", 0.7), ("mk", 0.2)], ["en", "mk"], "en") == "en"


def test_choose_language_none_active_returns_primary():
    # only non-active languages present -> fall back to primary, never a third.
    assert choose_language([("bg", 0.9), ("sr", 0.1)], ["en", "mk"], "en") == "en"


def test_choose_language_is_case_insensitive():
    assert choose_language([("MK", 0.8), ("EN", 0.1)], ["en", "mk"], "en") == "mk"


# --- Transcriber decision tree (fake model) ----------------------------------

class _Seg:
    def __init__(self, text: str) -> None:
        self.text, self.no_speech_prob, self.avg_logprob = text, 0.1, -0.2


def _transcriber(monkeypatch, active, *, primary=None, detection="auto_pair",
                 detected="en", all_probs=None, no_detect=False):
    import faster_whisper

    import voice.stt as stt

    probs = all_probs if all_probs is not None else [("en", 0.9)]

    class FakeModel:
        def __init__(self, *a, **k):
            self.last_language = "__unset__"
            self.decode_count = 0

        def transcribe(self, audio, language=None, **k):
            self.last_language = language
            self.decode_count += 1
            return iter([_Seg("some words")]), SimpleNamespace(language=detected)

        def detect_language(self, audio=None, **k):
            if no_detect:
                raise AttributeError("this build has no detect_language")
            return (probs[0][0], probs[0][1], probs)

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    langs = LanguagesConfig(active=active, primary=primary, detection=detection)
    cfg = STTConfig(device="cpu", compute_type="int8", filter_hallucinations=False)
    return stt.Transcriber(cfg, langs)


def _run(t):
    return t.transcribe_with_language(np.ones(16000, dtype=np.float32) * 0.1)


def test_fixed_mode_always_uses_primary(monkeypatch):
    # detection would say mk, but fixed pins primary=en and never detects.
    t = _transcriber(monkeypatch, ["en", "mk"], primary="en", detection="fixed",
                     all_probs=[("mk", 0.9), ("en", 0.1)])
    _, lang = _run(t)
    assert lang == "en"
    assert t._model.last_language == "en"


def test_single_active_behaves_as_fixed(monkeypatch):
    t = _transcriber(monkeypatch, ["mk"], detection="auto_pair",
                     all_probs=[("en", 0.9)])
    _, lang = _run(t)
    assert lang == "mk"
    assert t._model.last_language == "mk"


def test_auto_pair_forces_the_active_winner_not_a_third(monkeypatch):
    # bg is the global argmax; the decode must be forced to the best ACTIVE (mk).
    t = _transcriber(monkeypatch, ["en", "mk"], detection="auto_pair",
                     all_probs=[("bg", 0.6), ("mk", 0.3), ("en", 0.1)])
    _, lang = _run(t)
    assert lang == "mk"
    assert t._model.last_language == "mk"


def test_auto_pair_english(monkeypatch):
    t = _transcriber(monkeypatch, ["en", "mk"], detection="auto_pair",
                     all_probs=[("en", 0.8), ("mk", 0.2)])
    _, lang = _run(t)
    assert lang == "en" and t._model.last_language == "en"


def test_snap_fallback_snaps_out_of_pair_to_primary(monkeypatch):
    # No detect_language: auto-detect says bg (not active) -> snap to primary en.
    t = _transcriber(monkeypatch, ["en", "mk"], primary="en", detection="auto_pair",
                     detected="bg", no_detect=True)
    _, lang = _run(t)
    assert lang == "en"
    assert t._model.last_language == "en"    # re-decoded forced to primary
    assert t._model.decode_count == 2        # auto-detect pass + forced pass


def test_snap_fallback_keeps_in_pair_detection(monkeypatch):
    # No detect_language, but auto-detect landed on an ACTIVE language -> keep it.
    t = _transcriber(monkeypatch, ["en", "mk"], primary="en", detection="auto_pair",
                     detected="mk", no_detect=True)
    _, lang = _run(t)
    assert lang == "mk"
    assert t._model.decode_count == 1        # no second pass needed


def test_legacy_path_when_no_language_config(monkeypatch):
    # Passing only stt (no LanguagesConfig) keeps the pre-S2 behavior alive.
    import faster_whisper

    import voice.stt as stt

    class FakeModel:
        def __init__(self, *a, **k):
            self.last_language = "__unset__"

        def transcribe(self, audio, language=None, **k):
            self.last_language = language
            return iter([_Seg("x")]), SimpleNamespace(language="en")

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    cfg = STTConfig(device="cpu", compute_type="int8", filter_hallucinations=False,
                    allowed_languages=["en", "mk"])
    t = stt.Transcriber(cfg)          # no languages arg -> legacy
    assert t._langs is None
    _, lang = _run(t)
    assert lang == "en"               # en detected + in allowed -> kept


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
