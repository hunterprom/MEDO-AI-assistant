"""Wake-word detection with openWakeWord ("Hey MEDO").

Runs continuously on the mic stream while the assistant is IDLE. Each 80 ms frame
is scored; when the score for the configured phrase crosses the threshold we hand
off to STT.

``wakeword.phrase`` accepts either a bundled openWakeWord model name
("hey_jarvis") or a path to your own trained model — ``.onnx`` (onnxruntime)
or ``.tflite`` (tflite-runtime), picked by extension. Training runbook:
``docs/Wake Word Training.md``.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import numpy as np

from core.config import PROJECT_ROOT, WakeWordConfig

logger = logging.getLogger(__name__)


def display_phrase(phrase: str) -> str:
    """Human-friendly wake phrase for banners/HUD.

    A model PATH ("models/wakeword/hey_medo.onnx") becomes its spoken form
    ("hey medo"); a bundled name ("hey_jarvis") just has underscores spaced.
    """
    stem = phrase
    if stem.lower().endswith((".onnx", ".tflite")):
        stem = Path(stem).stem
    return stem.replace("_", " ").strip()


#: The bundled model MEDO ships listening for until a custom "hey MEDO" model
#: is trained (voice/train_wakeword.py). Also the graceful fallback when a
#: configured custom model file is missing, so voice never dies over a bad path.
FALLBACK_PHRASE = "hey_jarvis"

#: Wake-phrase filler words that don't distinguish the phrase from noise, so a
#: transcript needn't contain them to confirm (a person says "medo", not always
#: the "hey").
_WAKE_FILLER = {"hey", "hi", "ok", "okay", "yo", "hello", "a", "the", "hej", "еј"}


def wake_phrase_confirmed(transcript: str, phrase: str) -> bool:
    """Does an STT read of the wake buffer plausibly contain the wake phrase?

    The second stage of loud-noise rejection (``wakeword.stt_confirm``): a door
    slam, music or a clap transcribes to nothing, so an empty/whitespace
    transcript is rejected outright — that alone kills most false wakes.
    Unrelated speech that lacks the phrase is rejected too. A real "hey medo"
    passes even when Whisper spells it oddly ("hey meadow", "medoh"), because
    the match is lenient — the point is never to drop a genuine wake.
    """
    import difflib
    import re as _re

    from core import mk

    # Romanize so a bilingual user's "медо" matches the Latin "medo" (Whisper
    # may transcribe the same wake word in either script).
    text = mk.to_latin((transcript or "").strip().lower())
    if not text:
        return False                               # non-speech -> no words
    distinct = [w for w in mk.to_latin(display_phrase(phrase).lower()).split()
                if w not in _WAKE_FILLER]
    if not distinct:
        return True                                # phrase is only filler
    tokens = _re.findall(r"[\w']+", text)
    for want in distinct:
        if want in text:                           # exact / substring
            return True
        floor = max(3, len(want) - 1)              # don't fuzzy-match tiny words
        for tok in tokens:
            if len(tok) >= floor and \
                    difflib.SequenceMatcher(None, tok, want).ratio() >= 0.7:
                return True
    return False


def _bundled_match(name: str) -> str | None:
    """First bundled openWakeWord ``.onnx`` whose filename starts with ``name``."""
    import openwakeword

    models_dir = os.path.join(
        os.path.dirname(openwakeword.__file__), "resources", "models"
    )
    matches = sorted(glob.glob(os.path.join(models_dir, f"{name}*.onnx")))
    return matches[0] if matches else None


def _resolve_model_path(phrase: str) -> str:
    """Map ``wakeword.phrase`` to a model file.

    A ``.onnx``/``.tflite`` value is a custom model path — absolute, ``~``, or
    relative to the project root (e.g. ``models/wakeword/hey_medo.onnx``). When
    that file is missing (e.g. "hey MEDO" isn't trained yet) MEDO does NOT die:
    it falls back to the bundled ``hey_jarvis`` so voice keeps working, and the
    day the trained file appears it is used automatically. A plain name is
    looked up among the bundled models.
    """
    if phrase.endswith((".onnx", ".tflite")):
        candidate = Path(os.path.expanduser(phrase))
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            return str(candidate)
        logger.warning(
            "custom wake model %s not found — falling back to %r "
            "(train it with voice/train_wakeword.py)", phrase, FALLBACK_PHRASE)
        return _bundled_match(FALLBACK_PHRASE) or FALLBACK_PHRASE

    return _bundled_match(phrase) or phrase


def effective_phrase(phrase: str) -> str:
    """The wake phrase the LOADED model actually listens for.

    Mirrors :func:`_resolve_model_path`'s fallback decision but returns a
    spoken NAME (not a file path): a custom-model path whose file is missing
    falls back to the bundled phrase, and STT confirmation must key off THIS —
    otherwise ``stt_confirm`` checks the transcript against "medo" while the
    fallback model is listening for "hey jarvis", and the fallback (whose whole
    job is "voice keeps working") could never wake by voice.
    """
    if phrase.endswith((".onnx", ".tflite")):
        candidate = Path(os.path.expanduser(phrase))
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return phrase if candidate.exists() else FALLBACK_PHRASE
    return phrase


class WakeWord:
    """Thin wrapper over ``openwakeword.model.Model`` for one wake phrase."""

    def __init__(self, config: WakeWordConfig) -> None:
        from openwakeword.model import Model

        self.threshold = config.threshold
        # How many consecutive over-threshold frames make a wake. The consumer
        # (voice loop) counts them; exposed here so both the standby wait and
        # the barge-in watcher agree on it.
        self.trigger_frames = max(1, int(getattr(config, "trigger_frames", 1)))
        # The phrase STT-confirm checks against: the one the loaded model really
        # listens for (may be the bundled fallback if a custom file is missing).
        self.phrase = effective_phrase(config.phrase)
        model_path = _resolve_model_path(config.phrase)
        # Framework follows the model file: .tflite needs tflite-runtime,
        # everything else (bundled + custom .onnx) runs on onnxruntime.
        framework = "tflite" if model_path.endswith(".tflite") else "onnx"
        kwargs: dict = {"wakeword_models": [model_path],
                        "inference_framework": framework}
        vad = float(getattr(config, "vad_threshold", 0.0) or 0.0)
        if vad > 0.0:
            kwargs["vad_threshold"] = vad
        try:
            self._model = Model(**kwargs)
        except Exception:
            # The speech-gate (VAD) model may be missing/undownloadable offline;
            # never let that kill voice — run without it.
            if "vad_threshold" in kwargs:
                logger.warning("wake-word VAD unavailable — running without it")
                kwargs.pop("vad_threshold")
                self._model = Model(**kwargs)
            else:
                raise
        # openWakeWord keys predictions by model name (e.g. "hey_jarvis_v0.1").
        keys = list(self._model.models.keys())
        root = config.phrase.split("_")[0]
        self._key = next((k for k in keys if config.phrase in k or root in k), keys[0])
        logger.info("wake word ready: %r (threshold %.2f, %d frame%s)",
                    self._key, self.threshold, self.trigger_frames,
                    "" if self.trigger_frames == 1 else "s")

    def predict(self, frame: np.ndarray) -> float:
        """Score one int16 frame; returns the confidence for the wake phrase."""
        scores = self._model.predict(frame)
        return float(scores.get(self._key, 0.0))

    def triggered(self, frame: np.ndarray) -> bool:
        """True when this frame pushes the wake score past the threshold."""
        return self.predict(frame) >= self.threshold

    def reset(self) -> None:
        """Clear the model's internal audio buffer between activations."""
        self._model.reset()


def _live_test() -> None:
    """Live wake-word tester: ``python -m voice.wakeword``.

    Opens the mic and prints the level + wake score every frame, so you can watch
    what happens while you say the phrase — the fastest way to tell a silent mic
    (``mic`` stays ~0) from a too-high threshold (``score`` peaks below it).
    """

    from core.config import load_settings
    from voice.audio import FRAME_SAMPLES, Microphone, frame_rms

    logging.basicConfig(level=logging.INFO)
    s = load_settings()

    # Show the input devices so a wrong/low default mic is easy to spot and fix.
    try:
        import sounddevice as sd

        print("Input devices (set audio.input_device in config.yaml to a number):")
        default_in = sd.default.device[0]
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                # Windows consoles are often cp1252 — strip names to ASCII so a
                # fancy device name can't crash the tester with UnicodeEncodeError.
                name = str(dev["name"]).encode("ascii", "replace").decode("ascii")
                mark = "  <- current default" if i == default_in else ""
                print(f"  [{i}] {name}{mark}")
        print()
    except Exception as exc:
        print(f"(could not list devices: {exc})\n")

    wake = WakeWord(s.wakeword)
    phrase = s.wakeword.phrase.replace("_", " ")
    dev = s.audio.input_device
    if dev is None:
        dev_note = "system default"
    elif isinstance(dev, list):  # priority list: first available wins
        dev_note = "first available of " + " > ".join(str(d) for d in dev)
    else:
        dev_note = f"device {dev!r}"
    print(f'Using {dev_note}. Say "{phrase}" (threshold {wake.threshold:.2f}). Ctrl-C to stop.\n')
    peak = 0.0
    with Microphone(s.audio.sample_rate, FRAME_SAMPLES, s.audio.input_device) as mic:
        try:
            while True:
                frame = mic.read_frame()
                score = wake.predict(frame)
                rms = frame_rms(frame)
                peak = max(peak, score)
                bar = "#" * int(score * 40)
                hit = "  <== WAKE" if score >= wake.threshold else ""
                print(f"mic {rms:5.3f} | score {score:4.2f} |{bar:<40}|{hit}", end="\r", flush=True)
                if score >= wake.threshold:
                    print(f"\nTRIGGERED at {score:.2f}\n")
                    wake.reset()
        except KeyboardInterrupt:
            print(f"\n\nPeak score seen: {peak:.2f} (threshold {wake.threshold:.2f}). "
                  f"{'Lower wakeword.threshold in config.yaml.' if 0 < peak < wake.threshold else ''}")


def _report() -> None:
    """``--report``: summarize captured activations (see voice.wake_debug)."""
    from core.config import PROJECT_ROOT
    from voice.wake_debug import WakeCaptureLog, format_report, load_labels

    directory = PROJECT_ROOT / "logs" / "wake_captures"
    log = WakeCaptureLog(directory, enabled=True)
    print(format_report(log.rows(), load_labels(directory)))


def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m voice.wakeword",
        description="Wake-word tools: live level monitor and capture report.")
    parser.add_argument("--monitor", action="store_true",
                        help="live score/RMS/threshold monitor (the default)")
    parser.add_argument("--report", action="store_true",
                        help="summarize logs/wake_captures/ (needs debug_capture)")
    args = parser.parse_args(argv)
    if args.report:
        _report()
    else:
        _live_test()          # default: the monitor, as before
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
