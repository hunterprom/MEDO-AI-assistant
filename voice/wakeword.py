"""Wake-word detection with openWakeWord ("Hey MEDO").

Runs continuously on the mic stream while the assistant is IDLE. Each 80 ms frame
is scored; when the score for the configured phrase crosses the threshold we hand
off to STT. onnxruntime is the inference backend (no tflite on this platform).
"""

from __future__ import annotations

import glob
import logging
import os

import numpy as np

from core.config import WakeWordConfig

logger = logging.getLogger(__name__)


def _resolve_model_path(phrase: str) -> str:
    """Map a phrase like ``"hey_jarvis"`` to its bundled ``.onnx`` file.

    Falls back to returning the phrase unchanged so a user can point the config
    at a custom model path.
    """
    import openwakeword

    models_dir = os.path.join(
        os.path.dirname(openwakeword.__file__), "resources", "models"
    )
    matches = sorted(glob.glob(os.path.join(models_dir, f"{phrase}*.onnx")))
    return matches[0] if matches else phrase


class WakeWord:
    """Thin wrapper over ``openwakeword.model.Model`` for one wake phrase."""

    def __init__(self, config: WakeWordConfig) -> None:
        from openwakeword.model import Model

        self.threshold = config.threshold
        model_path = _resolve_model_path(config.phrase)
        self._model = Model(
            wakeword_models=[model_path],
            inference_framework="onnx",
        )
        # openWakeWord keys predictions by model name (e.g. "hey_jarvis_v0.1").
        keys = list(self._model.models.keys())
        root = config.phrase.split("_")[0]
        self._key = next((k for k in keys if config.phrase in k or root in k), keys[0])
        logger.info("wake word ready: %r (threshold %.2f)", self._key, self.threshold)

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
    import time

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
    dev_note = "system default" if s.audio.input_device is None else f"device #{s.audio.input_device}"
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


if __name__ == "__main__":
    _live_test()
