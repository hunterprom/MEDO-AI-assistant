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
