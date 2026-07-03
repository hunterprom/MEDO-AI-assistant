"""Camera capture over OpenCV, kept behind a small interface.

Blocking (like PortAudio), so the gesture engine drives it from a worker thread.
``cv2`` is imported lazily so importing this module never requires OpenCV.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """A webcam you read BGR frames from; use as a context manager."""

    def __init__(self, index: int = 0, flip: bool = True) -> None:
        self.index = index
        self.flip = flip
        self._cap = None  # cv2.VideoCapture

    def open(self) -> "Camera":
        import cv2

        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open camera index {self.index}")
        return self

    def read(self) -> np.ndarray | None:
        """Return the next BGR frame (mirrored if configured), or None on failure."""
        if self._cap is None:
            raise RuntimeError("camera not open")
        ok, frame = self._cap.read()
        if not ok:
            return None
        if self.flip:
            import cv2

            frame = cv2.flip(frame, 1)  # selfie mirror
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()
