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
        import sys

        import cv2

        # On macOS, force the AVFoundation backend — the default can silently
        # pick a dead legacy path on some OpenCV builds.
        if sys.platform == "darwin":
            self._cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        else:
            self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise RuntimeError(self._open_failure_hint())
        return self

    def _open_failure_hint(self) -> str:
        """A message that names the usual cause instead of just the index.

        The #1 macOS cause is a denied Camera permission (TCC): OpenCV reports
        "camera access has been denied" and VideoCapture never opens. macOS
        often does NOT prompt for it, so the fix is manual — say so.
        """
        import sys

        base = f"could not open camera index {self.index}"
        if sys.platform == "darwin":
            return (
                f"{base} — on macOS this is almost always the Camera "
                f"permission: grant it to the app that launches the sidecar "
                f"(Terminal / run.command) in System Settings → Privacy & "
                f"Security → Camera. If it isn't listed, run "
                f"'tccutil reset Camera' in that terminal and relaunch. Also "
                f"check no other app is using the webcam, or try "
                f"vision.camera_index: 1."
            )
        return (
            f"{base} — check the camera isn't in use by another app, that the "
            f"index is right (try vision.camera_index: 1), and on Linux that "
            f"the sidecar user can read /dev/video*."
        )

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
