"""The gesture engine: a blocking camera/inference loop with a plain callback.

Runs in the vision sidecar process (its own venv, so MediaPipe's numpy<2 pin never
touches the voice stack). On a confirmed gesture it calls ``on_gesture(gesture,
utterance)``; the sidecar's callback POSTs that utterance to the companion API, so
gestures flow through the same Intent Router as voice and text. The latest
annotated frame is exposed as JPEG for the HUD video stream.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from vision.camera import Camera
from vision.gestures import UNKNOWN, GestureRecognizer, GestureStabilizer

logger = logging.getLogger(__name__)

#: Called with (gesture_name, utterance) whenever a gesture is confirmed.
GestureHandler = Callable[[str, str], None]


def _draw_debug_overlay(frame, gesture: str, last_fired: str, utterance: str) -> None:
    """Stamp what vision sees onto the annotated frame (shown in the HUD feed).

    Each line is drawn twice — a thick dark pass under a thin bright pass — so
    the text stays readable over any camera background. Mutates ``frame``.
    """
    import cv2  # lazy: keep module importable without OpenCV

    lines = (
        f"gesture: {gesture}",
        f"last fired: {last_fired}",
        f"utterance: {utterance or '-'}",
    )
    for i, line in enumerate(lines):
        origin = (8, 24 + 22 * i)
        cv2.putText(frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (80, 255, 190), 1, cv2.LINE_AA)


class GestureEngine:
    """Owns the camera thread and turns gestures into handler calls."""

    def __init__(self, config, on_gesture: GestureHandler) -> None:
        self._config = config
        self._on_gesture = on_gesture
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._jpeg: bytes | None = None
        self._jpeg_lock = threading.Lock()
        self.error: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="gesture-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def latest_jpeg(self) -> bytes | None:
        with self._jpeg_lock:
            return self._jpeg

    def _run(self) -> None:
        try:
            import cv2
        except Exception as exc:  # pragma: no cover
            self.error = f"OpenCV unavailable: {exc}"
            logger.error(self.error)
            return

        c = self._config
        cooldown_frames = int(c.cooldown_s * max(1, c.max_fps))
        stabilizer = GestureStabilizer(c.stability_frames, cooldown_frames)
        min_period = 1.0 / max(1, c.max_fps)

        try:
            recognizer = GestureRecognizer(c.min_detection_confidence, c.min_tracking_confidence)
        except Exception as exc:
            self.error = f"MediaPipe unavailable: {exc}"
            logger.error(self.error)
            return
        try:
            camera = Camera(c.camera_index, c.flip).open()
        except Exception as exc:
            self.error = str(exc)
            logger.error("camera error: %s", exc)
            recognizer.close()
            return

        logger.info("gesture engine running on camera %d", c.camera_index)
        last_fired = UNKNOWN
        last_utterance = ""
        try:
            while not self._stop.is_set():
                tick = time.monotonic()
                frame = camera.read()
                if frame is None:
                    time.sleep(0.05)
                    continue

                gesture, annotated = recognizer.process(frame)

                confirmed = stabilizer.update(gesture)
                if confirmed and confirmed != UNKNOWN:
                    utterance = c.gestures.get(confirmed)
                    if utterance:
                        last_fired, last_utterance = confirmed, utterance
                        logger.info("gesture %s -> %r", confirmed, utterance)
                        try:
                            self._on_gesture(confirmed, utterance)
                        except Exception:
                            logger.exception("gesture handler failed")

                # Debug HUD text goes on after firing so "last fired" is current.
                _draw_debug_overlay(annotated, gesture, last_fired, last_utterance)
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    with self._jpeg_lock:
                        self._jpeg = buf.tobytes()

                elapsed = time.monotonic() - tick
                if elapsed < min_period:
                    time.sleep(min_period - elapsed)
        finally:
            camera.close()
            recognizer.close()
