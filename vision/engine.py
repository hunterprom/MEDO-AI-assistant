"""The gesture engine: a blocking camera/inference loop with a plain callback.

Runs in the vision sidecar process (its own venv, so MediaPipe's numpy<2 pin never
touches the voice stack). Two modes share the camera:

* **Discrete** (default): on a confirmed gesture, call ``on_gesture(gesture,
  utterance)``; the sidecar's callback POSTs that utterance to the companion API,
  so gestures flow through the same Intent Router as voice and text.
* **Pointer** (ported from v1 jarvis-web): the index fingertip drives the OS
  cursor — pinch left-clicks, three-fingers right-clicks, a stabilized fist
  exits. Discrete utterances are suspended while it's on, so cursor poses don't
  fire their mapped commands. Toggled by voice ("pointer on"), the HUD switch,
  or ``POST :stream_port/pointer`` — and it always boots OFF.

The latest annotated frame is exposed as JPEG for the HUD video stream.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from vision.camera import Camera
from vision.gestures import (
    FIST,
    PINCH,
    THREE,
    UNKNOWN,
    GestureRecognizer,
    GestureStabilizer,
)
from vision.pointer import ClickDebouncer, Ema, PoseHold, to_screen

logger = logging.getLogger(__name__)

#: Called with (gesture_name, utterance) whenever a gesture is confirmed.
GestureHandler = Callable[[str, str], None]


def _draw_debug_overlay(
    frame, gesture: str, last_fired: str, utterance: str, pointer_on: bool = False
) -> None:
    """Stamp what vision sees onto the annotated frame (shown in the HUD feed).

    Each line is drawn twice — a thick dark pass under a thin bright pass — so
    the text stays readable over any camera background. Mutates ``frame``.
    """
    import cv2  # lazy: keep module importable without OpenCV

    lines = [
        f"gesture: {gesture}",
        f"last fired: {last_fired}",
        f"utterance: {utterance or '-'}",
    ]
    if pointer_on:
        lines.append("POINTER ACTIVE - fist to exit")
    for i, line in enumerate(lines):
        origin = (8, 24 + 22 * i)
        color = (90, 190, 255) if line.startswith("POINTER") else (80, 255, 190)
        cv2.putText(frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 1, cv2.LINE_AA)


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
        # Pointer-mode flag: the HTTP thread flips it, the camera thread reads
        # it — a plain bool is atomic under the GIL. All motion/click state
        # stays camera-thread-local in _run().
        self._pointer = False

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

    # -- pointer mode ---------------------------------------------------------

    def pointer_on(self) -> bool:
        return self._pointer

    def set_pointer(self, on: bool) -> bool:
        """Toggle pointer mode; returns the resulting state.

        Turning ON is refused when the feature is disabled in config, so a
        stray API call can't grab the mouse on a machine that opted out.
        """
        pcfg = getattr(self._config, "pointer", None)
        if on and not (pcfg is not None and getattr(pcfg, "enabled", False)):
            logger.info("pointer mode requested but disabled in config")
            return False
        self._pointer = bool(on)
        logger.info("pointer mode %s", "ON" if self._pointer else "OFF")
        return self._pointer

    # -- camera loop ----------------------------------------------------------

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

        # Pointer plumbing (camera-thread-local). winmouse is imported lazily so
        # the engine still runs on machines where user32 is unavailable.
        pcfg = getattr(c, "pointer", None)
        pointer_ready = False
        winmouse = None
        screen_w = screen_h = 0
        if pcfg is not None and getattr(pcfg, "enabled", False):
            try:
                from vision import winmouse as _winmouse

                screen_w, screen_h = _winmouse.screen_size()
                winmouse = _winmouse
                pointer_ready = screen_w > 0 and screen_h > 0
            except Exception as exc:
                logger.warning("pointer mode unavailable on this system: %s", exc)
        ema = Ema(getattr(pcfg, "ema_alpha", 0.4)) if pointer_ready else None
        click_gate = ClickDebouncer(getattr(pcfg, "click_debounce_ms", 600))
        pinch_hold = PoseHold(getattr(pcfg, "hold_frames", 3))
        three_hold = PoseHold(getattr(pcfg, "hold_frames", 3))
        was_pointer = False

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

        logger.info(
            "gesture engine running on camera %d (pointer %s)",
            c.camera_index,
            "available" if pointer_ready else "unavailable",
        )
        last_fired = UNKNOWN
        last_utterance = ""
        try:
            while not self._stop.is_set():
                tick = time.monotonic()
                frame = camera.read()
                if frame is None:
                    time.sleep(0.05)
                    continue

                gesture, annotated, landmarks = recognizer.process(frame)
                confirmed = stabilizer.update(gesture)

                pointer_now = pointer_ready and self._pointer
                if pointer_now and not was_pointer:
                    ema.reset()  # fresh smoothing on every activation
                was_pointer = pointer_now

                if pointer_now:
                    if landmarks is not None and len(landmarks) > 8:
                        tip = landmarks[8]  # index fingertip
                        # Frames are already selfie-mirrored when c.flip is on;
                        # only unmirrored cameras need the horizontal flip.
                        px, py = to_screen(
                            float(tip.x), float(tip.y),
                            getattr(pcfg, "sensitivity", 2.5),
                            screen_w, screen_h,
                            mirror_x=not c.flip,
                        )
                        sx, sy = ema.update(px, py)
                        try:
                            winmouse.move(int(sx), int(sy))
                            now = time.monotonic()
                            if pinch_hold.update(gesture == PINCH) and click_gate.ready(now):
                                winmouse.click_left()
                                last_fired, last_utterance = PINCH, "left click"
                            if three_hold.update(gesture == THREE) and click_gate.ready(now):
                                winmouse.click_right()
                                last_fired, last_utterance = THREE, "right click"
                        except Exception:
                            logger.exception("pointer control failed; disabling")
                            self._pointer = False
                    else:
                        pinch_hold.update(False)
                        three_hold.update(False)
                        ema.reset()  # hand lost: don't lerp across the gap
                    if confirmed == FIST:
                        self.set_pointer(False)
                elif confirmed and confirmed != UNKNOWN:
                    utterance = c.gestures.get(confirmed)
                    if utterance:
                        last_fired, last_utterance = confirmed, utterance
                        logger.info("gesture %s -> %r", confirmed, utterance)
                        try:
                            self._on_gesture(confirmed, utterance)
                        except Exception:
                            logger.exception("gesture handler failed")

                # Debug HUD text goes on after firing so "last fired" is current.
                _draw_debug_overlay(annotated, gesture, last_fired, last_utterance, pointer_now)
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
