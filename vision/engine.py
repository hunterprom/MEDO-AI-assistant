"""The gesture engine: a blocking camera/inference loop with a plain callback.

Runs in the vision sidecar process (its own venv, so MediaPipe's numpy<2 pin never
touches the voice stack). This build is **pointer-only** — discrete command
gestures are disabled:

* **Pointer** (ported from v1 jarvis-web, then expanded): the index fingertip
  drives the OS cursor. The recognized pose selects the action —

    - index only ............ move the cursor
    - thumb+index pinch ..... press/hold = drag; a quick tap = left click
    - two fingers (victory) . scroll (move the hand up/down)
    - index+pinky (rock) .... zoom (Ctrl+wheel; move the hand up/down)
    - three fingers ......... right click
    - thumbs up ............. volume up      (repeats while held)
    - pinky only ............ volume down    (repeats while held)
    - fist HELD ~1 s ........ exit pointer mode (brief fist misreads while
                              pointing must not kick you out — anything
                              unrecognized just keeps moving the cursor)

  Toggled by voice ("pointer on"), the HUD switch, or ``POST :stream_port/pointer``
  — and it always boots OFF.

When pointer mode is OFF the camera simply streams the annotated feed; no gesture
is routed as an utterance. (``on_gesture`` is retained for API compatibility but
no longer invoked.)

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
    UNKNOWN,
    GestureRecognizer,
    index_thumb_pinch,
)
from vision.pointer import (
    DRAG,
    MOVE,
    RIGHT_CLICK,
    SCROLL,
    VOLUME_DOWN,
    VOLUME_UP,
    ZOOM,
    ClickDebouncer,
    Ema,
    PoseHold,
    ScrollAccumulator,
    pointer_action,
    to_screen,
)

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
        lines.append("POINTER ACTIVE - hold fist to exit")
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
        three_hold = PoseHold(getattr(pcfg, "hold_frames", 3))
        # Exiting pointer mode takes a deliberate, HELD fist (~1.2 s at 15 fps):
        # pointing at the camera often momentarily classifies as a fist, and a
        # short confirm was kicking users out of pointer mode mid-move.
        fist_hold = PoseHold(int(getattr(pcfg, "exit_hold_frames", 18)))
        # Continuous-motion actions (scroll/zoom) and repeatable ones (volume).
        scroll_acc = ScrollAccumulator(float(getattr(pcfg, "scroll_gain", 45.0)))
        zoom_acc = ScrollAccumulator(float(getattr(pcfg, "zoom_gain", 25.0)))
        vol_interval = int(getattr(pcfg, "volume_interval_ms", 180))
        vol_up_gate = ClickDebouncer(vol_interval)
        vol_dn_gate = ClickDebouncer(vol_interval)
        prev_iy: float | None = None   # previous fingertip y, for scroll/zoom deltas
        pinch_down = False             # is the left button currently held (drag)?
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

                pointer_now = pointer_ready and self._pointer
                if pointer_now and not was_pointer:
                    ema.reset()              # fresh smoothing on every activation
                    fist_hold.update(False)  # and a fresh exit hold
                was_pointer = pointer_now

                if pointer_now:
                    now = time.monotonic()
                    fist_exit = False
                    if landmarks is not None and len(landmarks) > 8:
                        tip = landmarks[8]  # index fingertip
                        iy = float(tip.y)
                        # Frames are already selfie-mirrored when c.flip is on;
                        # only unmirrored cameras need the horizontal flip.
                        px, py = to_screen(
                            float(tip.x), iy,
                            getattr(pcfg, "sensitivity", 2.5),
                            screen_w, screen_h,
                            mirror_x=not c.flip,
                        )
                        sx, sy = ema.update(px, py)
                        # A pinch always means drag/click; otherwise the pose picks
                        # the action (move / scroll / zoom / right-click / volume).
                        action = pointer_action(gesture, index_thumb_pinch(landmarks))
                        dy = 0.0 if prev_iy is None else (iy - prev_iy)
                        if abs(dy) < 0.004:
                            dy = 0.0  # deadzone: ignore fingertip jitter
                        try:
                            if action == DRAG:
                                if not pinch_down:
                                    winmouse.press_left()
                                    pinch_down = True
                                    last_fired, last_utterance = "pinch", "drag / click"
                                winmouse.move(int(sx), int(sy))
                            else:
                                if pinch_down:
                                    winmouse.release_left()
                                    pinch_down = False
                                if action == MOVE:
                                    winmouse.move(int(sx), int(sy))
                                elif action == SCROLL:
                                    n = scroll_acc.update(dy)
                                    if n:
                                        winmouse.scroll(n)
                                        last_fired, last_utterance = "victory", "scroll"
                                elif action == ZOOM:
                                    n = zoom_acc.update(dy)
                                    if n:
                                        winmouse.zoom(n)
                                        last_fired, last_utterance = "rock", "zoom"
                                elif action == RIGHT_CLICK:
                                    if three_hold.update(True) and click_gate.ready(now):
                                        winmouse.click_right()
                                        last_fired, last_utterance = "three", "right click"
                                elif action == VOLUME_UP:
                                    if vol_up_gate.ready(now):
                                        winmouse.volume_up()
                                        last_fired, last_utterance = "thumbs_up", "volume up"
                                elif action == VOLUME_DOWN:
                                    if vol_dn_gate.ready(now):
                                        winmouse.volume_down()
                                        last_fired, last_utterance = "open_palm", "volume down"
                                # action == IDLE: hold the cursor still.
                            # Re-arm the latches/accumulators the moment their pose ends.
                            if action != RIGHT_CLICK:
                                three_hold.update(False)
                            if action != SCROLL:
                                scroll_acc.reset()
                            if action != ZOOM:
                                zoom_acc.reset()
                            prev_iy = iy
                        except Exception:
                            logger.exception("pointer control failed; disabling")
                            self._pointer = False
                            pinch_down = False
                        # Exit only on a deliberately HELD fist, not a misread.
                        fist_exit = fist_hold.update(gesture == FIST)
                    else:
                        # Hand lost: release any held drag and forget motion history.
                        if pinch_down:
                            try:
                                winmouse.release_left()
                            except Exception:
                                pass
                            pinch_down = False
                        three_hold.update(False)
                        fist_hold.update(False)
                        scroll_acc.reset()
                        zoom_acc.reset()
                        ema.reset()  # don't lerp across the gap
                        prev_iy = None
                    if fist_exit:
                        if pinch_down:
                            try:
                                winmouse.release_left()
                            except Exception:
                                pass
                            pinch_down = False
                        self.set_pointer(False)
                # Command gestures are intentionally disabled: this build is
                # pointer-only. When pointer mode is OFF the camera just streams
                # the annotated feed; no gesture is routed as an utterance.

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
