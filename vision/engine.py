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

  With a SECOND hand in frame the single-hand poses pause (cursor keeps
  following the primary hand, pinch-drag still works) and the pair takes over:

    - hands apart/together .. zoom out/in (Ctrl+wheel)
    - second hand thumbs up . play / pause (media key)

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
import math

from vision.gestures import (
    FIST,
    THUMBS_UP,
    UNKNOWN,
    GestureRecognizer,
    index_thumb_pinch,
)
from vision.snap import UiaProvider, draw_highlight, resolve_click
from vision.pointer import (
    DRAG,
    MOVE,
    NEXT_TAB,
    RIGHT_CLICK,
    SCROLL,
    SWITCH_WINDOW,
    TASKBAR,
    VOLUME_DOWN,
    VOLUME_UP,
    ZOOM,
    ClickDebouncer,
    Ema,
    PoseHold,
    ScrollAccumulator,
    build_pose_actions,
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

        # Pointer plumbing (camera-thread-local). The mouse backend is imported
        # lazily so the engine still runs on machines where no backend works
        # (vision/mouse.py picks win32/mac/other per platform).
        pcfg = getattr(c, "pointer", None)
        pointer_ready = False
        mouse = None
        screen_x0 = screen_y0 = 0
        screen_w = screen_h = 0
        if pcfg is not None and getattr(pcfg, "enabled", False):
            try:
                from vision import mouse as _mouse

                # Full virtual desktop (all monitors) so the cursor can cross
                # onto a second/third screen; fall back to the primary size.
                if hasattr(_mouse, "screen_bounds"):
                    screen_x0, screen_y0, screen_w, screen_h = _mouse.screen_bounds()
                else:
                    screen_w, screen_h = _mouse.screen_size()
                mouse = _mouse
                pointer_ready = screen_w > 0 and screen_h > 0
                logger.info("pointer canvas: %dx%d at (%d,%d) [all monitors]",
                            screen_w, screen_h, screen_x0, screen_y0)
            except Exception as exc:
                logger.warning("pointer mode unavailable on this system: %s", exc)
        ema = Ema(getattr(pcfg, "ema_alpha", 0.4)) if pointer_ready else None
        click_gate = ClickDebouncer(getattr(pcfg, "click_debounce_ms", 600))
        three_hold = PoseHold(getattr(pcfg, "hold_frames", 3))
        # Navigation poses get their own hold + debounce: they move you between
        # windows, so they must be held deliberately and can't machine-gun.
        nav_hold = PoseHold(getattr(pcfg, "nav_hold_frames", 5))
        nav_gate = ClickDebouncer(getattr(pcfg, "nav_debounce_ms", 900))
        # Exiting pointer mode takes a deliberate, HELD fist (~1.2 s at 15 fps):
        # pointing at the camera often momentarily classifies as a fist, and a
        # short confirm was kicking users out of pointer mode mid-move.
        fist_hold = PoseHold(int(getattr(pcfg, "exit_hold_frames", 18)))
        # Second hand: hands apart/together = zoom, secondary thumbs-up =
        # play/pause. Single-hand action poses are suspended while both hands
        # are up so the pair can't fire scroll/volume by accident.
        two_spread_prev: float | None = None
        pp_hold = PoseHold(getattr(pcfg, "hold_frames", 3))
        pp_gate = ClickDebouncer(800)
        prev_tip: tuple[float, float] | None = None  # keeps the cursor on the same hand
        # Continuous-motion actions (scroll/zoom) and repeatable ones (volume).
        scroll_acc = ScrollAccumulator(float(getattr(pcfg, "scroll_gain", 45.0)))
        zoom_acc = ScrollAccumulator(float(getattr(pcfg, "zoom_gain", 25.0)))
        pose_actions = build_pose_actions(getattr(pcfg, "pose_actions", None))
        vol_interval = int(getattr(pcfg, "volume_interval_ms", 180))
        vol_up_gate = ClickDebouncer(vol_interval)
        vol_dn_gate = ClickDebouncer(vol_interval)
        prev_iy: float | None = None   # previous fingertip y, for scroll/zoom deltas
        pinch_down = False             # is the left button currently held (drag)?
        # Accessibility provider for click snapping. Absent (or unusable) means
        # every click stays exactly where the cursor is, as before.
        snap_provider = UiaProvider() if getattr(pcfg, "snap_enabled", True) else None
        if snap_provider is not None and not snap_provider.available:
            logger.info("click snapping on, but no UI Automation — clicks stay raw")
            snap_provider = None
        was_pointer = False

        try:
            recognizer = GestureRecognizer(
            c.min_detection_confidence,
            c.min_tracking_confidence,
            # Without these the strictness knobs in config.yaml are dead
            # letters: the recognizer would silently use its defaults.
            strict=c.pointer.strict_gestures,
            extended_min_deg=c.pointer.extended_min_deg,
            curled_max_deg=c.pointer.curled_max_deg,
            zoom_min_spread_deg=c.pointer.zoom_min_spread_deg,
            l_shape_tolerance_deg=c.pointer.l_shape_tolerance_deg,
        )
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

                gesture, annotated, landmarks, hands = recognizer.process(frame)
                # Keep the cursor on the hand the user was already pointing
                # with: when a second hand enters the frame, MediaPipe's order
                # is arbitrary, so pick primary by proximity to the last tip.
                if len(hands) == 2 and prev_tip is not None:
                    d0 = ((hands[0][1][8].x - prev_tip[0]) ** 2
                          + (hands[0][1][8].y - prev_tip[1]) ** 2)
                    d1 = ((hands[1][1][8].x - prev_tip[0]) ** 2
                          + (hands[1][1][8].y - prev_tip[1]) ** 2)
                    if d1 < d0:
                        hands = [hands[1], hands[0]]
                    gesture, landmarks = hands[0]

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
                            x0=screen_x0, y0=screen_y0,
                        )
                        sx, sy = ema.update(px, py)
                        # A pinch always means drag/click; otherwise the pose picks
                        # the action (move / scroll / zoom / right-click / volume).
                        action = pointer_action(gesture, index_thumb_pinch(landmarks),
                                                pose_actions)
                        second = hands[1] if len(hands) == 2 else None
                        if second is not None and action != DRAG:
                            # Pair mode: the primary hand only moves the cursor;
                            # the PAIR zooms (spread) and the second hand's
                            # thumbs-up toggles play/pause. Suspending the
                            # single-hand poses stops accidental scroll/volume.
                            action = MOVE
                        dy = 0.0 if prev_iy is None else (iy - prev_iy)
                        if abs(dy) < 0.004:
                            dy = 0.0  # deadzone: ignore fingertip jitter
                        try:
                            if action == DRAG:
                                if not pinch_down:
                                    # Snap BEFORE pressing: the pinch drags the
                                    # fingertip down, so the raw point sits low by
                                    # the time the button goes down.
                                    decision = resolve_click(
                                        int(sx), int(sy),
                                        enabled=bool(getattr(pcfg, "snap_enabled", True)),
                                        radius_px=int(getattr(pcfg, "snap_radius_px", 100)),
                                        above_bias_px=float(
                                            getattr(pcfg, "snap_above_bias_px", 20.0)),
                                        confirm_ms=float(
                                            getattr(pcfg, "snap_confirm_ms", 250)),
                                        provider=snap_provider,
                                        highlight=(draw_highlight
                                                   if getattr(pcfg, "snap_highlight", True)
                                                   else None),
                                    )
                                    mouse.move(decision.x, decision.y)
                                    mouse.press_left()
                                    pinch_down = True
                                    last_fired = "pinch"
                                    last_utterance = (
                                        f"click -> {decision.target.name[:24]}"
                                        if decision.snapped and decision.target
                                        else "drag / click")
                                else:
                                    mouse.move(int(sx), int(sy))
                            else:
                                if pinch_down:
                                    mouse.release_left()
                                    pinch_down = False
                                if action == MOVE:
                                    mouse.move(int(sx), int(sy))
                                elif action == SCROLL:
                                    n = scroll_acc.update(dy)
                                    if n:
                                        mouse.scroll(n)
                                        last_fired, last_utterance = "victory", "scroll"
                                elif action == ZOOM:
                                    n = zoom_acc.update(dy)
                                    if n:
                                        mouse.zoom(n)
                                        last_fired, last_utterance = "rock", "zoom"
                                elif action == RIGHT_CLICK:
                                    if three_hold.update(True) and click_gate.ready(now):
                                        mouse.click_right()
                                        last_fired, last_utterance = "three", "right click"
                                elif action == VOLUME_UP:
                                    if vol_up_gate.ready(now):
                                        mouse.volume_up()
                                        last_fired, last_utterance = "thumbs_up", "volume up"
                                elif action == VOLUME_DOWN:
                                    if vol_dn_gate.ready(now):
                                        mouse.volume_down()
                                        last_fired, last_utterance = "pinky_up", "volume down"
                                # Window/tab navigation. Held like the
                                # right-click pose and debounced on the same
                                # gate: these jump you between windows, so a
                                # single misread frame must not fire one.
                                elif action == NEXT_TAB:  # noqa: SIM114 - distinct actions
                                    if nav_hold.update(True) and nav_gate.ready(now):
                                        mouse.next_tab()
                                        last_fired, last_utterance = "l_shape", "next tab"
                                elif action == SWITCH_WINDOW:
                                    if nav_hold.update(True) and nav_gate.ready(now):
                                        mouse.switch_window()
                                        last_fired, last_utterance = "switch", "switch window"
                                elif action == TASKBAR:
                                    if nav_hold.update(True) and nav_gate.ready(now):
                                        mouse.taskbar()
                                        last_fired, last_utterance = "four", "taskbar"
                                # action == IDLE: hold the cursor still.
                            # Two-hand gestures: spread = zoom, second thumbs-up
                            # = play/pause (held briefly, debounced).
                            if second is not None:
                                sg, slms = second
                                spread = math.hypot(float(slms[8].x) - float(tip.x),
                                                    float(slms[8].y) - iy)
                                if two_spread_prev is not None:
                                    dspread = spread - two_spread_prev
                                    if abs(dspread) < 0.004:
                                        dspread = 0.0
                                    n = zoom_acc.update(-dspread)  # apart => zoom in
                                    if n:
                                        mouse.zoom(n)
                                        last_fired, last_utterance = "two hands", "zoom"
                                two_spread_prev = spread
                                if pp_hold.update(sg == THUMBS_UP) and pp_gate.ready(now):
                                    mouse.play_pause()
                                    last_fired, last_utterance = "second thumbs_up", "play/pause"
                            else:
                                two_spread_prev = None
                                pp_hold.update(False)
                            # Re-arm the latches/accumulators the moment their pose ends.
                            if action != RIGHT_CLICK:
                                three_hold.update(False)
                            if action != SCROLL:
                                scroll_acc.reset()
                            if action != ZOOM and second is None:
                                zoom_acc.reset()
                            prev_iy = iy
                            prev_tip = (float(tip.x), iy)
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
                                mouse.release_left()
                            except Exception:
                                pass
                            pinch_down = False
                        three_hold.update(False)
                        fist_hold.update(False)
                        pp_hold.update(False)
                        scroll_acc.reset()
                        zoom_acc.reset()
                        ema.reset()  # don't lerp across the gap
                        prev_iy = None
                        prev_tip = None
                        two_spread_prev = None
                    if fist_exit:
                        if pinch_down:
                            try:
                                mouse.release_left()
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
