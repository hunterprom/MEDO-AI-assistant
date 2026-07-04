"""Hand-gesture recognition from MediaPipe Hands landmarks.

Split into three pieces so the logic is testable without a camera or MediaPipe:

* :func:`classify_landmarks` — pure geometry: 21 landmarks -> a gesture name.
* :class:`GestureStabilizer` — pure debounce: only fire after N stable frames,
  then hold off for a cooldown.
* :class:`GestureRecognizer` — the MediaPipe wrapper (lazy-imported) that turns a
  camera frame into (gesture, annotated frame).

Landmark indices follow the MediaPipe Hands model (0 = wrist, 4 = thumb tip,
8/12/16/20 = finger tips, 6/10/14/18 = PIP joints).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Gesture names (also the keys used in config.vision.gestures).
OPEN_PALM = "open_palm"
FIST = "fist"
THUMBS_UP = "thumbs_up"
POINT_UP = "point_up"
VICTORY = "victory"
PINCH = "pinch"
THREE = "three"
ROCK = "rock"
PINKY_UP = "pinky_up"
UNKNOWN = "unknown"

# Thumb tip and index tip count as "touching" (a pinch) when they are closer
# than this fraction of the palm length (wrist -> middle-finger MCP).
PINCH_RATIO = 0.28

Point = tuple[float, float]


def _xy(landmark: object) -> Point:
    """Accept either a (x, y, ...) sequence or an object with .x/.y attributes."""
    if hasattr(landmark, "x"):
        return float(landmark.x), float(landmark.y)  # type: ignore[attr-defined]
    return float(landmark[0]), float(landmark[1])  # type: ignore[index]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def classify_landmarks(landmarks: Sequence[object]) -> str:
    """Classify 21 hand landmarks into a gesture name.

    Uses a rotation-tolerant rule: a finger is "extended" when its tip is farther
    from the wrist than its middle joint. Works for hands at any angle, which a
    naive "tip is above the joint" test does not.
    """
    if landmarks is None or len(landmarks) < 21:
        return UNKNOWN

    p = [_xy(lm) for lm in landmarks]
    wrist = p[0]

    def extended(tip: int, joint: int) -> bool:
        return _dist(p[tip], wrist) > _dist(p[joint], wrist)

    thumb = extended(4, 2)
    index = extended(8, 6)
    middle = extended(12, 10)
    ring = extended(16, 14)
    pinky = extended(20, 18)
    n_fingers = sum((index, middle, ring, pinky))

    # PINCH first: with thumb and index tips touching, the index often still
    # measures as "extended", which would misread as OPEN_PALM below. The scale
    # reference is the palm length (wrist -> middle-finger MCP), so the rule is
    # distance-invariant: it works whether the hand is near or far from the camera.
    palm = _dist(wrist, p[9])
    if middle and ring and pinky and _dist(p[4], p[8]) < PINCH_RATIO * palm:
        return PINCH
    if thumb and n_fingers == 4:
        return OPEN_PALM
    if not thumb and n_fingers == 0:
        return FIST
    if thumb and n_fingers == 0:
        return THUMBS_UP
    if n_fingers == 1 and index:
        return POINT_UP
    if n_fingers == 1 and pinky:
        return PINKY_UP
    if n_fingers == 2 and index and middle:
        return VICTORY
    if index and middle and ring and not pinky:
        return THREE
    if index and pinky and not middle and not ring:
        return ROCK
    return UNKNOWN


def index_thumb_pinch(landmarks: Sequence[object], ratio: float = PINCH_RATIO) -> bool:
    """True when the thumb tip touches the index tip — regardless of other fingers.

    Pointer-mode click detector. Unlike :func:`classify_landmarks`' ``PINCH`` (which
    also requires middle/ring/pinky extended), this only measures the thumb→index
    tip gap against palm length, so it fires from the natural pointing pose where
    the other fingers are curled. Distance-invariant: works near or far from camera.
    """
    if landmarks is None or len(landmarks) < 21:
        return False
    p = [_xy(lm) for lm in landmarks]
    palm = _dist(p[0], p[9])
    if palm <= 0:
        return False
    return _dist(p[4], p[8]) < ratio * palm


class GestureStabilizer:
    """Emit a gesture only after it holds for ``stability_frames`` frames, then
    wait ``cooldown_frames`` before the same gesture can fire again."""

    def __init__(self, stability_frames: int = 6, cooldown_frames: int = 30) -> None:
        self._need = max(1, stability_frames)
        self._cooldown = max(0, cooldown_frames)
        self._current = UNKNOWN
        self._count = 0
        self._cooldown_left = 0
        self._last_fired = UNKNOWN

    def update(self, gesture: str) -> str | None:
        """Feed one frame's gesture; return a gesture name when one is confirmed."""
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        if gesture == self._current:
            self._count += 1
        else:
            self._current = gesture
            self._count = 1

        if gesture == UNKNOWN:
            self._last_fired = UNKNOWN  # let a repeat re-fire after a neutral pose
            return None

        ready = self._count == self._need  # fire exactly once at the threshold
        if ready and self._cooldown_left == 0 and gesture != self._last_fired:
            self._last_fired = gesture
            self._cooldown_left = self._cooldown
            return gesture
        return None


class GestureRecognizer:
    """MediaPipe Hands wrapper: a BGR frame -> (gesture, annotated, landmarks)."""

    def __init__(self, min_detection_confidence: float, min_tracking_confidence: float) -> None:
        import mediapipe as mp

        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # second hand: spread-zoom + media control
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._draw = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles

    def process(self, frame_bgr):
        """Return (gesture, annotated_frame, landmarks, hands) for one frame.

        ``landmarks`` is the first hand's 21-point list (objects with .x/.y)
        or None. ``hands`` is ``[(gesture_name, landmarks), ...]`` for up to
        two visible hands in MediaPipe order — the engine re-orders them so
        the cursor stays on the hand the user was already pointing with.
        """
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)

        gesture = UNKNOWN
        annotated = frame_bgr
        landmarks = None
        hands: list[tuple[str, object]] = []
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks[:2]:
                lms = hand.landmark
                hands.append((classify_landmarks(lms), lms))
                self._draw.draw_landmarks(
                    annotated,
                    hand,
                    self._mp.solutions.hands.HAND_CONNECTIONS,
                    self._styles.get_default_hand_landmarks_style(),
                    self._styles.get_default_hand_connections_style(),
                )
            gesture, landmarks = hands[0]
        return gesture, annotated, landmarks, hands

    def close(self) -> None:
        self._hands.close()
