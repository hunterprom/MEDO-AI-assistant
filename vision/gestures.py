"""Hand-gesture recognition from MediaPipe Hands landmarks.

Split into three pieces so the logic is testable without a camera or MediaPipe:

* :func:`classify_landmarks` — pure geometry: 21 landmarks -> a gesture name.
* :class:`GestureStabilizer` — pure debounce: only fire after N stable frames,
  then hold off for a cooldown.
* :class:`GestureRecognizer` — the MediaPipe wrapper (lazy-imported) that turns a
  camera frame into (gesture, annotated frame).

Landmark indices follow the MediaPipe Hands model (0 = wrist, 4 = thumb tip,
8/12/16/20 = finger tips, 6/10/14/18 = PIP joints).

Finger state comes from JOINT ANGLES, not from a tip-vs-joint distance test: a
distance test says "extended" for every pose between fully open and fully closed,
so a hand halfway into a fist momentarily reads as a real gesture (the classic
"the zoom fired while I was closing my hand" bug). An angle has a genuine middle
band instead, and :func:`classify_landmarks` refuses to guess inside it.
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
L_SHAPE = "l_shape"
FOUR = "four"
UNKNOWN = "unknown"

# Thumb tip and index tip count as "touching" (a pinch) when they are closer
# than this fraction of the palm length (wrist -> middle-finger MCP).
PINCH_RATIO = 0.28

# Joint-angle thresholds, in degrees. A digit is EXTENDED at/above
# EXTENDED_MIN_DEG (180 = perfectly straight) and CURLED at/below
# CURLED_MAX_DEG; the gap between them is the ambiguous band. Mirrored by
# core.config.PointerConfig so they can be tuned from config.yaml.
EXTENDED_MIN_DEG = 160.0
CURLED_MAX_DEG = 100.0
# ROCK drives zoom, so it must be a deliberate horns pose: the index and pinky
# have to actually splay apart, not merely both be straight.
ZOOM_MIN_SPREAD_DEG = 50.0
# L_SHAPE is thumb-vs-index at (roughly) a right angle.
L_SHAPE_TARGET_DEG = 90.0
L_SHAPE_TOLERANCE_DEG = 25.0

# Digits in a fixed order, mapped to their bend joint as (proximal, vertex,
# distal) landmark indices. The four fingers bend at the PIP. The thumb is
# measured at its MCP (2) instead: its IP joint stays nearly straight even when
# the thumb folds across the palm, so an IP angle would call a tucked thumb
# "extended".
_BEND_JOINTS: dict[str, tuple[int, int, int]] = {
    "thumb": (1, 2, 4),
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}

# Direction of each digit for spread measurements: MCP -> tip. Taken from the
# MCP (not the wrist) so the spread is the angle the fingers make with each
# other, independent of where the hand sits in frame.
_DIRECTIONS: dict[str, tuple[int, int]] = {
    "thumb": (2, 4),
    "index": (5, 8),
    "middle": (9, 12),
    "ring": (13, 16),
    "pinky": (17, 20),
}

Point = tuple[float, float]


def _xy(landmark: object) -> Point:
    """Accept either a (x, y, ...) sequence or an object with .x/.y attributes."""
    if hasattr(landmark, "x"):
        return float(landmark.x), float(landmark.y)  # type: ignore[attr-defined]
    return float(landmark[0]), float(landmark[1])  # type: ignore[index]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_between(u: Point, v: Point) -> float:
    """Angle in degrees between two vectors; NaN when either has no direction."""
    nu = math.hypot(*u)
    nv = math.hypot(*v)
    if nu <= 1e-9 or nv <= 1e-9:
        return math.nan
    # Clamp: floating point can push the cosine a hair outside [-1, 1].
    cos = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (nu * nv)))
    return math.degrees(math.acos(cos))


def joint_angle(a: Point, vertex: Point, b: Point) -> float:
    """Interior angle at ``vertex`` between the rays to ``a`` and ``b``, in degrees.

    180 = the three points are collinear (a straight finger), 90 = a right angle,
    0 = folded back on itself. NaN if a ray is degenerate.
    """
    return _angle_between((a[0] - vertex[0], a[1] - vertex[1]),
                          (b[0] - vertex[0], b[1] - vertex[1]))


def finger_angles(landmarks: Sequence[object]) -> dict[str, float]:
    """Bend angle of every digit, keyed by name (thumb/index/middle/ring/pinky)."""
    p = [_xy(lm) for lm in landmarks]
    return {
        name: joint_angle(p[prox], p[vertex], p[distal])
        for name, (prox, vertex, distal) in _BEND_JOINTS.items()
    }


def spread_angle(landmarks: Sequence[object], first: str, second: str) -> float:
    """Angle in degrees between two digits' MCP -> tip direction vectors."""
    p = [_xy(lm) for lm in landmarks]

    def direction(name: str) -> Point:
        mcp, tip = _DIRECTIONS[name]
        return p[tip][0] - p[mcp][0], p[tip][1] - p[mcp][1]
    return _angle_between(direction(first), direction(second))


def _state(angle: float, extended_min_deg: float, curled_max_deg: float) -> bool | None:
    """True = extended, False = curled, None = ambiguous (mid-transition)."""
    if math.isnan(angle):
        return None
    if angle >= extended_min_deg:
        return True
    if angle <= curled_max_deg:
        return False
    return None


def classify_landmarks(
    landmarks: Sequence[object],
    *,
    strict: bool = True,
    extended_min_deg: float = EXTENDED_MIN_DEG,
    curled_max_deg: float = CURLED_MAX_DEG,
    zoom_min_spread_deg: float = ZOOM_MIN_SPREAD_DEG,
    l_shape_tolerance_deg: float = L_SHAPE_TOLERANCE_DEG,
) -> str:
    """Classify 21 hand landmarks into a gesture name.

    Rotation- and distance-tolerant: every test is an angle between landmarks or
    a ratio against the palm length, so it holds for a hand at any orientation
    and any distance from the camera.

    With ``strict`` (the default) a single digit whose bend angle lands between
    ``curled_max_deg`` and ``extended_min_deg`` makes the whole hand UNKNOWN.
    That is what stops a hand in transit — closing into a fist, opening out of
    one — from flashing through a real gesture and firing an action. Pass
    ``strict=False`` for the old permissive behaviour, where an ambiguous digit
    is resolved to whichever side of the band it is nearer.
    """
    if landmarks is None or len(landmarks) < 21:
        return UNKNOWN

    p = [_xy(lm) for lm in landmarks]
    wrist = p[0]

    angles = finger_angles(p)
    states = {n: _state(a, extended_min_deg, curled_max_deg) for n, a in angles.items()}
    if strict and any(s is None for s in states.values()):
        return UNKNOWN

    # Non-strict fallback: split the ambiguous band down the middle so every
    # digit still gets a verdict (NaN compares False, i.e. counts as curled).
    midpoint = (extended_min_deg + curled_max_deg) / 2.0
    resolved = {
        n: s if s is not None else angles[n] >= midpoint
        for n, s in states.items()
    }
    thumb = resolved["thumb"]
    index = resolved["index"]
    middle = resolved["middle"]
    ring = resolved["ring"]
    pinky = resolved["pinky"]
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
    if n_fingers == 4:
        return FOUR
    if not thumb and n_fingers == 0:
        return FIST
    if thumb and n_fingers == 0:
        return THUMBS_UP
    # L_SHAPE before POINT_UP: both are "index only", but the L needs the thumb
    # out at roughly a right angle. A thumb at any other angle stays POINT_UP,
    # which is how people naturally point.
    if thumb and n_fingers == 1 and index:
        spread = spread_angle(p, "thumb", "index")
        if abs(spread - L_SHAPE_TARGET_DEG) <= l_shape_tolerance_deg:
            return L_SHAPE
    if n_fingers == 1 and index:
        return POINT_UP
    if n_fingers == 1 and pinky:
        return PINKY_UP
    if n_fingers == 2 and index and middle:
        return VICTORY
    if index and middle and ring and not pinky:
        return THREE
    if index and pinky and not middle and not ring:
        # Zoom pose: demand a real splay. Index and pinky held straight but
        # parallel is what a half-closed hand looks like, and it used to zoom.
        spread = spread_angle(p, "index", "pinky")
        if not math.isnan(spread) and spread >= zoom_min_spread_deg:
            return ROCK
        return UNKNOWN
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

    def __init__(
        self,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        *,
        strict: bool = True,
        extended_min_deg: float = EXTENDED_MIN_DEG,
        curled_max_deg: float = CURLED_MAX_DEG,
        zoom_min_spread_deg: float = ZOOM_MIN_SPREAD_DEG,
        l_shape_tolerance_deg: float = L_SHAPE_TOLERANCE_DEG,
    ) -> None:
        import mediapipe as mp

        # Keyword-only with defaults so existing call sites keep working; the
        # caller passes config.vision.pointer values in when it wants to tune.
        self._opts = dict(
            strict=strict,
            extended_min_deg=extended_min_deg,
            curled_max_deg=curled_max_deg,
            zoom_min_spread_deg=zoom_min_spread_deg,
            l_shape_tolerance_deg=l_shape_tolerance_deg,
        )
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
                hands.append((classify_landmarks(lms, **self._opts), lms))
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
