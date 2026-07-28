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
from dataclasses import dataclass

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
# Two-handed. Kept alongside ROCK, which remains the one-hand zoom pose.
ZOOM = "zoom"
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
# Minimum depth-inside-band across a signature's required digits. Raising it
# demands firmer poses; lowering it accepts fingers nearer the ambiguous band.
MIN_CONFIDENCE = 0.85
# Consecutive frames the SAME signature must hold before it fires.
HOLD_FRAMES = 3

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


# --- exact gesture signatures --------------------------------------------------
#
# One entry per gesture, and this table is the ONLY definition of what each
# gesture is. Recognition used to be an if-chain of loose heuristics, so poses
# that resembled two gestures resolved to whichever branch happened to come
# first, and a hand mid-transition could satisfy a branch by accident.
#
# A signature demands an exact per-finger state. Anything that does not match
# every required digit is simply not that gesture — there is no "close enough".

#: Finger state in a signature. ANY means the digit is genuinely irrelevant to
#: this gesture, not that we could not decide: an ambiguous digit still fails.
EXTENDED = "extended"
CURLED = "curled"
ANY = "any"

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class GestureSignature:
    """The exact hand shape that IS a given gesture."""

    name: str
    thumb: str
    index: str
    middle: str
    ring: str
    pinky: str
    #: Extra geometric gates beyond finger state, as (label, predicate). All
    #: must pass. These are hard booleans, not scored — a pinch either has the
    #: tips touching or it does not.
    constraints: tuple = ()
    #: Hands this gesture needs. 2 = both hands must match it independently;
    #: two hands making the same thumb-and-index pose ARE mirror images.
    hands: int = 1
    #: Per-gesture floor, overriding the global one. Raised for gestures whose
    #: action is disruptive or easily confused with a neighbour.
    min_confidence: float | None = None
    doc: str = ""

    def states(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in FINGERS}


def _tips_touching(p, ratio: float = PINCH_RATIO) -> bool:
    """Thumb tip on index tip, scaled by palm length (distance-invariant)."""
    palm = _dist(p[0], p[9])
    return palm > 0 and _dist(p[4], p[8]) < ratio * palm


def _tips_apart(p) -> bool:
    return not _tips_touching(p)


def _l_angle(p) -> bool:
    """Thumb and index at roughly a right angle."""
    spread = spread_angle(p, "thumb", "index")
    return (not math.isnan(spread)
            and abs(spread - L_SHAPE_TARGET_DEG) <= L_SHAPE_TOLERANCE_DEG)


def _not_l_angle(p) -> bool:
    return not _l_angle(p)


def _horns_splayed(p) -> bool:
    """Index and pinky genuinely apart — a half-closed hand has them parallel."""
    spread = spread_angle(p, "index", "pinky")
    return not math.isnan(spread) and spread >= ZOOM_MIN_SPREAD_DEG


def _v_splayed(p) -> bool:
    """Index and middle apart, so a two-finger point is not a victory sign."""
    spread = spread_angle(p, "index", "middle")
    return not math.isnan(spread) and spread >= 15.0


#: Ordered most-specific first: the first signature that matches wins, so a
#: pose that could read as two gestures resolves the same way every time.
SIGNATURES: tuple[GestureSignature, ...] = (
    GestureSignature(
        PINCH, EXTENDED, EXTENDED, CURLED, CURLED, CURLED,
        constraints=(("tips touching", _tips_touching),),
        doc="thumb and index tips together, other fingers curled",
    ),
    GestureSignature(
        L_SHAPE, EXTENDED, EXTENDED, CURLED, CURLED, CURLED,
        constraints=(("tips apart", _tips_apart), ("thumb-index ~90 deg", _l_angle)),
        doc="thumb and index straight at a right angle, other fingers curled",
    ),
    GestureSignature(
        ZOOM, EXTENDED, EXTENDED, CURLED, CURLED, CURLED,
        constraints=(("tips apart", _tips_apart), ("not an L", _not_l_angle)),
        hands=2, min_confidence=0.90,
        doc="BOTH hands: thumb and index extended, others curled, hands mirrored",
    ),
    GestureSignature(
        ROCK, ANY, EXTENDED, CURLED, CURLED, EXTENDED,
        constraints=(("index-pinky splayed", _horns_splayed),),
        doc="horns: index and pinky extended and splayed, middle and ring curled",
    ),
    GestureSignature(
        VICTORY, CURLED, EXTENDED, EXTENDED, CURLED, CURLED,
        constraints=(("index-middle splayed", _v_splayed),),
        doc="V sign: index and middle extended and apart, thumb tucked",
    ),
    GestureSignature(
        THREE, CURLED, EXTENDED, EXTENDED, EXTENDED, CURLED,
        doc="index, middle and ring extended; thumb and pinky curled",
    ),
    GestureSignature(
        FOUR, CURLED, EXTENDED, EXTENDED, EXTENDED, EXTENDED,
        doc="all four fingers extended, thumb tucked across the palm",
    ),
    GestureSignature(
        OPEN_PALM, EXTENDED, EXTENDED, EXTENDED, EXTENDED, EXTENDED,
        doc="every digit extended",
    ),
    GestureSignature(
        THUMBS_UP, EXTENDED, CURLED, CURLED, CURLED, CURLED,
        # No "points up" gate on purpose: this module is rotation tolerant by
        # design, and thumb-extended with all four fingers curled is already
        # unique in the table. An absolute up-test would fail a tilted hand.
        doc="thumb extended, all four fingers curled",
    ),
    GestureSignature(
        PINKY_UP, CURLED, CURLED, CURLED, CURLED, EXTENDED,
        doc="pinky extended, everything else curled",
    ),
    GestureSignature(
        POINT_UP, ANY, EXTENDED, CURLED, CURLED, CURLED,
        constraints=(("tips apart", _tips_apart), ("not an L", _not_l_angle)),
        doc="index extended, other fingers curled; the natural pointing pose",
    ),
    GestureSignature(
        FIST, CURLED, CURLED, CURLED, CURLED, CURLED,
        doc="every digit curled",
    ),
)

#: Name -> signature, for lookups and the --list check.
SIGNATURES_BY_NAME = {s.name: s for s in SIGNATURES}


def finger_confidence(angle: float, want: str, extended_min_deg: float,
                      curled_max_deg: float) -> float:
    """How firmly a digit holds the state a signature asked for, 0..1.

    There is no model score behind a geometric classifier, so confidence is
    depth inside the band: a finger at exactly ``extended_min_deg`` scores 0
    and a dead-straight one scores 1. A gesture's confidence is the MINIMUM
    across the digits it requires, so "0.85" means every required finger is
    comfortably inside its band rather than sitting on the line.

    ANY scores 1.0 — an irrelevant digit cannot weaken the match. An ambiguous
    digit is rejected before this is reached.
    """
    if want == ANY:
        return 1.0
    if math.isnan(angle):
        return 0.0
    # Both states are scored against the SAME band width — the room a finger
    # has above the extended threshold. Normalising "curled" against zero
    # instead would score a perfectly ordinary 40-degree curl at 0.6 and fail
    # it, because fingers do not fold flat.
    span = max(1e-6, 180.0 - extended_min_deg)
    past = (angle - extended_min_deg) if want == EXTENDED else (curled_max_deg - angle)
    return max(0.0, min(1.0, past / span))


def match_signature(
    landmarks: Sequence[object],
    signature: GestureSignature,
    *,
    extended_min_deg: float = EXTENDED_MIN_DEG,
    curled_max_deg: float = CURLED_MAX_DEG,
    strict: bool = True,
) -> float | None:
    """Confidence that one hand IS this gesture, or None if it is not.

    None and 0.0 are different answers: None means the shape is wrong, 0.0
    means the shape is right but every required finger is on the edge of its
    band. The caller wants to reject the first outright and threshold the
    second.
    """
    if landmarks is None or len(landmarks) < 21:
        return None
    p = [_xy(lm) for lm in landmarks]
    angles = finger_angles(p)
    wanted = signature.states()

    scores = []
    for finger, want in wanted.items():
        angle = angles[finger]
        state = _state(angle, extended_min_deg, curled_max_deg)
        resolved_from_ambiguous = False
        if state is None:
            # Mid-transition. Strict mode fails the whole hand even for a digit
            # this gesture does not care about — that is the point: a hand in
            # motion is not holding any pose. Permissive mode resolves it to
            # whichever side of the band it is nearer, the old behaviour.
            if strict:
                return None
            midpoint = (extended_min_deg + curled_max_deg) / 2.0
            state = angle >= midpoint
            resolved_from_ambiguous = True
        if want != ANY and state != (want == EXTENDED):
            return None
        if resolved_from_ambiguous:
            # Scoring it would give 0 (it is outside its band by definition) and
            # sink the whole gesture — which would make permissive mode reject
            # everything it just went out of its way to accept.
            continue
        scores.append(finger_confidence(angle, want, extended_min_deg, curled_max_deg))

    for _label, predicate in signature.constraints:
        try:
            if not predicate(p):
                return None
        except Exception:                     # noqa: BLE001 - geometry on junk data
            return None
    return min(scores) if scores else 1.0


def classify_landmarks(
    landmarks: Sequence[object],
    *,
    strict: bool = True,
    extended_min_deg: float = EXTENDED_MIN_DEG,
    curled_max_deg: float = CURLED_MAX_DEG,
    zoom_min_spread_deg: float = ZOOM_MIN_SPREAD_DEG,
    l_shape_tolerance_deg: float = L_SHAPE_TOLERANCE_DEG,
    min_confidence: float = MIN_CONFIDENCE,
) -> str:
    """Classify one hand against :data:`SIGNATURES`; UNKNOWN if none match.

    Walks the table in order and returns the first signature whose finger
    states, constraints and confidence all pass. Order is priority, so a pose
    that could read as two gestures resolves the same way every time — the old
    if-chain made that ordering implicit and easy to break.

    Rotation- and distance-tolerant: every test is an angle between landmarks
    or a ratio against palm length, so it holds at any hand orientation and any
    distance from the camera.

    With ``strict`` (the default) a single digit whose bend angle lands in the
    ambiguous band fails every signature, so a hand in transit — closing into a
    fist, opening out of one — matches nothing instead of flashing through a
    real gesture and firing an action.
    """
    name, _confidence = classify_with_confidence(
        landmarks, strict=strict, extended_min_deg=extended_min_deg,
        curled_max_deg=curled_max_deg, zoom_min_spread_deg=zoom_min_spread_deg,
        l_shape_tolerance_deg=l_shape_tolerance_deg, min_confidence=min_confidence,
    )
    return name


def classify_with_confidence(
    landmarks: Sequence[object],
    *,
    strict: bool = True,
    extended_min_deg: float = EXTENDED_MIN_DEG,
    curled_max_deg: float = CURLED_MAX_DEG,
    zoom_min_spread_deg: float = ZOOM_MIN_SPREAD_DEG,
    l_shape_tolerance_deg: float = L_SHAPE_TOLERANCE_DEG,
    min_confidence: float = MIN_CONFIDENCE,
) -> tuple[str, float]:
    """As :func:`classify_landmarks`, but also returns how firm the match was."""
    global ZOOM_MIN_SPREAD_DEG, L_SHAPE_TOLERANCE_DEG
    # The two shape constraints read their thresholds from module state so the
    # signature predicates stay simple one-argument functions; swap them for
    # the caller's values for the duration of this classification.
    prev_zoom, prev_l = ZOOM_MIN_SPREAD_DEG, L_SHAPE_TOLERANCE_DEG
    ZOOM_MIN_SPREAD_DEG, L_SHAPE_TOLERANCE_DEG = zoom_min_spread_deg, l_shape_tolerance_deg
    try:
        for signature in SIGNATURES:
            if signature.hands != 1:
                continue                      # two-hand poses need classify_pair
            confidence = match_signature(
                landmarks, signature, extended_min_deg=extended_min_deg,
                curled_max_deg=curled_max_deg, strict=strict)
            if confidence is None:
                continue
            floor = signature.min_confidence
            if floor is None:
                floor = min_confidence
            if confidence >= floor:
                return signature.name, confidence
            # Right shape, not held firmly enough: an in-between pose. Keep
            # looking rather than falling through to a looser gesture.
        return UNKNOWN, 0.0
    finally:
        ZOOM_MIN_SPREAD_DEG, L_SHAPE_TOLERANCE_DEG = prev_zoom, prev_l


def classify_pair(
    hands: Sequence[object],
    *,
    strict: bool = True,
    extended_min_deg: float = EXTENDED_MIN_DEG,
    curled_max_deg: float = CURLED_MAX_DEG,
    min_confidence: float = MIN_CONFIDENCE,
) -> tuple[str, float]:
    """Two-hand gestures. ``hands`` is a sequence of landmark lists.

    A two-hand signature must be satisfied by BOTH hands independently, and the
    pair's confidence is the weaker of the two. Mirroring falls out of that:
    two hands making the same thumb-and-index shape while facing each other ARE
    mirror images, so no handedness lookup is needed to enforce it.
    """
    if not hands or len(hands) < 2:
        return UNKNOWN, 0.0
    for signature in SIGNATURES:
        if signature.hands != 2:
            continue
        scores = [
            match_signature(h, signature, extended_min_deg=extended_min_deg,
                            curled_max_deg=curled_max_deg, strict=strict)
            for h in hands[:2]
        ]
        if any(s is None for s in scores):
            continue
        confidence = min(scores)
        floor = signature.min_confidence
        if floor is None:
            floor = min_confidence
        if confidence >= floor:
            return signature.name, confidence
    return UNKNOWN, 0.0


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
    """Debounce plus hysteresis: hold N frames to fire, release to re-arm.

    Two separate protections, both needed:

    * **hold_frames** — the same signature must survive N consecutive frames.
      One misread frame in a stream of noise can never fire anything.
    * **hysteresis** — after firing, the hand must return to neutral (UNKNOWN,
      or any *other* gesture) before that gesture can fire again. Without it a
      pose held steady re-fires every cooldown, and a hand drifting out of a
      pose and back in double-fires. Holding a gesture is one event, not many.

    ``cooldown_frames`` remains as a floor between *different* firings so a
    hand sweeping through several poses cannot machine-gun actions.
    """

    def __init__(self, stability_frames: int = HOLD_FRAMES,
                 cooldown_frames: int = 30) -> None:
        self._need = max(1, stability_frames)
        self._cooldown = max(0, cooldown_frames)
        self._current = UNKNOWN
        self._count = 0
        self._cooldown_left = 0
        #: The gesture that fired and has not yet been released. None = armed.
        self._latched: str | None = None

    def update(self, gesture: str) -> str | None:
        """Feed one frame's gesture; returns a name only when one fires."""
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        if gesture == self._current:
            self._count += 1
        else:
            self._current = gesture
            self._count = 1

        # Hysteresis: anything other than the latched gesture releases it. That
        # includes UNKNOWN, so relaxing the hand re-arms exactly as expected.
        if self._latched is not None and gesture != self._latched:
            self._latched = None

        if gesture == UNKNOWN:
            return None
        if self._latched is not None:
            return None                     # still held down; not a new event
        # Fire once the hold is satisfied. `>=` not `==`: if a previous
        # gesture's cooldown was still counting down on the exact frame _count
        # first hit _need, an `==` test would miss it and _count would climb
        # past _need forever, so that gesture could never fire again.
        if self._count >= self._need and self._cooldown_left == 0:
            self._latched = gesture
            self._cooldown_left = self._cooldown
            return gesture
        return None

    def reset(self) -> None:
        """Forget all state (pointer mode toggling, camera restart)."""
        self._current = UNKNOWN
        self._count = 0
        self._cooldown_left = 0
        self._latched = None


def loaded_gestures() -> list[str]:
    """Every gesture the table defines, in priority order."""
    return [s.name for s in SIGNATURES]


def dispatch_report(pose_actions: dict | None = None,
                    utterances: dict | None = None) -> list[tuple[str, str]]:
    """(gesture, where it goes) for every signature — the reachability check.

    The bug this exists for: signatures were added to the table but never bound
    to anything, so they classified perfectly and then did nothing. Anything
    reported as "UNBOUND" is defined but unreachable.
    """
    from vision.pointer import build_pose_actions

    actions = build_pose_actions(pose_actions)
    said = utterances or {}
    report = []
    for name in loaded_gestures():
        where = []
        if name in actions:
            where.append(f"pointer:{actions[name]}")
        if name in said:
            where.append(f"say:{said[name]!r}")
        if name == PINCH:
            where.append("pointer:drag (via index_thumb_pinch)")
        if name == ZOOM:
            where.append("pointer:zoom (two-hand)")
        report.append((name, ", ".join(where) or "UNBOUND"))
    return report


def _main(argv: list[str] | None = None) -> int:
    """``python -m vision.gestures --list`` — what is defined and where it goes."""
    import argparse

    parser = argparse.ArgumentParser(description="MEDO gesture signatures")
    parser.add_argument("--list", action="store_true",
                        help="print every gesture, its signature and its binding")
    args = parser.parse_args(argv)
    if not args.list:
        parser.print_help()
        return 0

    from vision.run import DEFAULT_GESTURES

    bindings = dict(dispatch_report(utterances=DEFAULT_GESTURES))
    print(f"{len(SIGNATURES)} gesture signatures (priority order)")
    print()
    header = f"{'gesture':<11} {'hands':<6} {'T I M R P':<10} {'conf':<5} binding"
    print(header)
    print("-" * len(header))
    unbound = 0
    for sig in SIGNATURES:
        states = sig.states()
        row = " ".join({EXTENDED: "E", CURLED: "C", ANY: "."}[states[f]]
                       for f in FINGERS)
        floor = sig.min_confidence if sig.min_confidence is not None else MIN_CONFIDENCE
        binding = bindings[sig.name]
        unbound += binding == "UNBOUND"
        print(f"{sig.name:<11} {sig.hands:<6} {row:<10} {floor:<5.2f} {binding}")
        if sig.constraints:
            print(f"{'':<11} {'':<6} also: "
                  + ", ".join(label for label, _ in sig.constraints))
    if unbound:
        print()
        print(f"{unbound} gesture(s) UNBOUND - defined but nothing dispatches them.")
    return 1 if unbound else 0


if __name__ == "__main__":
    raise SystemExit(_main())


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
        draw_overlay: bool = False,
    ) -> None:
        import mediapipe as mp

        # Whether to paint the coloured hand skeleton onto the frame. Off by
        # default: that frame is also what the vision model sees, and tracking
        # works without it.
        self._draw_overlay = draw_overlay

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
                # Tracking always runs (classify_landmarks above); only the
                # visible skeleton is optional, so the streamed feed and the
                # vision model see a clean camera image by default.
                if self._draw_overlay:
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
