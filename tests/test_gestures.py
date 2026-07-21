"""Hand-gesture classification and debounce — pure logic, no camera/MediaPipe."""

from __future__ import annotations

import math

import pytest

from vision.gestures import (
    FIST, FOUR, L_SHAPE, OPEN_PALM, PINCH, PINKY_UP, POINT_UP, ROCK, THREE,
    THUMBS_UP, UNKNOWN, VICTORY, GestureStabilizer, classify_landmarks,
    finger_angles, index_thumb_pinch, joint_angle, spread_angle,
)

WRIST = (0.5, 0.9)      # normalized image coords, y grows downward
MCP_Y = 0.62            # knuckle row -> palm length (wrist -> middle MCP) = 0.28
THUMB_MCP = (0.36, 0.74)
STRAIGHT_DEG = 180.0    # a fully extended digit is collinear
CURLED_DEG = 40.0       # a folded digit doubles back on itself

# (MCP, PIP, DIP, TIP) landmark indices, x of the knuckle, and how far the
# finger splays from vertical (negative = toward the thumb). The fan is wide so
# the default index/pinky pair clears the rock/zoom spread minimum, like a real
# horns pose does.
FINGERS = {
    "index":  ((5, 6, 7, 8), 0.44, -30.0),
    "middle": ((9, 10, 11, 12), 0.50, -10.0),
    "ring":   ((13, 14, 15, 16), 0.56, 10.0),
    "pinky":  ((17, 18, 19, 20), 0.62, 30.0),
}


def _rotate(v, deg):
    a = math.radians(deg)
    return (v[0] * math.cos(a) - v[1] * math.sin(a),
            v[0] * math.sin(a) + v[1] * math.cos(a))


def _unit(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    return (dx / n, dy / n)


def _step(origin, direction, length):
    return (origin[0] + length * direction[0], origin[1] + length * direction[1])


def _bend(vertex, other, angle_deg, length):
    """Point ``length`` away from ``vertex`` at ``angle_deg`` to the ray vertex->other.

    180 gives a straight joint, small angles fold the segment back alongside it —
    which is exactly the interior angle the classifier measures.
    """
    return _step(vertex, _rotate(_unit(vertex, other), angle_deg), length)


def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def hand(thumb=False, index=False, middle=False, ring=False, pinky=False,
         bend=None, splay=None, thumb_splay_deg=-45.0):
    """Build 21 synthetic landmarks with the chosen digits extended.

    Geometry is real: each digit is MCP -> PIP -> TIP with a controlled interior
    angle at the middle joint, so the joint-angle classifier sees what a camera
    would. ``bend`` overrides that angle per digit (to build in-between poses),
    ``splay`` overrides how far a finger fans out from vertical (to control the
    spread between fingers), and ``thumb_splay_deg`` aims the thumb.
    """
    bend = bend or {}
    splay = splay or {}
    extended = {"thumb": thumb, "index": index, "middle": middle,
                "ring": ring, "pinky": pinky}

    def angle_of(name):
        return bend.get(name, STRAIGHT_DEG if extended[name] else CURLED_DEG)

    p = [(0.0, 0.0)] * 21
    p[0] = WRIST
    for name, (idx, mcp_x, default_splay) in FINGERS.items():
        mcp_i, pip_i, dip_i, tip_i = idx
        direction = math.radians(splay.get(name, default_splay))
        d = (math.sin(direction), -math.cos(direction))
        p[mcp_i] = (mcp_x, MCP_Y)
        p[pip_i] = _step(p[mcp_i], d, 0.12)
        p[tip_i] = _bend(p[pip_i], p[mcp_i], angle_of(name), 0.18)
        p[dip_i] = _mid(p[pip_i], p[tip_i])
    t = math.radians(thumb_splay_deg)
    d = (math.sin(t), -math.cos(t))
    p[2] = THUMB_MCP
    p[1] = _step(p[2], d, -0.10)
    # Negative angle so a curled thumb folds across the palm, not away from it.
    p[4] = _bend(p[2], p[1], -angle_of("thumb"), 0.18)
    p[3] = _mid(p[2], p[4])
    return p


def _reach_thumb(p, target):
    """Aim a straight thumb so its tip lands exactly on ``target``."""
    p = list(p)
    d = _unit(THUMB_MCP, target)
    p[2] = THUMB_MCP
    p[1] = _step(p[2], d, -0.10)     # CMC behind the MCP => joint angle 180
    p[4] = target
    p[3] = _mid(p[2], p[4])
    return p


def pinch_hand(touching=True):
    """Middle/ring/pinky extended; thumb & index tips touching (or spread apart).

    The pinch threshold is 0.28 * dist(wrist 0, middle MCP 9) = 0.28 * 0.28,
    i.e. ~0.078 in these coordinates.
    """
    p = hand(index=True, middle=True, ring=True, pinky=True)
    target = (p[8][0] + 0.02, p[8][1] + 0.01) if touching else (0.20, 0.55)
    return _reach_thumb(p, target)


@pytest.mark.parametrize("builder, expected", [
    (dict(thumb=True, index=True, middle=True, ring=True, pinky=True), OPEN_PALM),
    (dict(), FIST),
    (dict(thumb=True), THUMBS_UP),
    (dict(index=True), POINT_UP),
    (dict(pinky=True), PINKY_UP),         # pinky only = volume down in pointer mode
    (dict(index=True, middle=True), VICTORY),
    (dict(ring=True), UNKNOWN),           # a lone ring finger stays unknown
    # THREE: index+middle+ring up, pinky folded — thumb must not matter.
    (dict(index=True, middle=True, ring=True), THREE),
    (dict(thumb=True, index=True, middle=True, ring=True), THREE),
    # ROCK: index+pinky up, middle+ring folded — thumb must not matter.
    (dict(index=True, pinky=True), ROCK),
    (dict(thumb=True, index=True, pinky=True), ROCK),
    (dict(middle=True, pinky=True), UNKNOWN),   # wrong two fingers => unknown
    # FOUR: all four fingers up with the thumb tucked in.
    (dict(index=True, middle=True, ring=True, pinky=True), FOUR),
])
def test_classify(builder, expected):
    assert classify_landmarks(hand(**builder)) == expected


# --- joint-angle geometry ----------------------------------------------------

def test_joint_angle_on_known_geometry():
    assert joint_angle((0.0, 1.0), (0.0, 0.0), (0.0, -1.0)) == pytest.approx(180.0)
    assert joint_angle((0.0, 1.0), (0.0, 0.0), (1.0, 0.0)) == pytest.approx(90.0)
    assert joint_angle((0.0, 1.0), (0.0, 0.0), (1.0, 1.0)) == pytest.approx(45.0)
    # A joint with no direction (duplicate landmarks) must not blow up.
    assert math.isnan(joint_angle((0.0, 0.0), (0.0, 0.0), (1.0, 0.0)))


def test_finger_angles_separate_straight_from_folded():
    angles = finger_angles(hand(index=True))
    assert angles["index"] == pytest.approx(180.0)          # straight finger
    assert angles["middle"] == pytest.approx(CURLED_DEG)    # folded back on itself
    assert angles["thumb"] == pytest.approx(CURLED_DEG)


def test_spread_angle_measures_the_fan_between_fingers():
    p = hand(index=True, pinky=True)                        # splayed -30 / +30
    assert spread_angle(p, "index", "pinky") == pytest.approx(60.0)
    tight = hand(index=True, pinky=True, splay={"index": -10.0, "pinky": 10.0})
    assert spread_angle(tight, "index", "pinky") == pytest.approx(20.0)


# --- strictness: no gestures from in-between poses ---------------------------

def closing_fist():
    """A hand caught halfway into a fist: index and pinky already straight, middle
    and ring still on their way down. The old tip-vs-joint test read this as ROCK
    and fired a zoom mid-motion — that is the bug strict mode exists to kill."""
    return hand(index=True, pinky=True, bend={"middle": 120.0, "ring": 120.0})


def test_half_curled_hand_is_unknown_in_strict_mode():
    p = closing_fist()
    assert 100.0 < finger_angles(p)["middle"] < 160.0       # inside the dead band
    assert classify_landmarks(p) == UNKNOWN                 # strict is the default


def test_same_half_curled_hand_still_classifies_when_not_strict():
    # Non-strict resolves each ambiguous finger to the nearer side of the band,
    # reproducing the old permissive behaviour.
    assert classify_landmarks(closing_fist(), strict=False) == ROCK


def test_an_ambiguous_thumb_alone_blocks_the_gesture():
    p = hand(index=True, bend={"thumb": 125.0})
    assert classify_landmarks(p) == UNKNOWN
    assert classify_landmarks(p, strict=False) == POINT_UP


def test_thresholds_are_tunable():
    # Widen "extended" far enough and the half-closed fingers count as up again.
    assert classify_landmarks(closing_fist(), extended_min_deg=110.0) == FOUR


# --- ROCK needs a real spread, and the new L_SHAPE ---------------------------

def test_rock_requires_the_fingers_to_splay():
    tight = hand(index=True, pinky=True, splay={"index": -10.0, "pinky": 10.0})
    assert classify_landmarks(tight) == UNKNOWN             # 20 deg < 50 deg minimum
    assert classify_landmarks(tight, zoom_min_spread_deg=15.0) == ROCK


def test_l_shape_needs_a_right_angle_between_thumb_and_index():
    # thumb_splay_deg is measured from vertical and the index sits at -30, so
    # -120 puts the two digits exactly 90 deg apart.
    square = hand(thumb=True, index=True, thumb_splay_deg=-120.0)
    assert spread_angle(square, "thumb", "index") == pytest.approx(90.0)
    assert classify_landmarks(square) == L_SHAPE

    # 40 deg apart is a normal pointing hand with the thumb up, not an L.
    narrow = hand(thumb=True, index=True, thumb_splay_deg=-70.0)
    assert spread_angle(narrow, "thumb", "index") == pytest.approx(40.0)
    assert classify_landmarks(narrow) == POINT_UP
    assert classify_landmarks(narrow, l_shape_tolerance_deg=55.0) == L_SHAPE


def test_l_shape_needs_the_other_fingers_curled():
    # Same thumb/index corner but with the middle finger up => not an L.
    p = hand(thumb=True, index=True, middle=True, thumb_splay_deg=-120.0)
    assert classify_landmarks(p) == VICTORY


def test_pinch_touching_tips_beats_open_palm():
    # All five digits measure "extended", but the touching thumb/index tips
    # must classify as PINCH — the rule runs before the OPEN_PALM check.
    assert classify_landmarks(pinch_hand(touching=True)) == PINCH


def test_spread_thumb_and_index_is_not_a_pinch():
    # Same finger states with the tips far apart => an ordinary open palm.
    assert classify_landmarks(pinch_hand(touching=False)) == OPEN_PALM


def test_too_few_landmarks_is_unknown():
    assert classify_landmarks([(0, 0)] * 5) == UNKNOWN
    assert classify_landmarks(None) == UNKNOWN


def pointing_pinch(touching=True):
    """Natural pointer-mode click: index extended (pointing), other fingers folded,
    thumb tip brought to (or away from) the index tip. Crucially this is NOT the
    classifier's PINCH (which needs middle/ring/pinky extended) — it's the pose a
    user actually makes to click while pointing."""
    p = hand(index=True)          # index extended, middle/ring/pinky folded
    target = (p[8][0] + 0.02, p[8][1] + 0.01) if touching else (0.20, 0.55)
    return _reach_thumb(p, target)


def test_pointer_click_pinch_fires_from_pointing_pose():
    # The decoupled click detector must fire even though the pose is not PINCH.
    assert classify_landmarks(pointing_pinch(touching=True)) != PINCH
    assert index_thumb_pinch(pointing_pinch(touching=True)) is True
    assert index_thumb_pinch(pointing_pinch(touching=False)) is False


def test_pointer_click_needs_full_landmarks():
    assert index_thumb_pinch(None) is False
    assert index_thumb_pinch([(0, 0)] * 5) is False


def test_stabilizer_fires_once_at_threshold_then_holds():
    s = GestureStabilizer(stability_frames=3, cooldown_frames=5)
    assert s.update(FIST) is None       # 1
    assert s.update(FIST) is None       # 2
    assert s.update(FIST) == FIST       # 3 -> fire
    assert s.update(FIST) is None       # already fired, no repeat
    assert s.update(FIST) is None


def test_stabilizer_resets_after_neutral_pose():
    s = GestureStabilizer(stability_frames=2, cooldown_frames=0)
    assert s.update(THUMBS_UP) is None
    assert s.update(THUMBS_UP) == THUMBS_UP
    # a neutral (unknown) pose clears the latch so the same gesture can re-fire
    assert s.update(UNKNOWN) is None
    assert s.update(THUMBS_UP) is None
    assert s.update(THUMBS_UP) == THUMBS_UP
