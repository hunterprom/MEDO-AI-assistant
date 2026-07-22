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
    # THREE: index+middle+ring up, pinky folded, THUMB TUCKED. The signature
    # table specifies every digit, so a thumb sticking out is no longer three —
    # it matches nothing, rather than silently resolving to the nearest pose.
    (dict(index=True, middle=True, ring=True), THREE),
    (dict(thumb=True, index=True, middle=True, ring=True), UNKNOWN),
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
    # The confidence floor has to come down with it: a finger only just inside a
    # widened band is, by definition, barely holding the state — which is
    # exactly what confidence measures.
    assert classify_landmarks(closing_fist(), extended_min_deg=110.0,
                              min_confidence=0.0) == FOUR


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
    """Same thumb/index corner with the middle finger up is not an L.

    Nor is it a victory sign: that signature tucks the thumb. A pose matching
    no signature is UNKNOWN — the table has no "nearest gesture" fallback, and
    that absence is the feature.
    """
    p = hand(thumb=True, index=True, middle=True, thumb_splay_deg=-120.0)
    assert classify_landmarks(p) == UNKNOWN
    # Tuck the thumb and it becomes a clean victory.
    assert classify_landmarks(hand(index=True, middle=True)) == VICTORY


def test_pinch_needs_the_other_fingers_curled():
    """PINCH is thumb and index together with the hand closed.

    It used to require middle/ring/pinky EXTENDED — an OK-sign, not a pinch —
    which meant the natural pinching pose classified as something else.
    """
    # Tips touching but every finger still out: an OK-sign, not a pinch.
    assert classify_landmarks(pinch_hand(touching=True)) == OPEN_PALM
    # Tips touching with the hand closed IS the pinch.
    assert classify_landmarks(pointing_pinch(touching=True)) == PINCH


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
    """The click detector is independent of classification.

    It measures only the thumb-to-index gap, so it fires from the natural
    pointing-and-pinching pose whatever the table calls that hand — pointer
    mode must not depend on a gesture name resolving.
    """
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


# --- exact signatures, confidence, debounce, hysteresis ------------------------
#
# The A2 contract: a gesture is an exact per-finger signature. One finger in the
# wrong state is not "close enough", a pose held loosely is not confident, and a
# pose held steady is ONE event, not a stream of them.


def test_every_signature_has_a_complete_finger_spec():
    from vision.gestures import ANY, CURLED, EXTENDED, FINGERS, SIGNATURES

    assert SIGNATURES, "the table is the only definition of a gesture"
    for sig in SIGNATURES:
        states = sig.states()
        assert set(states) == set(FINGERS), sig.name
        assert all(v in (EXTENDED, CURLED, ANY) for v in states.values()), sig.name
        assert sig.doc, f"{sig.name} has no description"


def test_signature_names_are_unique():
    from vision.gestures import SIGNATURES

    names = [s.name for s in SIGNATURES]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("builder,expected", [
    (dict(index=True, middle=True, ring=True), THREE),
    (dict(index=True, middle=True, ring=True, pinky=True), FOUR),
    (dict(index=True, middle=True), VICTORY),
    (dict(index=True, pinky=True), ROCK),
    (dict(thumb=True, index=True, middle=True, ring=True, pinky=True), OPEN_PALM),
    (dict(), FIST),
])
def test_the_exact_signature_fires(builder, expected):
    assert classify_landmarks(hand(**builder)) == expected


@pytest.mark.parametrize("builder", [
    dict(index=True, middle=True, ring=True, thumb=True),   # three + thumb out
    dict(index=True, middle=True, pinky=True),              # victory + pinky
    dict(middle=True, ring=True, pinky=True),               # four minus the index
    dict(middle=True, pinky=True),                          # wrong two fingers
    dict(ring=True),                                        # a lone ring finger
    dict(thumb=True, middle=True),                          # thumb + middle only
])
def test_one_finger_off_does_not_fire(builder):
    """No nearest-match fallback: a pose that is not a signature is nothing."""
    assert classify_landmarks(hand(**builder)) == UNKNOWN


def test_sub_threshold_confidence_does_not_fire():
    """Right shape, fingers barely inside their bands -> not confident enough."""
    from vision.gestures import classify_with_confidence

    # Every finger sits 2 degrees inside its band: the shape is a clean THREE.
    marginal = hand(index=True, middle=True, ring=True,
                    bend={"index": 162.0, "middle": 162.0, "ring": 162.0,
                          "pinky": 98.0, "thumb": 98.0})
    name, confidence = classify_with_confidence(marginal, min_confidence=0.0)
    assert name == THREE and confidence < 0.2, "should read as a weak match"
    # With the shipped floor it is rejected outright.
    assert classify_landmarks(marginal) == UNKNOWN
    # Lower the floor and the same hand is accepted — the knob works.
    assert classify_landmarks(marginal, min_confidence=0.05) == THREE


def test_confidence_is_the_weakest_required_finger():
    from vision.gestures import CURLED, EXTENDED, finger_confidence

    assert finger_confidence(180.0, EXTENDED, 160.0, 100.0) == 1.0
    assert finger_confidence(160.0, EXTENDED, 160.0, 100.0) == 0.0
    assert finger_confidence(40.0, CURLED, 160.0, 100.0) == 1.0    # a real curl
    assert finger_confidence(100.0, CURLED, 160.0, 100.0) == 0.0


# --- debounce ------------------------------------------------------------------


def test_two_frames_do_not_fire_but_three_do():
    stabilizer = GestureStabilizer(stability_frames=3, cooldown_frames=0)
    assert stabilizer.update(FIST) is None          # 1
    assert stabilizer.update(FIST) is None          # 2
    assert stabilizer.update(FIST) == FIST          # 3


def test_a_broken_run_restarts_the_count():
    """One misread frame in a stream must not accumulate toward a fire."""
    stabilizer = GestureStabilizer(stability_frames=3, cooldown_frames=0)
    stabilizer.update(FIST)
    stabilizer.update(UNKNOWN)                      # hand in transit
    assert stabilizer.update(FIST) is None
    assert stabilizer.update(FIST) is None
    assert stabilizer.update(FIST) == FIST


# --- hysteresis ----------------------------------------------------------------


def test_a_held_gesture_fires_once_not_repeatedly():
    stabilizer = GestureStabilizer(stability_frames=2, cooldown_frames=0)
    fired = [stabilizer.update(VICTORY) for _ in range(12)]
    assert fired.count(VICTORY) == 1, "holding a pose is one event"


def test_it_re_arms_only_after_the_pose_is_released():
    stabilizer = GestureStabilizer(stability_frames=2, cooldown_frames=0)
    assert [stabilizer.update(VICTORY) for _ in range(4)].count(VICTORY) == 1
    stabilizer.update(UNKNOWN)                      # hand relaxes: re-armed
    assert [stabilizer.update(VICTORY) for _ in range(2)].count(VICTORY) == 1


def test_switching_straight_to_another_gesture_releases_the_first():
    stabilizer = GestureStabilizer(stability_frames=2, cooldown_frames=0)
    [stabilizer.update(VICTORY) for _ in range(2)]
    assert [stabilizer.update(FIST) for _ in range(2)].count(FIST) == 1


def test_reset_clears_the_latch():
    stabilizer = GestureStabilizer(stability_frames=2, cooldown_frames=0)
    [stabilizer.update(FIST) for _ in range(2)]
    stabilizer.reset()
    assert [stabilizer.update(FIST) for _ in range(2)].count(FIST) == 1


# --- every gesture is reachable ------------------------------------------------


def test_every_defined_gesture_is_dispatchable():
    """The "not all gestures apply" bug: defined in the table, bound to nothing.

    Anything reported UNBOUND classifies perfectly and then does nothing, which
    is indistinguishable from broken recognition when you are waving at a camera.
    """
    from vision.gestures import dispatch_report
    from vision.run import DEFAULT_GESTURES

    unbound = [name for name, where in dispatch_report(utterances=DEFAULT_GESTURES)
               if where == "UNBOUND"]
    assert not unbound, f"defined but unreachable: {unbound}"


def test_the_list_cli_reports_success_when_everything_is_bound():
    from vision.gestures import _main

    assert _main(["--list"]) == 0


def test_two_hand_zoom_needs_both_hands():
    from vision.gestures import ZOOM, classify_pair

    one = hand(thumb=True, index=True, thumb_splay_deg=-150.0)
    assert classify_pair([one])[0] == UNKNOWN, "one hand is not a two-hand gesture"
    name, _confidence = classify_pair([one, one])
    assert name == ZOOM


def test_two_hand_zoom_rejects_a_mismatched_pair():
    from vision.gestures import classify_pair

    zoom_hand = hand(thumb=True, index=True, thumb_splay_deg=-150.0)
    other = hand(index=True, middle=True)
    assert classify_pair([zoom_hand, other])[0] == UNKNOWN
