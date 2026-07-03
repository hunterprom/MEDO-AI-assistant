"""Hand-gesture classification and debounce — pure logic, no camera/MediaPipe."""

from __future__ import annotations

import pytest

from vision.gestures import (
    FIST, OPEN_PALM, PINCH, POINT_UP, ROCK, THREE, THUMBS_UP, UNKNOWN, VICTORY,
    GestureStabilizer, classify_landmarks,
)

WRIST_Y = 0.9  # normalized image coords, y grows downward


def hand(thumb=False, index=False, middle=False, ring=False, pinky=False):
    """Build 21 synthetic landmarks with the chosen fingers extended.

    Extended => tip farther from the wrist than its middle joint; folded => closer.
    Only the indices the classifier reads need meaningful values.
    """
    p = [(0.5, WRIST_Y)] * 21
    p[0] = (0.5, WRIST_Y)                          # wrist
    p[2] = (0.5, 0.70)                             # thumb MCP  (dist 0.20)
    p[4] = (0.5, 0.40) if thumb else (0.5, 0.82)   # thumb tip  (0.50 vs 0.08)
    for pip_i, tip_i, ext in [(6, 8, index), (10, 12, middle),
                              (14, 16, ring), (18, 20, pinky)]:
        p[pip_i] = (0.5, 0.60)                      # PIP dist 0.30
        p[tip_i] = (0.5, 0.20) if ext else (0.5, 0.76)  # 0.70 vs 0.14
    return p


def pinch_hand(touching=True):
    """Middle/ring/pinky extended; thumb & index tips touching (or spread apart).

    The pinch threshold is 0.28 * dist(wrist 0, middle MCP 9), so the middle MCP
    must sit away from the wrist (palm length 0.30 -> threshold 0.084 here).
    """
    p = hand(middle=True, ring=True, pinky=True)
    p[9] = (0.5, 0.60)                              # middle MCP: palm length 0.30
    if touching:
        p[4] = (0.50, 0.50)                         # thumb tip ─┐ dist 0.02 < 0.084
        p[8] = (0.52, 0.50)                         # index tip ─┘ (both "extended")
    else:
        p[4] = (0.30, 0.40)                         # spread apart: dist ~0.45
        p[8] = (0.70, 0.20)
    return p


@pytest.mark.parametrize("builder, expected", [
    (dict(thumb=True, index=True, middle=True, ring=True, pinky=True), OPEN_PALM),
    (dict(), FIST),
    (dict(thumb=True), THUMBS_UP),
    (dict(index=True), POINT_UP),
    (dict(index=True, middle=True), VICTORY),
    (dict(ring=True), UNKNOWN),           # a single non-index finger => unknown
    # THREE: index+middle+ring up, pinky folded — thumb must not matter.
    (dict(index=True, middle=True, ring=True), THREE),
    (dict(thumb=True, index=True, middle=True, ring=True), THREE),
    # ROCK: index+pinky up, middle+ring folded — thumb must not matter.
    (dict(index=True, pinky=True), ROCK),
    (dict(thumb=True, index=True, pinky=True), ROCK),
    (dict(middle=True, pinky=True), UNKNOWN),   # wrong two fingers => unknown
])
def test_classify(builder, expected):
    assert classify_landmarks(hand(**builder)) == expected


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
