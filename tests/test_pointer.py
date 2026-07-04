"""Pointer-mode math: mapping, smoothing, debounce, pose-hold (pure units)."""

from __future__ import annotations

from vision.pointer import (
    DRAG,
    IDLE,
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


def test_center_maps_to_screen_center():
    assert to_screen(0.5, 0.5, 2.5, 1920, 1080) == (959, 539)


def test_sensitivity_recenters_and_clamps():
    # 0.3 with sensitivity 2.5 → 0.5 + (-0.2 * 2.5) = 0.0 → left edge
    x, y = to_screen(0.3, 0.9, 2.5, 1920, 1080)
    assert x == 0
    assert y == 1079  # clamped to the bottom edge


def test_mirror_only_when_requested():
    # Mirrored (selfie) frames pass mirror_x=False: x used as-is.
    plain, _ = to_screen(0.3, 0.5, 1.0, 1920, 1080, mirror_x=False)
    flipped, _ = to_screen(0.3, 0.5, 1.0, 1920, 1080, mirror_x=True)
    assert plain == int(0.3 * 1919)
    assert flipped == int(0.7 * 1919)


def test_ema_first_sample_passthrough_then_smooths():
    e = Ema(0.5)
    assert e.update(10, 10) == (10, 10)
    assert e.update(20, 20) == (15.0, 15.0)
    e.reset()
    assert e.update(100, 100) == (100, 100)  # no lerp across a reset


def test_click_debouncer_interval():
    d = ClickDebouncer(600)
    assert d.ready(0.0) is True
    assert d.ready(0.3) is False
    assert d.ready(0.61) is True


def test_pose_hold_fires_once_and_rearms_on_release():
    h = PoseHold(3)
    assert [h.update(True) for _ in range(5)] == [False, False, True, False, False]
    assert h.update(False) is False
    assert [h.update(True) for _ in range(3)] == [False, False, True]


def test_pointer_action_pinch_always_drags():
    # A pinch means drag/click no matter what pose the classifier reports.
    for g in ("point_up", "victory", "rock", "three", "unknown"):
        assert pointer_action(g, True) == DRAG


def test_pointer_action_pose_map():
    assert pointer_action("point_up", False) == MOVE
    assert pointer_action("victory", False) == SCROLL
    assert pointer_action("rock", False) == ZOOM
    assert pointer_action("three", False) == RIGHT_CLICK
    assert pointer_action("thumbs_up", False) == VOLUME_UP
    assert pointer_action("pinky_up", False) == VOLUME_DOWN
    # Only a fist idles (deliberate hold-to-exit pose); ambiguous poses keep
    # moving the cursor so a misread pointing hand doesn't freeze mid-move.
    assert pointer_action("fist", False) == IDLE
    assert pointer_action("unknown", False) == MOVE
    assert pointer_action("open_palm", False) == MOVE


def test_scroll_accumulator_direction_and_carry():
    acc = ScrollAccumulator(gain=50.0)
    # Hand moving up (dy negative) scrolls up (positive notches).
    assert acc.update(-0.02) == 1          # 0.02*50 = 1.0
    # Sub-notch motion carries until it crosses a whole notch.
    assert acc.update(-0.01) == 0          # 0.5 accumulated
    assert acc.update(-0.01) == 1          # now 1.0
    # Hand moving down scrolls down (negative notches).
    acc.reset()
    assert acc.update(0.04) == -2
