"""Pointer-mode math: mapping, smoothing, debounce, pose-hold (pure units)."""

from __future__ import annotations

from vision.pointer import ClickDebouncer, Ema, PoseHold, to_screen


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
