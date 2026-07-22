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


# --- cross-platform mouse backend (vision/mouse.py) ---------------------------

MOUSE_SURFACE = (
    "screen_size", "move", "click_left", "click_right", "press_left",
    "release_left", "scroll", "zoom", "volume_up", "volume_down", "play_pause",
)


def test_mouse_dispatcher_exposes_the_full_surface():
    """Pointer mode must resolve a complete backend on THIS platform.

    winmouse (win32), macmouse (darwin), or anymouse (everything else) — the
    engine calls all of these; a missing one would crash mid-gesture.
    """
    from vision import mouse

    for name in MOUSE_SURFACE:
        assert callable(getattr(mouse, name)), f"backend lacks {name}()"


def test_macmouse_drags_with_dragged_events(monkeypatch):
    """While the pinch holds the button, moves must be LeftMouseDragged —
    plain MouseMoved mid-drag makes most apps drop the drag."""
    import sys

    import pytest as _pytest

    if sys.platform != "darwin":
        _pytest.skip("macOS backend")
    from vision import macmouse

    posted: list[int] = []
    monkeypatch.setattr(
        macmouse, "_mouse",
        lambda etype, x, y, button=macmouse._BTN_LEFT: posted.append(etype),
    )
    monkeypatch.setattr(macmouse, "_cursor", lambda: macmouse._CGPoint(10, 10))
    monkeypatch.setattr(macmouse, "_left_down", False)

    macmouse.move(1, 2)
    macmouse.press_left()
    macmouse.move(3, 4)
    macmouse.release_left()
    macmouse.move(5, 6)
    assert posted == [
        macmouse._MOVED, macmouse._LDOWN, macmouse._LDRAG,
        macmouse._LUP, macmouse._MOVED,
    ]


# --- window / tab navigation poses -------------------------------------------
#
# The cursor is worst at exactly the targets these replace: taskbar icons and
# tab strips. Bound to the two strict poses that are hard to make by accident.


def test_new_poses_drive_window_navigation():
    from vision.pointer import NEXT_TAB, TASKBAR, pointer_action

    assert pointer_action("l_shape", False) == NEXT_TAB
    assert pointer_action("four", False) == TASKBAR


def test_open_palm_stays_unbound_by_default():
    """It is what a hand passes through while opening — binding it would fire
    a window switch every time you relax your hand."""
    from vision.pointer import MOVE, pointer_action

    assert pointer_action("open_palm", False) == MOVE


def test_a_pinch_still_wins_over_a_navigation_pose():
    from vision.pointer import DRAG, pointer_action

    assert pointer_action("l_shape", True) == DRAG


def test_build_pose_actions_applies_overrides():
    from vision.pointer import SWITCH_WINDOW, build_pose_actions, pointer_action

    table = build_pose_actions({"open_palm": "switch_window"})
    assert pointer_action("open_palm", False, table) == SWITCH_WINDOW
    # The defaults survive alongside the override.
    assert pointer_action("victory", False, table) == "scroll"


def test_build_pose_actions_drops_a_typo_instead_of_binding_it():
    from vision.pointer import MOVE, build_pose_actions, pointer_action

    table = build_pose_actions({"four": "taskbarr", "victory": ""})
    assert "taskbarr" not in table.values()
    # The bad binding is dropped, so the pose keeps its default meaning.
    assert pointer_action("four", False, table) == "taskbar"
    assert pointer_action("nothing_like_this", False, table) == MOVE


def test_every_backend_exposes_the_navigation_surface():
    """vision/mouse.py is a star-import facade: a backend missing one of these
    only fails at the moment a user makes the gesture."""
    import re
    from pathlib import Path

    required = {"next_tab", "prev_tab", "switch_window", "taskbar", "show_desktop"}
    for name in ("winmouse", "macmouse", "anymouse"):
        source = Path(f"vision/{name}.py").read_text(encoding="utf-8")
        defined = set(re.findall(r"^def (\w+)", source, re.MULTILINE))
        assert required <= defined, f"{name} is missing {sorted(required - defined)}"
