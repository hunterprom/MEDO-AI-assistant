"""Pointer-mode math: pure, stdlib-only, unit-testable.

The gesture engine maps the index-fingertip landmark to screen coordinates
through these helpers (ported from v1 jarvis-web's cursor control: sensitivity
recentered around 0.5, EMA smoothing, 600 ms click debounce). No Win32 here —
vision/winmouse.py owns the OS calls — so tests run anywhere.
"""

from __future__ import annotations

# Pointer-mode actions, decided purely from the recognized pose (+ pinch state)
# so the mapping is unit-testable without a camera. The engine turns these into
# OS calls (vision/winmouse.py).
MOVE = "move"
DRAG = "drag"
SCROLL = "scroll"
ZOOM = "zoom"
RIGHT_CLICK = "right_click"
VOLUME_UP = "volume_up"
VOLUME_DOWN = "volume_down"
# Window/tab navigation. These reach the parts of the desktop the cursor is
# worst at: a taskbar icon is a 40 px target, and tab strips are worse. As
# key chords they need no aiming at all.
NEXT_TAB = "next_tab"
SWITCH_WINDOW = "switch_window"
TASKBAR = "taskbar"
IDLE = "idle"

#: pose name -> action when the thumb/index are NOT pinched.
_POSE_ACTION = {
    "point_up": MOVE,
    "victory": SCROLL,
    "rock": ZOOM,
    "three": RIGHT_CLICK,
    "thumbs_up": VOLUME_UP,
    "pinky_up": VOLUME_DOWN,
    # New strict poses (vision/gestures.py). Both are hard to make by accident
    # — an L needs a real right angle, FOUR needs a tucked thumb with all four
    # fingers straight — which is what makes them safe for actions that move
    # you between windows rather than within one.
    "l_shape": NEXT_TAB,
    "four": TASKBAR,
    # open_palm is deliberately NOT mapped: it is what a hand passes through
    # while opening, so binding it to a window switch would fire it by
    # accident. Assign it in config if you want it.
    "fist": IDLE,
}


#: Actions a pose may be bound to in config (``vision.pointer.pose_actions``).
ACTIONS = frozenset({MOVE, DRAG, SCROLL, ZOOM, RIGHT_CLICK, VOLUME_UP,
                     VOLUME_DOWN, NEXT_TAB, SWITCH_WINDOW, TASKBAR, IDLE})


def build_pose_actions(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """The pose->action map with the user's bindings applied.

    An unknown action name is dropped rather than accepted: a typo in YAML
    should cost you that one binding, not put the engine into a state where a
    gesture dispatches to nothing.
    """
    table = dict(_POSE_ACTION)
    for pose, action in (overrides or {}).items():
        name = str(action).strip().lower()
        if name in ACTIONS:
            table[str(pose).strip().lower()] = name
    return table


def pointer_action(gesture: str, is_pinch: bool,
                   table: dict[str, str] | None = None) -> str:
    """Map a recognized pose to a pointer-mode action.

    A pinch (thumb tip on index tip) always means :data:`DRAG` — pressing/holding
    the left button, which doubles as a click on a quick tap — regardless of the
    classified pose. Otherwise the pose selects the action. Anything unmapped
    (``unknown``, ``open_palm``, transitional poses) defaults to :data:`MOVE`, so
    a briefly misread pointing hand keeps driving the cursor instead of freezing;
    only a ``fist`` idles (it's the deliberate hold-to-exit pose).
    """
    if is_pinch:
        return DRAG
    return (table or _POSE_ACTION).get(gesture, MOVE)


class ScrollAccumulator:
    """Turn continuous hand motion into discrete wheel notches.

    Fed the per-frame vertical delta of the fingertip (normalized, y grows
    downward), it accumulates ``-dy * gain`` — moving the hand *up* scrolls up —
    and emits whole notches, carrying the remainder so slow drifts still register.
    """

    def __init__(self, gain: float) -> None:
        self._gain = gain
        self._acc = 0.0

    def update(self, dy: float) -> int:
        self._acc += -dy * self._gain
        notches = int(self._acc)          # trunc toward zero keeps the sign
        self._acc -= notches
        return notches

    def reset(self) -> None:
        self._acc = 0.0


def to_screen(
    nx: float,
    ny: float,
    sensitivity: float,
    width: int,
    height: int,
    mirror_x: bool = False,
) -> tuple[int, int]:
    """Map normalized hand coords (0..1) to clamped pixel coords.

    Sensitivity recenters around 0.5: at 2.5 the middle ~40% of the camera
    frame sweeps the whole screen, so small hand motions reach the edges.
    ``mirror_x`` flips horizontally for *unmirrored* cameras; frames that are
    already selfie-mirrored (``vision.flip: true``) must not flip again.
    """
    if mirror_x:
        nx = 1.0 - nx
    sx = 0.5 + (nx - 0.5) * sensitivity
    sy = 0.5 + (ny - 0.5) * sensitivity
    sx = min(1.0, max(0.0, sx))
    sy = min(1.0, max(0.0, sy))
    return int(sx * (width - 1)), int(sy * (height - 1))


class Ema:
    """Exponential moving average over (x, y) — damps fingertip jitter."""

    def __init__(self, alpha: float) -> None:
        self._alpha = min(1.0, max(0.01, alpha))
        self._x: float | None = None
        self._y: float | None = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        if self._x is None or self._y is None:
            self._x, self._y = x, y
        else:
            a = self._alpha
            self._x += a * (x - self._x)
            self._y += a * (y - self._y)
        return self._x, self._y

    def reset(self) -> None:
        self._x = self._y = None


class ClickDebouncer:
    """Allow a click at most once every ``interval_ms`` (v1 used 600 ms)."""

    def __init__(self, interval_ms: int) -> None:
        self._interval = max(0, interval_ms) / 1000.0
        self._last = float("-inf")

    def ready(self, now_s: float) -> bool:
        """True (and consume the slot) when the interval has elapsed."""
        if now_s - self._last >= self._interval:
            self._last = now_s
            return True
        return False


class PoseHold:
    """Fire once when a pose holds for ``frames`` consecutive frames.

    Re-arms only after the pose releases, so holding a pinch can't
    machine-gun clicks even after the debounce interval passes.
    """

    def __init__(self, frames: int) -> None:
        self._need = max(1, frames)
        self._count = 0
        self._fired = False

    def update(self, active: bool) -> bool:
        if not active:
            self._count = 0
            self._fired = False
            return False
        self._count += 1
        if self._count >= self._need and not self._fired:
            self._fired = True
            return True
        return False
