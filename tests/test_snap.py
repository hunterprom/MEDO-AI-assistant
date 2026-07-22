"""Click snapping: which element wins, and the order things happen in.

The geometry decides where the user's click lands, so it is tested exhaustively
and without a screen. The ordering test is the important one: a snap that
clicks first and shows the target afterwards would be exactly the silent
relocation this feature exists to avoid.
"""

from __future__ import annotations

import pytest

from vision.snap import (
    Clickable,
    SnapResult,
    choose_target,
    distance_to,
    resolve_click,
)


def button(name, left, top, right, bottom, role="Button") -> Clickable:
    return Clickable(name, role, left, top, right, bottom)


class _Provider:
    """Stands in for UI Automation; records what it was asked."""

    def __init__(self, elements=(), raises=False):
        self.elements = list(elements)
        self.raises = raises
        self.asked: list[tuple] = []

    def near(self, x, y, radius_px):
        self.asked.append((x, y, radius_px))
        if self.raises:
            raise RuntimeError("COM went away")
        return self.elements


# --- geometry ------------------------------------------------------------------


def test_centre_distance_not_edge_distance():
    """A huge pane overlapping the circle must not win on edge proximity."""
    pane = button("Pane", 0, 0, 4000, 4000)
    assert distance_to(pane, 100, 100) > 100


def test_an_element_within_the_radius_is_chosen():
    target = button("Save", 460, 380, 560, 420)          # centre (510, 400)
    assert choose_target([target], 500, 450, 100) is target


def test_an_element_outside_the_radius_is_not():
    far = button("Save", 460, 100, 560, 140)             # centre 310 px above
    assert choose_target([far], 500, 450, 100) is None


def test_no_elements_means_no_snap():
    assert choose_target([], 500, 450, 100) is None


def test_a_zero_radius_disables_snapping():
    target = button("Save", 490, 440, 510, 460)
    assert choose_target([target], 500, 450, 0) is None


def test_the_nearest_centre_wins():
    near = button("Near", 480, 420, 520, 440)            # centre (500, 430)
    far = button("Far", 480, 370, 520, 390)              # centre (500, 380)
    assert choose_target([near, far], 500, 450, 100) is near


# --- the downward-drift bias ---------------------------------------------------


def test_above_the_cursor_wins_a_tie():
    """The pinch drags the fingertip down, so an equally-near target above is
    the one that was actually being aimed at."""
    above = button("Above", 480, 400, 520, 420)          # centre 40 px up
    below = button("Below", 480, 480, 520, 500)          # centre 40 px down
    assert choose_target([above, below], 500, 450, 100) is above


def test_the_bias_can_win_against_a_slightly_nearer_target_below():
    above = button("Above", 480, 390, 520, 410)          # 50 px up
    below = button("Below", 480, 480, 520, 500)          # 40 px down
    # Without a bias the lower one is nearer and wins.
    assert choose_target([above, below], 500, 450, 100, above_bias_px=0) is below
    # With one, the upward target is preferred — that is the drift correction.
    assert choose_target([above, below], 500, 450, 100, above_bias_px=25) is above


def test_the_bias_does_not_reach_outside_the_radius():
    """A generous bias must not drag in something the circle excluded."""
    far_above = button("Far", 480, 100, 520, 140)
    assert choose_target([far_above], 500, 450, 100, above_bias_px=500) is None


# --- already on target ---------------------------------------------------------


def test_an_element_under_the_cursor_wins_outright():
    """If you are already on the button there is nothing to correct."""
    under = button("Under", 450, 430, 550, 470)
    tempting = button("Above", 480, 400, 520, 415)
    assert choose_target([under, tempting], 500, 450, 100) is under


def test_the_innermost_control_wins_when_nested():
    outer = button("Toolbar", 400, 400, 600, 500, role="ComboBox")
    inner = button("Bold", 490, 440, 515, 465)
    assert choose_target([outer, inner], 500, 450, 100) is inner


# --- resolve_click: ordering and fallbacks -------------------------------------


def _resolve(provider, **kw):
    order, slept = [], []
    defaults = dict(enabled=True, radius_px=100, above_bias_px=20.0,
                    confirm_ms=250.0, provider=provider,
                    highlight=lambda el: order.append(("highlight", el.name)),
                    sleep=lambda s: slept.append(s) or order.append(("sleep", s)))
    defaults.update(kw)
    result = resolve_click(500, 450, **defaults)
    return result, order, slept


def test_the_target_is_highlighted_and_waited_on_before_the_click():
    """The safety property: preview, pause, THEN move the click.

    A wrong target has to be visible and abortable. Clicking first and drawing
    afterwards would defeat the entire point of the feature.
    """
    target = button("Save", 460, 400, 560, 430)
    result, order, slept = _resolve(_Provider([target]))
    assert result.snapped and result.target is target
    assert order == [("highlight", "Save"), ("sleep", 0.25)]
    assert slept == [0.25]


def test_the_click_lands_on_the_target_centre():
    target = button("Save", 460, 400, 560, 430)
    result, _order, _slept = _resolve(_Provider([target]))
    assert (result.x, result.y) == target.center


def test_a_zero_delay_still_highlights_but_does_not_wait():
    """Power users turn the delay off; they should still see what was hit."""
    target = button("Save", 460, 400, 560, 430)
    result, order, slept = _resolve(_Provider([target]), confirm_ms=0)
    assert result.snapped and slept == []
    assert order == [("highlight", "Save")]


def test_nothing_in_radius_clicks_the_raw_point():
    result, order, _slept = _resolve(_Provider([button("Far", 0, 0, 20, 20)]))
    assert result == SnapResult(500, 450, None, False)
    assert order == [], "nothing should be highlighted when nothing was chosen"


def test_no_accessibility_data_clicks_the_raw_point():
    """The stated non-regression: no provider, no change in behaviour."""
    result, order, _slept = _resolve(None)
    assert result == SnapResult(500, 450, None, False)
    assert order == []


def test_disabled_clicks_the_raw_point_without_asking():
    provider = _Provider([button("Save", 460, 400, 560, 430)])
    result, _order, _slept = _resolve(provider, enabled=False)
    assert result == SnapResult(500, 450, None, False)
    assert provider.asked == [], "a disabled snap must not even query"


def test_a_provider_that_throws_falls_back_to_raw():
    """A click that lands where you pointed always beats a click that doesn't."""
    result, order, _slept = _resolve(_Provider(raises=True))
    assert result == SnapResult(500, 450, None, False)
    assert order == []


def test_a_highlight_that_throws_still_clicks_the_target():
    def boom(_element):
        raise RuntimeError("no desktop DC")

    target = button("Save", 460, 400, 560, 430)
    result, _order, slept = _resolve(_Provider([target]), highlight=boom)
    assert result.snapped and (result.x, result.y) == target.center
    assert slept == [0.25], "the confirm delay must survive a failed highlight"


def test_the_provider_is_asked_about_the_cursor_and_the_configured_radius():
    provider = _Provider([])
    _resolve(provider, radius_px=140)
    assert provider.asked == [(500, 450, 140)]


# --- config plumbing -----------------------------------------------------------


def test_the_snap_settings_exist_and_are_mirrored_to_the_sidecar():
    """The sidecar keeps a hand-written copy of PointerConfig; a field added to
    one and not the other kills the camera at startup."""
    import dataclasses

    from core.config import PointerConfig
    from vision.run import PointerRunConfig

    fields = {"snap_enabled", "snap_radius_px", "snap_confirm_ms",
              "snap_above_bias_px", "snap_highlight"}
    assert fields <= set(PointerConfig.model_fields)
    assert fields <= {f.name for f in dataclasses.fields(PointerRunConfig)}


def test_the_shipped_defaults_match_the_brief():
    from core.config import load_settings

    pointer = load_settings().vision.pointer
    assert pointer.snap_enabled is True
    assert pointer.snap_radius_px == 100
    assert pointer.snap_confirm_ms == 250
