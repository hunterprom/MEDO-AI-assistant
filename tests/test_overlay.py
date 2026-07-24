"""Desktop presence sphere (ui/overlay.py) — the parts that run without a screen.

Tk and PIL are imported lazily inside the window/render paths, so the geometry,
the frame composition and the event parsing are all testable headlessly.
"""

from __future__ import annotations

import queue

import numpy as np
import pytest

from ui.overlay import (
    KEY_COLOUR,
    STATE_COLOUR,
    EventLink,
    _KEY_LEVEL,
    _stamp,
    build_web,
    render_sphere,
)


# --- geometry ----------------------------------------------------------------

def test_build_web_is_deterministic_and_bounded():
    nodes_a, sizes_a, fil_a = build_web(seed=3)
    nodes_b, _, fil_b = build_web(seed=3)
    assert np.array_equal(nodes_a, nodes_b) and np.array_equal(fil_a, fil_b)
    # the centre node is the core, and every node sits inside the unit sphere
    assert np.allclose(nodes_a[0], 0.0)
    assert np.all(np.linalg.norm(nodes_a, axis=1) <= 1.001)
    assert len(sizes_a) == len(nodes_a) and fil_a.shape[1] == 3


def test_different_seeds_differ():
    assert not np.array_equal(build_web(seed=1)[0], build_web(seed=2)[0])


# --- the additive stamp ------------------------------------------------------

def test_stamp_adds_and_clips_at_the_edges():
    acc = np.zeros((10, 10), dtype=np.float32)
    sprite = np.ones((5, 5), dtype=np.float32)
    _stamp(acc, sprite, 0, 0, 1.0)          # mostly off-canvas, must not raise
    assert acc[0, 0] == pytest.approx(1.0)
    before = acc.sum()
    _stamp(acc, sprite, 100, 100, 1.0)      # fully outside: no-op
    assert acc.sum() == pytest.approx(before)
    _stamp(acc, sprite, 5, 5, 0.0)          # zero amount: no-op
    assert acc.sum() == pytest.approx(before)


# --- frame composition -------------------------------------------------------

def _corner_pixel(img):
    return img.getpixel((0, 0))


def test_render_is_square_and_keyed_out_at_the_corners():
    geo = build_web(seed=5)
    img = render_sphere(geo, 96, t=0.0, colour=STATE_COLOUR["idle"])
    assert img.size == (96, 96)
    # Unlit pixels MUST equal the chroma key, or Windows can't punch them out
    # and the sphere gets an opaque black box around it.
    assert _corner_pixel(img) == (_KEY_LEVEL, _KEY_LEVEL, _KEY_LEVEL)
    assert KEY_COLOUR == "#%02x%02x%02x" % ((_KEY_LEVEL,) * 3)


def test_centre_is_lit_and_glow_scales_brightness():
    geo = build_web(seed=5)
    dim = render_sphere(geo, 96, t=0.0, colour=STATE_COLOUR["idle"], glow=0.4)
    bright = render_sphere(geo, 96, t=0.0, colour=STATE_COLOUR["idle"], glow=1.4)
    c = (48, 48)
    assert sum(bright.getpixel(c)) > sum(dim.getpixel(c)) > _KEY_LEVEL * 3


def test_every_state_has_a_colour():
    for state in ("idle", "listening", "thinking", "speaking", "offline"):
        assert state in STATE_COLOUR


# --- event parsing -----------------------------------------------------------

def _link():
    q: queue.Queue = queue.Queue()
    return EventLink("http://x", "http://y", q), q


def test_emit_maps_hud_events():
    link, q = _link()
    link._emit('{"type": "state", "state": "listening"}')
    link._emit('{"type": "transcript", "text": "hello there"}')
    link._emit('{"type": "routed", "speech": "Hi."}')
    link._emit('{"type": "hello", "name": "MEDO"}')
    assert q.get_nowait() == ("state", "listening")
    assert q.get_nowait() == ("said", "hello there")
    assert q.get_nowait() == ("reply", "Hi.")
    assert q.get_nowait() == ("state", "idle")


def test_emit_ignores_junk_and_unknown_types():
    link, q = _link()
    link._emit("not json at all")
    link._emit('{"type": "something-else"}')
    assert q.empty()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
