"""Desktop presence overlay — a small CORE sphere pinned to a screen corner.

MEDO already hears you everywhere (the wake word runs with the mic open however
the screen looks), but there was no way to SEE that while you're in another app.
This is that: a frameless, always-on-top mini version of the CORE sphere that
lives in a corner, shows what MEDO is doing (idle / listening / thinking /
speaking), and briefly bubbles up what you said and what it answered.

    single click  -> start listening now (same as saying the wake word)
    double click  -> open the full HUD
    right click   -> menu (open HUD / hide)
    drag          -> move it; the position is remembered

Rendering: Tk has no per-pixel alpha, so each frame is composed with numpy +
Pillow (additive glow, exactly the cosmic-web the HUD's CORE canvas draws — a
bright centre node, shell nodes, filaments between them, three spinning rings)
and blitted into a window whose key colour is punched out with
``-transparentcolor``. Both libraries are already dependencies; nothing new to
install, which matters on a nearly-full disk.

Runs as its own process (``python -m ui.overlay``) so Tk owns a real main loop
and a crash here can never take the assistant down. It talks to the running
MEDO over HTTP only: the HUD's SSE ``/events`` for state, the companion API's
``/wake`` and ``/interrupt`` for control. Nothing reachable => it just sits
dim and keeps retrying.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import queue
import threading
import urllib.error
import urllib.request

import numpy as np

logger = logging.getLogger("overlay")

#: Colour per assistant state (r, g, b) — the HUD's accent family.
STATE_COLOUR: dict[str, tuple[int, int, int]] = {
    "idle": (56, 168, 255),        # #38a8ff, the HUD accent
    "listening": (127, 208, 255),  # #7fd0ff, brighter — it's hearing you
    "thinking": (86, 140, 255),
    "speaking": (125, 255, 176),   # #7dffb0, the HUD's "active" green
    "offline": (70, 90, 110),      # can't reach MEDO
}
#: The window's chroma key. Unlit sphere pixels are rendered as EXACTLY this so
#: Windows punches them out — anything brighter (the glow, rings, bubble) stays.
_KEY_LEVEL = 1
KEY_COLOUR = "#010101"             # == (_KEY_LEVEL,)*3
BUBBLE_BG = "#0a1420"


# --------------------------------------------------------------------------
# geometry: the same cosmic web the CORE canvas builds, scaled down
# --------------------------------------------------------------------------

def build_web(seed: int = 7, n_nodes: int = 34, filament_steps: int = 13):
    """Nodes inside a unit sphere + filament points linking near neighbours.

    Mirrors ui/web/index.html's initOrb(): one bright centre node, the rest on
    shells, then short curved filaments between each node and its two nearest
    neighbours. Pure + deterministic so frames are stable across restarts.
    """
    rng = np.random.default_rng(seed)

    def in_sphere(rad):
        while True:
            p = rng.uniform(-1, 1, 3)
            if float(p @ p) <= 1.0:
                return p * rad

    nodes = [np.zeros(3)]
    sizes = [2.6]
    for _ in range(n_nodes):
        shell = rng.random() ** 0.6
        nodes.append(in_sphere(0.25 + shell * 0.75))
        sizes.append(0.7 + rng.random() * 1.3)
    nodes_arr = np.asarray(nodes, dtype=np.float32)

    fil: list[np.ndarray] = []
    for i in range(1, len(nodes_arr)):
        a = nodes_arr[i]
        d = ((nodes_arr - a) ** 2).sum(axis=1)
        d[i] = 9.0
        for j in np.argsort(d)[:2]:
            b = nodes_arr[j]
            bend = rng.uniform(-0.22, 0.22, 3)
            for s in range(1, filament_steps):
                u = s / filament_steps
                w = math.sin(u * math.pi)
                fil.append(a + (b - a) * u + bend * w)
    return nodes_arr, np.asarray(sizes, np.float32), np.asarray(fil, np.float32)


@functools.lru_cache(maxsize=64)
def _gaussian_sprite(radius: int) -> np.ndarray:
    """A small radial falloff used as the additive glow stamp.

    Cached: the same handful of radii are stamped every frame (and the haze one
    is large), so regenerating them 40x a frame was most of the render cost.
    """
    span = np.arange(-radius, radius + 1, dtype=np.float32)
    gx, gy = np.meshgrid(span, span)
    d = np.sqrt(gx * gx + gy * gy) / max(radius, 1)
    return np.clip(1.0 - d, 0.0, 1.0) ** 2.0


def _stamp(acc: np.ndarray, sprite: np.ndarray, cx: int, cy: int, amount: float) -> None:
    """Additively blend ``sprite * amount`` into ``acc`` at (cx, cy), clipped."""
    if amount <= 0.0:
        return
    r = sprite.shape[0] // 2
    h, w = acc.shape[:2]
    x0, y0, x1, y1 = cx - r, cy - r, cx + r + 1, cy + r + 1
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    acc[y0:y1, x0:x1] += sprite[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)] * amount


def render_sphere(geo, size: int, t: float, colour: tuple[int, int, int],
                  glow: float = 1.0, supersample: int = 2):
    """One frame of the mini CORE sphere as a PIL RGB image on ``KEY_COLOUR``.

    Drawn at ``supersample``x and downscaled, which is what gives the filaments
    and rings clean edges at 100-odd pixels.
    """
    from PIL import Image, ImageDraw

    nodes, sizes, fil = geo
    S = size * supersample
    cx = cy = S / 2.0
    radius = S * 0.34
    acc = np.zeros((S, S), dtype=np.float32)

    ay, tilt = t * 0.5, 0.42
    cos_y, sin_y = math.cos(ay), math.sin(ay)
    cos_x, sin_x = math.cos(tilt), math.sin(tilt)

    def project(p):
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        x2, z2 = x * cos_y + z * sin_y, -x * sin_y + z * cos_y      # yaw
        y2, z3 = y * cos_x - z2 * sin_x, y * sin_x + z2 * cos_x     # pitch
        return x2, y2, z3

    # Core bloom only — NO wide volume haze. On the HUD a haze reads as glow
    # against a dark page, but floating over an arbitrary desktop it becomes an
    # opaque dark disc smudging whatever is behind it. Only luminous pixels earn
    # their opacity here; everything else stays keyed out.
    _stamp(acc, _gaussian_sprite(int(radius * 0.34)), int(cx), int(cy), 0.85 * glow)

    # filaments: depth-shaded gas between the nodes
    fx, fy, fz = project(fil)
    fsprite = _gaussian_sprite(max(1, int(1.9 * supersample)))
    depth = (fz + 1.0) / 2.0
    for i in range(fil.shape[0]):
        _stamp(acc, fsprite, int(cx + fx[i] * radius), int(cy + fy[i] * radius),
               (0.22 + 0.70 * float(depth[i])) * glow)

    # nodes: brighter, size-scaled; index 0 is the bright centre core
    nx, ny, nz = project(nodes)
    ndepth = (nz + 1.0) / 2.0
    for i in range(nodes.shape[0]):
        r = max(2, int(sizes[i] * supersample * 2.1))
        _stamp(acc, _gaussian_sprite(r),
               int(cx + nx[i] * radius), int(cy + ny[i] * radius),
               (0.60 + 1.30 * float(ndepth[i])) * glow)

    # Extra gain: additive glow that looks vivid on the HUD's near-black page
    # washes out against a mid-grey desktop, and only pixels BRIGHTER than the
    # backdrop read at all. Push the luminance so the sphere holds up on light
    # and dark wallpaper alike.
    acc = np.clip(acc * 1.65, 0.0, 1.85)
    r, g, b = colour
    rgb = np.zeros((S, S, 3), dtype=np.float32)
    rgb[..., 0] = acc * r
    rgb[..., 1] = acc * g
    rgb[..., 2] = acc * b
    # Composite onto the KEY colour exactly: Tk's -transparentcolor punches out
    # pixels that MATCH it, so unlit pixels must BE the key (1,1,1) — leaving
    # them pure black is what put an opaque square around the sphere.
    rgb = np.clip(rgb, 0, 255 - _KEY_LEVEL) + _KEY_LEVEL
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")

    # the three CORE rings (dashed slow / arc / counter-arc)
    draw = ImageDraw.Draw(img)
    ring = (int(r * 0.85), int(g * 0.85), int(b * 0.85))
    # Widths are +1 past the supersample: a 1-px ring blends toward the black
    # key on downscale and all but vanishes over a light desktop.
    lw = supersample + 1
    box0 = [cx - radius * 1.28, cy - radius * 1.28, cx + radius * 1.28, cy + radius * 1.28]
    for k in range(24):                                    # dashed outer ring
        a0 = math.degrees(t * 0.12) + k * 15
        draw.arc(box0, a0, a0 + 8, fill=ring, width=lw)
    box1 = [cx - radius * 1.14, cy - radius * 1.14, cx + radius * 1.14, cy + radius * 1.14]
    a1 = math.degrees(t * 0.9)
    draw.arc(box1, a1, a1 + 110, fill=(r, g, b), width=lw)
    box2 = [cx - radius * 1.05, cy - radius * 1.05, cx + radius * 1.05, cy + radius * 1.05]
    a2 = -math.degrees(t * 1.3)
    draw.arc(box2, a2, a2 + 80, fill=ring, width=lw)

    if supersample > 1:
        img = img.resize((size, size), Image.LANCZOS)
    return img


# --------------------------------------------------------------------------
# live link to the running assistant
# --------------------------------------------------------------------------

class EventLink:
    """Background reader of the HUD's SSE /events, plus control POSTs.

    Never raises into the UI: when MEDO isn't up it reports ``offline`` and
    keeps retrying, so the sphere can sit on the desktop before/after a run.
    """

    def __init__(self, hud_url: str, api_url: str, out: queue.Queue) -> None:
        self._hud, self._api, self._out = hud_url.rstrip("/"), api_url.rstrip("/"), out
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                req = urllib.request.Request(f"{self._hud}/events",
                                             headers={"Accept": "text/event-stream"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    self._out.put(("link", True))
                    for raw in resp:
                        if self._stop.is_set():
                            return
                        line = raw.decode("utf-8", "replace").strip()
                        if line.startswith("data:"):
                            self._emit(line[5:].strip())
            except Exception:
                self._out.put(("link", False))
                self._stop.wait(3.0)          # MEDO not up yet — retry quietly

    def _emit(self, payload: str) -> None:
        try:
            data = json.loads(payload)
        except ValueError:
            return
        kind = data.get("type")
        if kind == "state":
            self._out.put(("state", str(data.get("state") or "idle")))
        elif kind == "transcript":
            self._out.put(("said", str(data.get("text") or "")))
        elif kind == "routed":
            self._out.put(("reply", str(data.get("speech") or data.get("text") or "")))
        elif kind == "hello":
            self._out.put(("state", "idle"))

    def post(self, path: str) -> None:
        """Fire-and-forget control call (``/wake``, ``/interrupt``)."""
        def _go():
            try:
                req = urllib.request.Request(f"{self._api}{path}", data=b"{}",
                                             headers={"Content-Type": "application/json"},
                                             method="POST")
                urllib.request.urlopen(req, timeout=5).read()
            except Exception as exc:
                logger.debug("control %s failed: %s", path, exc)
        threading.Thread(target=_go, daemon=True).start()


# --------------------------------------------------------------------------
# the window
# --------------------------------------------------------------------------

class Overlay:
    def __init__(self, size: int, corner: str, margin: int, fps: int,
                 hud_url: str, api_url: str, demo: bool = False) -> None:
        import tkinter as tk

        self._tk = tk
        self._size, self._fps = size, max(4, min(fps, 30))
        self._events: queue.Queue = queue.Queue()
        self._link = EventLink(hud_url, api_url, self._events)
        self._hud_url = hud_url
        self._geo = build_web()
        self._state, self._online = "idle", False
        self._t = 0.0
        self._bubble_until = 0.0
        self._click_job = None

        self.root = tk.Tk()
        self.root.title("MEDO")
        self.root.overrideredirect(True)                 # frameless
        self.root.attributes("-topmost", True)           # above other apps
        try:
            self.root.attributes("-transparentcolor", KEY_COLOUR)
        except Exception:                                # non-Windows: solid bg
            logger.info("transparent windows unsupported here; using a solid backdrop")
        self.root.configure(bg=KEY_COLOUR)

        self._canvas = tk.Label(self.root, bg=KEY_COLOUR, bd=0, highlightthickness=0)
        self._canvas.pack()
        self._bubble = tk.Label(
            self.root, text="", bg=BUBBLE_BG, fg="#cfe8ff", bd=0,
            font=("Consolas", 8), justify="left", wraplength=size + 60,
            padx=8, pady=5)

        self._place(corner, margin)
        self._bind(self._canvas)
        self._bind(self._bubble)
        if demo:
            self._demo_feed()          # preview the look with no MEDO running
        else:
            self._link.start()
        self.root.after(0, self._tick)

    def _demo_feed(self, step: int = 0) -> None:
        """Cycle a fake exchange so the look can be previewed/screenshotted."""
        script = [
            ("link", True), ("state", "listening"), ("said", "what's the weather"),
            ("state", "thinking"), ("state", "speaking"),
            ("reply", "It's 21 degrees and clear in Skopje right now."),
            ("state", "idle"),
        ]
        self._events.put(script[step % len(script)])
        self.root.after(2200, lambda: self._demo_feed(step + 1))

    # -- placement + dragging ------------------------------------------------

    def _place(self, corner: str, margin: int) -> None:
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w = self._size + 60
        h = self._size + 70
        x = margin if "left" in corner else sw - w - margin
        y = margin if "top" in corner else sh - h - margin
        self.root.geometry(f"+{max(0, x)}+{max(0, y)}")

    def _bind(self, widget) -> None:
        widget.bind("<Button-1>", self._press)
        widget.bind("<B1-Motion>", self._drag)
        widget.bind("<ButtonRelease-1>", self._release)
        widget.bind("<Double-Button-1>", self._double)
        widget.bind("<Button-3>", self._menu)

    def _press(self, ev) -> None:
        self._drag_from = (ev.x_root, ev.y_root)
        self._moved = False

    def _drag(self, ev) -> None:
        dx = ev.x_root - self._drag_from[0]
        dy = ev.y_root - self._drag_from[1]
        if abs(dx) > 3 or abs(dy) > 3:
            self._moved = True
            self.root.geometry(f"+{self.root.winfo_x() + dx}+{self.root.winfo_y() + dy}")
            self._drag_from = (ev.x_root, ev.y_root)

    def _release(self, ev) -> None:
        if self._moved:
            return
        # Delay the single-click action so a double-click can cancel it.
        self._click_job = self.root.after(260, self._single)

    def _single(self) -> None:
        self._click_job = None
        # Speaking? a click interrupts, exactly like the HUD mic button.
        self._link.post("/interrupt" if self._state == "speaking" else "/wake")
        self._show_bubble("listening…" if self._state != "speaking" else "stopped")

    def _double(self, _ev=None) -> None:
        if self._click_job is not None:
            self.root.after_cancel(self._click_job)
            self._click_job = None
        import webbrowser
        webbrowser.open(self._hud_url)

    def _menu(self, ev) -> None:
        m = self._tk.Menu(self.root, tearoff=0)
        m.add_command(label="Open HUD", command=self._double)
        m.add_command(label="Hide", command=self.root.destroy)
        m.tk_popup(ev.x_root, ev.y_root)

    # -- bubble + frame loop -------------------------------------------------

    def _show_bubble(self, text: str, seconds: float = 6.0) -> None:
        text = (text or "").strip()
        if not text:
            return
        if len(text) > 160:
            text = text[:157] + "…"
        self._bubble.configure(text=text)
        self._bubble.pack()
        self._bubble_until = self._t + seconds

    def _drain(self) -> None:
        while True:
            try:
                kind, value = self._events.get_nowait()
            except queue.Empty:
                return
            if kind == "state":
                self._state = value
            elif kind == "link":
                self._online = bool(value)
            elif kind == "said":
                self._show_bubble(f"you: {value}")
            elif kind == "reply":
                self._show_bubble(value, seconds=9.0)

    def _tick(self) -> None:
        self._drain()
        step = 1.0 / self._fps
        self._t += step
        if self._bubble_until and self._t > self._bubble_until:
            self._bubble.pack_forget()
            self._bubble_until = 0.0

        state = self._state if self._online else "offline"
        colour = STATE_COLOUR.get(state, STATE_COLOUR["idle"])
        # a slow breath at rest; a quicker, deeper pulse while it listens/speaks
        beat = 1.6 if state in ("listening", "speaking") else 0.5
        depth = 0.22 if state in ("listening", "speaking") else 0.10
        glow = (0.55 if state == "offline" else 1.0) * (
            1.0 + depth * math.sin(self._t * beat * math.pi))

        from PIL import ImageTk

        img = render_sphere(self._geo, self._size, self._t, colour, glow)
        self._photo = ImageTk.PhotoImage(img)        # keep a ref or Tk drops it
        self._canvas.configure(image=self._photo)
        self.root.after(int(1000 / self._fps), self._tick)

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            self._link.stop()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="MEDO desktop presence sphere")
    ap.add_argument("--size", type=int, default=0, help="sphere px (0 = config)")
    ap.add_argument("--corner", default="", help="bottom-right|bottom-left|top-right|top-left")
    ap.add_argument("--demo", action="store_true",
                    help="cycle a fake exchange to preview the look (no MEDO needed)")
    args = ap.parse_args()

    from core.config import load_settings

    s = load_settings()
    cfg = s.ui.overlay
    hud_url = f"http://127.0.0.1:{s.hud.port}"
    api_url = f"http://127.0.0.1:{s.remote.port}"
    Overlay(size=args.size or cfg.size,
            corner=args.corner or cfg.corner,
            margin=cfg.margin, fps=cfg.fps,
            hud_url=hud_url, api_url=api_url, demo=args.demo).run()


if __name__ == "__main__":
    main()
