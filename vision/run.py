"""Vision sidecar — run in the vision venv (MediaPipe needs numpy<2).

    .venv-vision/bin/python -m vision.run

Watches the webcam for hand gestures and, on each confirmed gesture, POSTs the
mapped utterance to the companion API (``POST /ask``) — so gestures reach MEDO
through the very same Intent Router as voice and text. Its little HTTP server
(:``stream_port``) offers:

    GET  /video      MJPEG stream of the annotated camera (HUD embeds this)
    GET  /frame.jpg  latest single frame (the vision skill sends it to moondream)
    GET  /pointer    {"ok": true, "on": bool} — pointer-mode state
    POST /pointer    {"on": bool} — toggle gesture mouse control

Deliberately dependency-light: stdlib + cv2/mediapipe/numpy (+ PyYAML for config).
It never imports the main app's pydantic/voice stack, keeping the two venvs apart.
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from vision.engine import GestureEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s vision | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("vision.run")


#: Default gesture -> utterance map; config.yaml ``vision.gestures`` entries
#: override per key (so a partial yaml map keeps these defaults for the rest).
DEFAULT_GESTURES: dict[str, str] = {
    "thumbs_up": "yes",
    "open_palm": "no",
    "victory": "take a screenshot",
    "point_up": "volume up",
    "fist": "mute",
    "three": "volume down",
    "pinch": "lock the screen",
    "rock": "play music",
}


@dataclass
class PointerRunConfig:
    """Gesture mouse control knobs (mirrors core.config.PointerConfig)."""

    enabled: bool = True
    sensitivity: float = 2.5
    ema_alpha: float = 0.4
    click_debounce_ms: int = 600
    hold_frames: int = 3
    scroll_gain: float = 45.0
    zoom_gain: float = 25.0
    volume_interval_ms: int = 180


@dataclass
class VisionRunConfig:
    """Plain config for the sidecar (mirrors core.config.VisionConfig fields)."""

    camera_index: int = 0
    stream_port: int = 8731
    flip: bool = True
    max_fps: int = 15
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    stability_frames: int = 6
    cooldown_s: float = 2.0
    gestures: dict = field(default_factory=lambda: dict(DEFAULT_GESTURES))
    pointer: PointerRunConfig = field(default_factory=PointerRunConfig)


def load_config(path: Path) -> tuple[VisionRunConfig, str]:
    """Read the vision + remote sections from config.yaml. Returns (config, api_url)."""
    import yaml

    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    v = data.get("vision", {}) or {}
    r = data.get("remote", {}) or {}
    p = v.get("pointer", {}) or {}
    cfg = VisionRunConfig(
        camera_index=v.get("camera_index", 0),
        stream_port=v.get("stream_port", 8731),
        flip=v.get("flip", True),
        max_fps=v.get("max_fps", 15),
        min_detection_confidence=v.get("min_detection_confidence", 0.6),
        min_tracking_confidence=v.get("min_tracking_confidence", 0.5),
        stability_frames=v.get("stability_frames", 6),
        cooldown_s=v.get("cooldown_s", 2.0),
        gestures={**DEFAULT_GESTURES, **(v.get("gestures", {}) or {})},
        pointer=PointerRunConfig(
            enabled=bool(p.get("enabled", True)),
            sensitivity=float(p.get("sensitivity", 2.5)),
            ema_alpha=float(p.get("ema_alpha", 0.4)),
            click_debounce_ms=int(p.get("click_debounce_ms", 600)),
            hold_frames=int(p.get("hold_frames", 3)),
            scroll_gain=float(p.get("scroll_gain", 45.0)),
            zoom_gain=float(p.get("zoom_gain", 25.0)),
            volume_interval_ms=int(p.get("volume_interval_ms", 180)),
        ),
    )
    host = r.get("host", "127.0.0.1")
    host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    api_url = f"http://{host}:{r.get('port', 8710)}"
    return cfg, api_url


def _make_video_handler(engine: GestureEngine):
    """An HTTP handler: MJPEG stream, latest-frame JPEG, pointer toggle."""
    import time

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence default per-request logging
            pass

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/frame.jpg":
                jpeg = engine.latest_jpeg()
                if not jpeg:
                    self._send_json(503, {"ok": False, "error": "no frame yet"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if self.path == "/pointer":
                self._send_json(200, {"ok": True, "on": engine.pointer_on()})
                return
            if self.path not in ("/video", "/"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    jpeg = engine.latest_jpeg()
                    if jpeg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    time.sleep(1 / 15)
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_POST(self):
            if self.path != "/pointer":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                data = json.loads(self.rfile.read(length) or b"{}")
                want = bool(data.get("on"))
            except (ValueError, TypeError):
                self._send_json(
                    400, {"ok": False, "error": 'body must be JSON like {"on": true}'}
                )
                return
            state = engine.set_pointer(want)
            payload: dict = {"ok": state == want, "on": state}
            if want and not state:
                payload["error"] = "pointer mode is disabled in config or unavailable"
            self._send_json(200, payload)

    return Handler


def make_gesture_poster(api_url: str):
    """A gesture handler that POSTs the mapped utterance to the companion API."""

    def handler(gesture: str, utterance: str) -> None:
        payload = json.dumps({"text": utterance}).encode()
        req = urllib.request.Request(
            f"{api_url}/ask", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                reply = json.loads(resp.read()).get("speech", "")
                logger.info("%s → %r → %s", gesture, utterance, reply)
        except urllib.error.URLError as exc:
            logger.warning("could not reach MEDO at %s (%s). Is it running with --serve?",
                           api_url, exc)

    return handler


def main() -> None:
    parser = argparse.ArgumentParser(description="MEDO vision sidecar (hand gestures)")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--api", default=None, help="override companion API URL")
    args = parser.parse_args()

    cfg, api_url = load_config(Path(args.config))
    if args.api:
        api_url = args.api.rstrip("/")

    engine = GestureEngine(cfg, make_gesture_poster(api_url))
    engine.start()

    server = ThreadingHTTPServer(("0.0.0.0", cfg.stream_port), _make_video_handler(engine))
    Thread(target=server.serve_forever, daemon=True).start()
    logger.info("pointer-only build | camera stream → http://127.0.0.1:%d/video | "
                "pointer toggle → POST http://127.0.0.1:%d/pointer | companion API %s",
                cfg.stream_port, cfg.stream_port, api_url)
    logger.info("controls: index tip = cursor, thumb+index pinch = left click, "
                "three fingers = right click, fist = exit pointer (boots OFF)")

    try:
        engine._thread.join()  # type: ignore[union-attr]
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
