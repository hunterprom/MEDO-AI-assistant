"""Vision sidecar — run in the vision venv (MediaPipe needs numpy<2).

    .venv-vision/bin/python -m vision.run

Watches the webcam for hand gestures and, on each confirmed gesture, POSTs the
mapped utterance to the companion API (``POST /ask``) — so gestures reach MEDO
through the very same Intent Router as voice and text. Also serves an MJPEG stream
of the annotated camera so the HUD can show what the camera sees.

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


def load_config(path: Path) -> tuple[VisionRunConfig, str]:
    """Read the vision + remote sections from config.yaml. Returns (config, api_url)."""
    import yaml

    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    v = data.get("vision", {}) or {}
    r = data.get("remote", {}) or {}
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
    )
    host = r.get("host", "127.0.0.1")
    host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    api_url = f"http://{host}:{r.get('port', 8710)}"
    return cfg, api_url


def _make_video_handler(engine: GestureEngine):
    """An HTTP handler that streams the engine's latest frame as MJPEG."""
    import time

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence default per-request logging
            pass

        def do_GET(self):
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
    logger.info("gestures → %s/ask  |  camera stream → http://127.0.0.1:%d/video",
                api_url, cfg.stream_port)
    logger.info("gesture map: %s", cfg.gestures)

    try:
        engine._thread.join()  # type: ignore[union-attr]
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
