"""The one process a user runs. Wires the pure Supervisor to real subprocesses,
a system-tray home (Open / Settings / Quit), and opening the HUD in the browser.

Kept thin: the testable command-building + child assembly is here as small pure
functions; the tray + watch loop are integration glue (no unit tests — they
touch real processes and the desktop). The dev workflow (run.bat) is untouched.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import List

from app.childspec import ChildSpec, is_frozen
from app.supervisor import Supervisor

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _exe(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def _venv_python(venv: str) -> str:
    root = _project_root()
    if os.name == "nt":
        return str(root / venv / "Scripts" / "python.exe")
    return str(root / venv / "bin" / "python")


# -- how to launch each child (dev checkout vs frozen bundle) -----------------

def ollama_command() -> List[str]:
    # Same on both: the daemon is an external install (the wizard, S3, installs it
    # if missing). We only START it when it isn't already reachable.
    return ["ollama", "serve"]


def engine_command() -> List[str]:
    if is_frozen():
        return [str(Path(sys.executable).parent / _exe("medo-engine"))]
    return [_venv_python(".venv"), str(_project_root() / "main.py"),
            "--voice", "--hud", "--serve"]


def vision_command() -> List[str]:
    # The vision sidecar lives in its OWN environment (mediapipe pins numpy<2), so
    # it is frozen as a SEPARATE bundle — never merged into the engine.
    if is_frozen():
        return [str(Path(sys.executable).parent / _exe("medo-vision"))]
    return [_venv_python(".venv-vision"), "-m", "vision.run"]


def _http_ok(url: str, timeout: float = 1.5) -> bool:
    try:
        import httpx
        return httpx.get(url, timeout=timeout).status_code < 500
    except Exception:
        return False


def build_children(settings) -> List[ChildSpec]:
    """The three OS processes MEDO owns, with health probes wired to config ports."""
    llm_host = str(getattr(settings.llm, "host", "http://127.0.0.1:11434")).rstrip("/")
    hud_port = int(getattr(settings.hud, "port", 8730))
    vision_port = int(getattr(settings.vision, "stream_port", 8731))

    def ollama_up() -> bool:
        return _http_ok(f"{llm_host}/api/tags")

    return [
        ChildSpec("ollama", ollama_command(), friendly_name="the local AI engine",
                  health=ollama_up, start_if=lambda: not ollama_up()),
        ChildSpec("engine", engine_command(), friendly_name="MEDO",
                  health=lambda: _http_ok(f"http://127.0.0.1:{hud_port}/")),
        ChildSpec("vision", vision_command(), friendly_name="hand gestures",
                  health=lambda: _http_ok(f"http://127.0.0.1:{vision_port}/status")),
    ]


def hud_url(settings) -> str:
    return f"http://localhost:{int(getattr(settings.hud, 'port', 8730))}/"


# -- real process spawner ------------------------------------------------------

def _spawn(cmd: List[str]) -> subprocess.Popen:
    kwargs: dict = {}
    if os.name == "nt":
        # Own process group so a wedged child TREE can be killed as a unit
        # (the current run.bat's orphan-leak comes from not doing this).
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


# -- integration glue (not unit-tested) ---------------------------------------

def _open_hud_when_ready(sup: Supervisor, settings, stop: threading.Event) -> None:
    url = hud_url(settings)
    deadline = time.monotonic() + 90.0
    while not stop.is_set() and time.monotonic() < deadline:
        if sup.all_ready():
            try:
                webbrowser.open(url)
            except Exception:
                logger.warning("couldn't open the browser to %s", url, exc_info=True)
            return
        stop.wait(1.0)


def _supervise_loop(sup: Supervisor, stop: threading.Event) -> None:
    while not stop.is_set():
        sup.supervise_once()
        stop.wait(1.5)


def _notify_error(spec: ChildSpec, message: str) -> None:
    # Plain-language surface. The tray (if present) shows a balloon; either way it
    # goes to the log. Never a traceback in front of the user.
    logger.error("child failure surfaced to user: %s", message)
    _TRAY_NOTIFY[0] and _TRAY_NOTIFY[0](spec.label(), message)


_TRAY_NOTIFY: list = [None]      # set by the tray once it's up


def run() -> None:  # pragma: no cover - desktop integration
    logging.basicConfig(level=logging.INFO)
    from core.config import load_settings

    settings = load_settings()
    children = build_children(settings)
    sup = Supervisor(children, spawn=_spawn, now=time.monotonic,
                     on_error=_notify_error)
    stop = threading.Event()

    def shutdown(*_a) -> None:
        stop.set()

    atexit.register(lambda: sup.stop_all())
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, shutdown)
        except (ValueError, OSError):
            pass

    sup.start_all()
    threading.Thread(target=_supervise_loop, args=(sup, stop), daemon=True).start()
    threading.Thread(target=_open_hud_when_ready, args=(sup, settings, stop),
                     daemon=True).start()

    icon = _make_tray(sup, settings, stop)
    try:
        if icon is not None:
            icon.run()               # blocks on the tray until Quit
        else:
            stop.wait()              # headless fallback (no pystray installed)
    finally:
        stop.set()
        sup.stop_all()


def _make_tray(sup: Supervisor, settings, stop: threading.Event):  # pragma: no cover
    try:
        import pystray
        from PIL import Image, ImageDraw
    except Exception:
        logger.warning("pystray/Pillow not available — running without a tray icon")
        return None

    img = Image.new("RGB", (64, 64), (13, 17, 23))
    ImageDraw.Draw(img).ellipse((16, 16, 48, 48), fill=(56, 189, 248))

    def _open(icon, _item):
        webbrowser.open(hud_url(settings))

    def _settings(icon, _item):
        webbrowser.open(hud_url(settings) + "#settings")

    def _quit(icon, _item):
        stop.set()
        icon.stop()

    icon = pystray.Icon("MEDO", img, "MEDO", menu=pystray.Menu(
        pystray.MenuItem("Open MEDO", _open, default=True),
        pystray.MenuItem("Settings", _settings),
        pystray.MenuItem("Quit", _quit),
    ))
    _TRAY_NOTIFY[0] = lambda title, msg: icon.notify(msg, title)
    return icon


if __name__ == "__main__":  # pragma: no cover
    run()
