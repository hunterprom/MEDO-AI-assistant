"""M.E.D.O. web HUD — an arc-reactor front end served over the LAN.

A tiny aiohttp server that renders `ui/web/index.html` and streams live events
(state transitions, transcript, routing path, spoken reply) to the browser over
Server-Sent Events. It subscribes to the same :class:`~core.events.EventBus` the
rest of the app uses, so the HUD is a pure observer — no component depends on it.

If the vision sidecar is running, the page embeds its MJPEG camera stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from aiohttp import web

from core import dirs
from core.config import Settings
from core.events import Event, EventBus, EventType
from core.router import RouteResult

logger = logging.getLogger(__name__)

_WEB_DIR = Path(__file__).resolve().parent / "web"


class HudServer:
    """Serves the HUD page and an SSE event feed off the EventBus."""

    def __init__(self, settings: Settings, bus: EventBus) -> None:
        self._settings = settings
        self._runner: web.AppRunner | None = None
        self._clients: set[asyncio.Queue] = set()
        bus.subscribe(EventType.STATE_CHANGED, self._on_state)
        bus.subscribe(EventType.TRANSCRIPT, self._on_transcript)
        bus.subscribe(EventType.ROUTED, self._on_routed)

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/events", self._events)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        host, port = self._settings.hud.host, self._settings.hud.port
        await web.TCPSite(self._runner, host, port).start()
        logger.info("HUD at http://%s:%d", "localhost" if host in ("0.0.0.0", "") else host, port)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    # -- routes -------------------------------------------------------------

    async def _index(self, request: web.Request) -> web.Response:
        html = (_WEB_DIR / "index.html").read_text(encoding="utf-8")
        page_config = {
            "name": self._settings.personality.name,
            "visionEnabled": self._settings.vision.enabled,
            "streamPort": self._settings.vision.stream_port,
            "apiPort": self._settings.remote.port,
            "wakePhrase": self._settings.wakeword.phrase.replace("_", " "),
            "weather": {
                "city": self._settings.weather.default_city,
                "lat": self._settings.weather.latitude,
                "lon": self._settings.weather.longitude,
            },
            # Folder shortcuts scattered across the orb dots (centre = system
            # drive; ~100+ real directories, only the curated ones are labelled).
            "dirs": dirs.sphere_dirs(),
        }
        html = html.replace("__MEDO_CONFIG__", json.dumps(page_config))
        return web.Response(text=html, content_type="text/html")

    async def _events(self, request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        })
        await resp.prepare(request)
        queue: asyncio.Queue = asyncio.Queue()
        self._clients.add(queue)
        try:
            await self._send(resp, {"type": "hello", "name": self._settings.personality.name})
            while True:
                data = await queue.get()
                await self._send(resp, data)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self._clients.discard(queue)
        return resp

    @staticmethod
    async def _send(resp: web.StreamResponse, data: dict) -> None:
        await resp.write(f"data: {json.dumps(data)}\n\n".encode())

    # -- event bus subscribers (broadcast to SSE clients) -------------------

    def _broadcast(self, data: dict) -> None:
        for queue in list(self._clients):
            queue.put_nowait(data)

    def _on_state(self, event: Event) -> None:
        _old, new = event.payload
        self._broadcast({"type": "state", "state": new.value})

    def _on_transcript(self, event: Event) -> None:
        self._broadcast({"type": "transcript", "text": event.payload})

    def _on_routed(self, event: Event) -> None:
        result: RouteResult = event.payload
        self._broadcast({
            "type": "routed",
            "path": result.path.value,
            "skill": result.skill_name,
            "speech": result.speech,
            "latency_ms": round(result.latency_ms, 0),
        })
