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
from voice.wakeword import display_phrase as _wake_display
from core.events import Event, EventBus, EventType
from core.router import RouteResult

logger = logging.getLogger(__name__)

_WEB_DIR = Path(__file__).resolve().parent / "web"


class HudServer:
    """Serves the HUD page and an SSE event feed off the EventBus."""

    def __init__(self, settings: Settings, bus: EventBus,
                 registry=None) -> None:
        self._settings = settings
        # The live skill registry, so the cluster-sphere view is built from
        # what is actually registered — not a hardcoded picture. None (tests /
        # HUD-only launches) simply yields an empty graph.
        self._registry = registry
        self._runner: web.AppRunner | None = None
        self._clients: set[asyncio.Queue] = set()
        bus.subscribe(EventType.STATE_CHANGED, self._on_state)
        bus.subscribe(EventType.TRANSCRIPT, self._on_transcript)
        bus.subscribe(EventType.ROUTED, self._on_routed)
        bus.subscribe(EventType.CAPTION, self._on_caption)

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
            # Companion-API auth token. Needed only when this page is opened
            # from another device (localhost requests are exempt); the HUD is
            # already the trust boundary — it serves folder paths and config.
            "apiToken": self._settings.remote.token,
            "wakePhrase": _wake_display(self._settings.wakeword.phrase),
            # Interface language default; the picker persists its own choice
            # in localStorage, so this only seeds a browser that has none.
            "uiLanguage": self._settings.hud.language,
            # Look: base theme + composable effect layers + cluster-view
            # quality. Cosmetic only; the browser persists its own choice.
            "ui": {
                "theme": self._settings.ui.theme,
                "effects": list(self._settings.ui.effects),
                "spheres": {
                    "enabled": self._settings.ui.spheres.enabled,
                    "quality": self._settings.ui.spheres.quality,
                },
            },
            # The agent cluster graph (spheres + skill-stars), built from the
            # live registry so it stays truthful as skills are added.
            "agents": self._agent_graph(),
            "weather": {
                "city": self._settings.weather.default_city,
                "lat": self._settings.weather.latitude,
                "lon": self._settings.weather.longitude,
            },
            # Folder shortcuts scattered across the orb dots (centre = system
            # drive; hundreds of real directories, only curated ones are labelled).
            # to_thread: this walks the filesystem — never block the event loop.
            "dirs": await asyncio.to_thread(
                dirs.sphere_dirs, self._settings.hud.max_dir_dots
            ),
        }
        html = html.replace("__MEDO_CONFIG__", json.dumps(page_config))
        return web.Response(text=html, content_type="text/html")

    def _agent_graph(self) -> dict:
        """The cluster-view graph, or an empty one when there's no registry.

        Built from the *enabled* council roster so a specialist switched off in
        config doesn't appear as a star it can never light.
        """
        if self._registry is None:
            return {"spheres": [], "skillDomain": {}}
        try:
            from core.agents import agent_graph
            from core.council import enabled_council, load_council

            council = enabled_council(
                load_council(self._settings.council.extra),
                self._settings.council.disabled,
            )
            return agent_graph(self._registry, council)
        except Exception:                       # never let the map break the page
            logger.warning("could not build the agent graph", exc_info=True)
            return {"spheres": [], "skillDomain": {}}

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

    def _on_caption(self, event: Event) -> None:
        # Interpreter mode: {src, src_text, dst, dst_text} — the HUD shows the
        # heard line and its translation as a live caption.
        self._broadcast({"type": "caption", **event.payload})

    def _on_routed(self, event: Event) -> None:
        result: RouteResult = event.payload
        self._broadcast({
            "type": "routed",
            "path": result.path.value,
            "skill": result.skill_name,
            "speech": result.speech,
            "latency_ms": round(result.latency_ms, 0),
        })
