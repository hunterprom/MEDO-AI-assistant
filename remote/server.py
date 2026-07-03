"""HTTP endpoint for companion clients (the Wear OS watch app, the HUD, etc.).

A deliberately tiny, LAN-only API. Clients send plain text; MEDO routes it
through the same Intent Router the REPL uses and returns the spoken reply.
Model/provider endpoints let clients switch the LLM at runtime.

    GET  /ping            -> {"ok": true, "name": "MEDO", "service": "jarvis-v2"}
    POST /ask {"text": …} -> {"speech": …, "path": "FAST"|"LLM",
                              "skill": str|null, "latency_ms": float}
    GET  /status          -> {"ok", "name", "provider", "model", "models", "state"}
    GET  /sys             -> cached CPU/RAM/GPU telemetry (2 s refresh, HUD poll)
    GET  /models          -> {"models": [...]}
    POST /model {"name": …}          -> {"ok", "model"}
    POST /provider {"provider": "ollama"|"openai", "api_key"?, "base_url"?,
                    "model"?}        -> {"ok", "provider", "model", "models"}

Every response carries permissive CORS headers because the HUD (served on its
own port) calls this API cross-origin from the browser. The API key accepted by
``POST /provider`` is stored in settings but never echoed back or logged.

No auth by design: the assistant is 100 % local and the server binds to the
local network. Do not expose this port beyond your LAN.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import psutil
from aiohttp import web

from core.config import Settings
from core.events import AssistantState, StateMachine
from core.router import Router

logger = logging.getLogger(__name__)

#: Reject absurdly long utterances before they reach the LLM.
MAX_TEXT_CHARS = 2000

#: Permissive CORS for the browser HUD; the API is LAN-only anyway.
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
}

#: Providers accepted by ``POST /provider`` (mirrors LLMConfig.provider).
VALID_PROVIDERS = ("ollama", "openai")


@web.middleware
async def cors_middleware(request: web.Request, handler) -> web.StreamResponse:
    """Attach CORS headers to every response, including error responses."""
    try:
        response = await handler(request)
    except web.HTTPException as exc:
        exc.headers.update(CORS_HEADERS)
        raise
    response.headers.update(CORS_HEADERS)
    return response


class RemoteServer:
    """Serves the companion API on top of an existing :class:`Router`."""

    def __init__(self, settings: Settings, router: Router, sm: StateMachine) -> None:
        self._settings = settings
        self._router = router
        self._sm = sm
        self._runner: web.AppRunner | None = None
        # /sys cache — refreshed by a background task so a 2 s HUD poll costs
        # a dict lookup, not a psutil/nvidia-smi round-trip per request.
        self._sys: dict = {"cpu": None, "ram": None, "gpu": None}
        self._sys_task: asyncio.Task | None = None
        self._no_nvidia = False

    def build_app(self) -> web.Application:
        """Create the aiohttp application (separated out for tests)."""
        app = web.Application(middlewares=[cors_middleware])
        app.router.add_get("/ping", self._handle_ping)
        app.router.add_post("/ask", self._handle_ask)
        app.router.add_get("/status", self._handle_status)
        app.router.add_get("/sys", self._handle_sys)
        app.router.add_get("/models", self._handle_models)
        app.router.add_post("/model", self._handle_set_model)
        app.router.add_post("/provider", self._handle_set_provider)
        # CORS preflight for any path (the HUD's fetch() sends OPTIONS first).
        app.router.add_route("OPTIONS", "/{tail:.*}", self._handle_options)
        return app

    async def start(self) -> None:
        """Bind and start serving; returns once the socket is listening."""
        host = self._settings.remote.host
        port = self._settings.remote.port
        self._runner = web.AppRunner(self.build_app())
        await self._runner.setup()
        await web.TCPSite(self._runner, host, port).start()
        self._sys_task = asyncio.create_task(self._collect_sys_forever())
        logger.info("remote API listening on http://%s:%d", host, port)

    async def stop(self) -> None:
        """Shut the server down cleanly (no-op if never started)."""
        if self._sys_task is not None:
            self._sys_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sys_task
            self._sys_task = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    # --- system telemetry (HUD SYSTEM DIAGNOSTICS panel) ---

    async def _collect_sys_forever(self) -> None:
        """Refresh the cached /sys payload every 2 s.

        ``psutil.cpu_percent(None)`` returns 0.0 on its very first call; the
        second tick self-heals. nvidia-smi is skipped permanently after a
        FileNotFoundError so non-NVIDIA machines don't fork a missing binary
        every 2 s.
        """
        while True:
            try:
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                gpu = None if self._no_nvidia else await self._read_gpu()
                self._sys = {
                    "cpu": round(cpu, 1),
                    "ram": {
                        "percent": round(mem.percent, 1),
                        "used_gb": round(mem.used / 1024**3, 1),
                        "total_gb": round(mem.total / 1024**3, 1),
                    },
                    "gpu": gpu,
                }
            except Exception:  # telemetry must never take the server down
                logger.exception("sys telemetry collection failed")
            await asyncio.sleep(2.0)

    async def _read_gpu(self) -> dict | None:
        """One nvidia-smi sample, or None when there's no usable NVIDIA GPU."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError:
            self._no_nvidia = True
            return None
        except OSError:
            return None
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=1.5)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            return None
        parts = [p.strip() for p in out.decode(errors="replace").split(",")]
        if proc.returncode != 0 or len(parts) < 4:
            return None
        try:
            return {
                "util": float(parts[0]),
                "vram_used_mb": float(parts[1]),
                "vram_total_mb": float(parts[2]),
                "temp_c": float(parts[3]),
            }
        except ValueError:
            return None

    async def _handle_sys(self, request: web.Request) -> web.Response:
        """Cached system telemetry — cheap enough for a 2 s HUD poll."""
        return web.json_response({"ok": True, **self._sys})

    # --- handlers ---

    async def _handle_options(self, request: web.Request) -> web.Response:
        """CORS preflight — headers are added by the middleware."""
        return web.Response(status=204)

    async def _handle_ping(self, request: web.Request) -> web.Response:
        """Liveness probe used by the watch app's connection check."""
        return web.json_response(
            {
                "ok": True,
                "name": self._settings.personality.name,
                "service": "jarvis-v2",
            }
        )

    async def _handle_status(self, request: web.Request) -> web.Response:
        """Full picture for UIs: provider, active model, choices, state."""
        models = await asyncio.to_thread(self._router.llm.list_models)
        return web.json_response(
            {
                "ok": True,
                "name": self._settings.personality.name,
                "provider": self._settings.llm.provider,
                "model": self._router.model,
                "models": models,
                "state": self._sm.state.value,
            }
        )

    async def _handle_models(self, request: web.Request) -> web.Response:
        """Models offered by the active provider (empty list when offline)."""
        models = await asyncio.to_thread(self._router.llm.list_models)
        return web.json_response({"models": models})

    async def _handle_set_model(self, request: web.Request) -> web.Response:
        """Switch the active LLM model at runtime."""
        try:
            payload = await request.json()
        except ValueError:
            return _error(400, "body must be JSON like {\"name\": \"…\"}")

        name = str(payload.get("name") or "").strip()
        if not name:
            return _error(400, "missing or empty 'name'")
        self._router.model = name
        logger.info("active model set to %r via companion API", name)
        return web.json_response({"ok": True, "model": name})

    async def _handle_set_provider(self, request: web.Request) -> web.Response:
        """Switch the LLM provider (and optionally key/base URL/model) live.

        Mutates the shared :class:`~core.config.LLMConfig`, which the client
        reads at call time — so the very next chat goes to the new provider.
        The api_key is write-only: stored, never echoed back or logged.
        """
        try:
            payload = await request.json()
        except ValueError:
            return _error(400, "body must be JSON like {\"provider\": \"…\"}")

        provider = str(payload.get("provider") or "").strip().lower()
        if provider not in VALID_PROVIDERS:
            return _error(400, "'provider' must be one of: " + ", ".join(VALID_PROVIDERS))

        llm_cfg = self._settings.llm
        llm_cfg.provider = provider
        api_key = payload.get("api_key")
        if api_key is not None:
            llm_cfg.api_key = str(api_key)
        base_url = str(payload.get("base_url") or "").strip()
        if base_url:
            if provider == "openai":
                llm_cfg.openai_base_url = base_url
            else:
                llm_cfg.host = base_url

        models = await asyncio.to_thread(self._router.llm.list_models)
        model = str(payload.get("model") or "").strip() or (models[0] if models else None)
        self._router.model = model
        logger.info("provider switched to %r (model %r) via companion API", provider, model)
        return web.json_response(
            {"ok": True, "provider": provider, "model": model, "models": models}
        )

    async def _handle_ask(self, request: web.Request) -> web.Response:
        """Route one utterance and return the reply as JSON."""
        try:
            payload = await request.json()
        except ValueError:
            return _error(400, "body must be JSON like {\"text\": \"…\"}")

        text = str(payload.get("text", "")).strip()
        if not text:
            return _error(400, "missing or empty 'text'")
        if len(text) > MAX_TEXT_CHARS:
            return _error(413, f"'text' longer than {MAX_TEXT_CHARS} characters")

        await self._sm.transition(AssistantState.THINKING)
        try:
            result = await self._router.route(text, context={"source": "remote"})
        finally:
            await self._sm.transition(AssistantState.IDLE)

        return web.json_response(
            {
                "speech": result.speech,
                "path": result.path.value,
                "skill": result.skill_name,
                "latency_ms": round(result.latency_ms, 1),
            }
        )


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"ok": False, "error": message}, status=status)
