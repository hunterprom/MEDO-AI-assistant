"""HTTP endpoint for companion clients (the Wear OS watch app, the HUD, etc.).

A deliberately tiny, LAN-only API. Clients send plain text; MEDO routes it
through the same Intent Router the REPL uses and returns the spoken reply.
Model/provider endpoints let clients switch the LLM at runtime.

    GET  /ping            -> {"ok": true, "name": "MEDO", "service": "medo"}
    POST /ask {"text": …} -> {"speech": …, "path": "FAST"|"LLM",
                              "skill": str|null, "latency_ms": float}
    GET  /status          -> {"ok", "name", "provider", "model", "models", "state"}
    GET  /sys             -> cached CPU/RAM/GPU telemetry (2 s refresh, HUD poll)
    GET  /models          -> {"models": [...]}
    POST /model {"name": …}          -> {"ok", "model"}
    POST /provider {"provider": "ollama"|"openai", "api_key"?, "base_url"?,
                    "model"?}        -> {"ok", "provider", "model", "models"}
    GET  /dirs            -> {"dirs": [{"key", "label", "path", "center"}, …]}
    POST /open {"key"|"path": …} -> {"ok", "path"}   (opens a folder/file in explorer)
    POST /wake            -> {"ok": true}      (start a voice turn without the wake word)
    POST /interrupt       -> {"ok": true}      (stop TTS mid-sentence; barge-in)
    GET  /audio/devices   -> {"devices": [{"index", "name"}], "current": …}
    POST /audio/input {"device": int|str|null} -> {"ok", "device"}  (pick the mic)
    GET  /search/files?q= -> {"results": [{"name", "path", "is_dir"}, …]}
    GET  /search/web?q=   -> {"results": [{"title", "url", "snippet"}, …]}
    POST /pair/start      -> {"ok", "expires_in", "name"}   (code shown on THIS pc)
    POST /pair/confirm {"code": …} -> {"ok", "token", "name"}   (watch pairing)

A UDP responder on the same port answers ``MEDO_DISCOVER_V1`` broadcasts with
``{"service": "medo", "name", "port"}`` so the watch finds this machine
without anyone typing an IP (``remote.discovery_enabled``).

Every response carries permissive CORS headers because the HUD (served on its
own port) calls this API cross-origin from the browser. The API key accepted by
``POST /provider`` is stored in settings but never echoed back or logged.

Auth: every endpoint requires ``Authorization: Bearer <token>`` (or a
``?token=`` query parameter for clients that can't set headers). The token is
generated on first serve and stored in the git-ignored secrets.local.yaml.
Requests from 127.0.0.1 are exempt so the HUD and vision sidecar keep their
zero-config startup; LAN clients (the watch app) must present the token.
``remote.auth_enabled: false`` restores the old open behavior (unsafe).
Still LAN-only: there is no TLS — do not forward this port beyond your LAN.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import secrets as secrets_mod
import threading
import time

import psutil
from aiohttp import web

from core import dirs
from core.config import Settings, save_llm_secrets
from core.events import AssistantState, StateMachine
from core.router import Router

logger = logging.getLogger(__name__)

#: Reject absurdly long utterances before they reach the LLM.
MAX_TEXT_CHARS = 2000

#: Permissive CORS for the browser HUD; the API is LAN-only anyway.
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
}

#: Peer addresses that skip token auth (the HUD + sidecar run on this machine).
_LOCAL_PEERS = ("127.0.0.1", "::1", "::ffff:127.0.0.1")

#: Paths reachable WITHOUT a token: the pairing bootstrap. /pair/start only
#: flashes a code on this PC's screen; /pair/confirm needs that code — so
#: neither leaks anything to a LAN peer who can't see the screen.
_AUTH_EXEMPT_PATHS = ("/pair/start", "/pair/confirm")

#: Watch pairing: code lifetime and how many wrong guesses invalidate it.
PAIR_CODE_TTL_S = 120.0
PAIR_MAX_ATTEMPTS = 5

#: UDP discovery probe the watch broadcasts; anything else is ignored.
DISCOVERY_PROBE = b"MEDO_DISCOVER_V1"


class _DiscoveryProtocol(asyncio.DatagramProtocol):
    """Answers "who is MEDO?" UDP broadcasts with our name + API port.

    Deliberately reveals nothing sensitive: the reply carries what a port
    scan of the LAN would find anyway (the service exists, its port, its
    display name) — never the token.
    """

    def __init__(self, name: str, port: int) -> None:
        self._name = name
        self._port = port
        self._transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport) -> None:  # type: ignore[override]
        self._transport = transport

    def datagram_received(self, data: bytes, addr) -> None:
        if data.strip() != DISCOVERY_PROBE or self._transport is None:
            return
        payload = json.dumps(
            {"service": "medo", "name": self._name, "port": self._port}
        ).encode()
        self._transport.sendto(payload, addr)
        logger.info("discovery probe answered for %s", addr[0])

#: Providers accepted by ``POST /provider`` (mirrors LLMConfig.provider).
VALID_PROVIDERS = ("ollama", "openai", "anthropic", "claude-code", "codex")


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

    def __init__(
        self,
        settings: Settings,
        router: Router,
        sm: StateMachine,
        *,
        persist_secrets: bool = False,
        wake_event: threading.Event | None = None,
        doc_index=None,
        mcp_manager=None,
        link=None,
        modes=None,
    ) -> None:
        self._settings = settings
        self._router = router
        self._sm = sm
        # MCP client manager (None when MCP is disabled/unconfigured).
        self._mcp = mcp_manager
        # MEDO Link device registry (link/registry.py); None disables /link/*.
        self._link = link
        # Shared session-mode toggles (continuous / interpreter); None -> a
        # fresh holder so /control/modes and /status still work in tests.
        if modes is None:
            from core.modes import SessionModes

            modes = SessionModes()
        self._modes = modes
        # When True, a /provider switch writes the key/model to the git-ignored
        # secrets file so it survives a restart. Off by default (and in tests).
        self._persist_secrets = persist_secrets
        # Set by POST /wake to break the voice loop out of STANDING BY without
        # the wake word. None when voice mode isn't running.
        self._wake_event = wake_event
        # Documents RAG index (None when embeddings are disabled).
        self._doc_index = doc_index
        self._runner: web.AppRunner | None = None
        # Active watch-pairing session: {"code", "expires", "attempts"}.
        # None when no pairing is in progress.
        self._pair: dict | None = None
        self._udp_transport: asyncio.DatagramTransport | None = None
        # /sys cache — refreshed by a background task so a 2 s HUD poll costs
        # a dict lookup, not a psutil/nvidia-smi round-trip per request.
        self._sys: dict = {"cpu": None, "ram": None, "gpu": None}
        self._sys_task: asyncio.Task | None = None
        self._no_nvidia = False

    def build_app(self) -> web.Application:
        """Create the aiohttp application (separated out for tests)."""

        @web.middleware
        async def auth_middleware(request: web.Request, handler) -> web.StreamResponse:
            if self._authorized(request):
                return await handler(request)
            return _error(401, "missing or invalid token")

        # cors first so even 401s carry the headers the browser HUD needs.
        app = web.Application(middlewares=[cors_middleware, auth_middleware])
        app.router.add_get("/ping", self._handle_ping)
        app.router.add_post("/ask", self._handle_ask)
        app.router.add_get("/status", self._handle_status)
        app.router.add_get("/sys", self._handle_sys)
        app.router.add_get("/models", self._handle_models)
        app.router.add_post("/model", self._handle_set_model)
        app.router.add_post("/provider", self._handle_set_provider)
        app.router.add_get("/dirs", self._handle_dirs)
        app.router.add_post("/open", self._handle_open)
        app.router.add_post("/wake", self._handle_wake)
        app.router.add_post("/interrupt", self._handle_interrupt)
        app.router.add_get("/audio/devices", self._handle_audio_devices)
        app.router.add_post("/audio/input", self._handle_set_audio_input)
        app.router.add_post("/control/languages", self._handle_set_languages)
        app.router.add_get("/search/files", self._handle_search_files)
        app.router.add_get("/search/web", self._handle_search_web)
        app.router.add_get("/docs/stats", self._handle_docs_stats)
        app.router.add_post("/docs/reindex", self._handle_docs_reindex)
        app.router.add_get("/mcp", self._handle_mcp_status)
        app.router.add_post("/control/pc", self._handle_pc_control)
        app.router.add_post("/control/modes", self._handle_modes)
        app.router.add_post("/control/lion", self._handle_lion)
        app.router.add_get("/council", self._handle_council)
        app.router.add_post("/council", self._handle_council_toggle)
        # MEDO Link (M9): manifest-driven device layer. Token-authed like
        # everything else — LAN devices must present the bearer token.
        app.router.add_post("/link/register", self._handle_link_register)
        app.router.add_get("/link/devices", self._handle_link_devices)
        app.router.add_get("/link/commands/{device_id}", self._handle_link_commands)
        app.router.add_post("/link/result", self._handle_link_result)
        app.router.add_get("/link/ws", self._handle_link_ws)
        app.router.add_post("/pair/start", self._handle_pair_start)
        app.router.add_post("/pair/confirm", self._handle_pair_confirm)
        app.router.add_get("/facts", self._handle_facts_list)
        app.router.add_post("/facts", self._handle_facts_add)
        app.router.add_post("/facts/delete", self._handle_facts_delete)
        # CORS preflight for any path (the HUD's fetch() sends OPTIONS first).
        app.router.add_route("OPTIONS", "/{tail:.*}", self._handle_options)
        return app

    # --- auth ---

    def _is_local(self, request: web.Request) -> bool:
        """True when the request comes from this machine (auth-exempt)."""
        return request.remote in _LOCAL_PEERS

    def _authorized(self, request: web.Request) -> bool:
        """Token gate for LAN clients.

        OPTIONS passes because browsers never attach Authorization to a CORS
        preflight — the actual request that follows is still checked. With
        auth on and no token configured, LAN requests are refused (fail
        closed) rather than silently open.
        """
        remote_cfg = self._settings.remote
        if not remote_cfg.auth_enabled or request.method == "OPTIONS":
            return True
        if request.path in _AUTH_EXEMPT_PATHS:  # pairing bootstrap (see above)
            return True
        if self._is_local(request):
            return True
        expected = remote_cfg.token
        if not expected:
            return False
        header = request.headers.get("Authorization", "")
        supplied = header[7:] if header.startswith("Bearer ") else request.query.get("token", "")
        if not supplied:
            return False
        try:
            return hmac.compare_digest(supplied, expected)
        except TypeError:
            # compare_digest rejects non-ASCII str operands; a token with a
            # non-ASCII char in it simply can't match ours — refuse, don't 500.
            return False

    async def start(self) -> None:
        """Bind and start serving; returns once the socket is listening."""
        host = self._settings.remote.host
        port = self._settings.remote.port
        self._runner = web.AppRunner(self.build_app())
        await self._runner.setup()
        await web.TCPSite(self._runner, host, port).start()
        self._sys_task = asyncio.create_task(self._collect_sys_forever())
        logger.info("remote API listening on http://%s:%d", host, port)
        if self._settings.remote.discovery_enabled:
            # Same port number over UDP; bind 0.0.0.0 so broadcasts arrive.
            # Best-effort: discovery failing must never take the API down.
            try:
                loop = asyncio.get_running_loop()
                self._udp_transport, _ = await loop.create_datagram_endpoint(
                    lambda: _DiscoveryProtocol(self._settings.personality.name, port),
                    local_addr=("0.0.0.0", port),
                )
                logger.info("watch discovery answering on udp/%d", port)
            except Exception:
                logger.warning("discovery responder unavailable", exc_info=True)

    async def stop(self) -> None:
        """Shut the server down cleanly (no-op if never started)."""
        if self._udp_transport is not None:
            self._udp_transport.close()
            self._udp_transport = None
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
        except asyncio.CancelledError:
            # Server shutdown cancelled us mid-communicate: reap the subprocess
            # now, or its transport is finalized after the loop closes and
            # spews "Event loop is closed" tracebacks on an otherwise clean exit.
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            raise
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
                "service": "medo",
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
                # Let the HUD show what's remembered. The keys themselves are
                # never returned — only whether one is stored locally.
                "openai_base_url": self._settings.llm.openai_base_url,
                "has_api_key": bool(self._settings.llm.api_key),
                "anthropic_base_url": self._settings.llm.anthropic_base_url,
                "has_anthropic_key": bool(self._settings.llm.anthropic_api_key),
                # The PC CONTROL switch (may MEDO act on this computer?).
                "pc_control": self._settings.safety.pc_control_enabled,
                # Session modes (continuous conversation, interpreter).
                "continuous": self._modes.continuous,
                "interpreter": self._modes.interpreter,
                # MEDO Lion Mode: the defensive-security profile (surfaces
                # read-only audit skills + deep-red skin; safety unchanged).
                "lion_mode": self._settings.mode.lion,
                # Which local CLI agents exist on this machine (for the HUD).
                "cli_available": await asyncio.to_thread(self._cli_availability),
            }
        )

    def _cli_availability(self) -> dict[str, bool]:
        import shutil

        return {
            "claude-code": shutil.which(self._settings.llm.claude_cmd) is not None,
            "codex": shutil.which(self._settings.llm.codex_cmd) is not None,
        }

    # --- watch pairing (auth bootstrap) ---

    async def _handle_pair_start(self, request: web.Request) -> web.Response:
        """Begin pairing: flash a 6-digit code on THIS PC, never in the reply.

        The watch (or any LAN client) calls this, the human reads the code off
        the MEDO console and types it into the watch — proving they're at the
        machine. Restarting pairing invalidates any previous code.
        """
        code = f"{secrets_mod.randbelow(1_000_000):06d}"
        self._pair = {
            "code": code,
            "expires": time.monotonic() + PAIR_CODE_TTL_S,
            "attempts": 0,
        }
        # The code goes to the PC screen only (console; visible in the MEDO
        # window). Kept out of the HTTP response and the shared log line.
        try:
            from rich.console import Console

            Console().print(
                f"\n[bold cyan]⌚ Watch pairing code: {code}[/bold cyan]  "
                f"[dim](valid {int(PAIR_CODE_TTL_S // 60)} min — type it on the watch)[/dim]\n"
            )
        except Exception:  # rich missing/odd console — plain print still works
            print(f"\nWatch pairing code: {code} (valid 2 min)\n")
        logger.info("pairing started by %s — code shown on this PC", request.remote)
        return web.json_response(
            {"ok": True, "expires_in": int(PAIR_CODE_TTL_S), "name": self._settings.personality.name}
        )

    async def _handle_pair_confirm(self, request: web.Request) -> web.Response:
        """Exchange the on-screen code for the API token (attempt-limited)."""
        try:
            payload = await _json_dict(request)
        except ValueError:
            return _error(400, "body must be JSON like {\"code\": \"123456\"}")
        supplied = str(payload.get("code") or "").strip()

        pair = self._pair
        if pair is None:
            return _error(409, "no pairing in progress — start pairing first")
        if time.monotonic() > pair["expires"]:
            self._pair = None
            return _error(410, "pairing code expired — start again")
        pair["attempts"] += 1
        if pair["attempts"] > PAIR_MAX_ATTEMPTS:
            self._pair = None
            logger.warning("pairing aborted: too many wrong codes from %s", request.remote)
            return _error(429, "too many attempts — start pairing again")
        if not supplied or not hmac.compare_digest(supplied, pair["code"]):
            return _error(401, "wrong code")

        self._pair = None  # single use
        token = self._settings.remote.token
        if not token:  # auth disabled or first run — mint one so pairing still works
            from core.config import ensure_remote_token

            token = ensure_remote_token(self._settings)
        logger.info("watch paired from %s", request.remote)
        return web.json_response(
            {"ok": True, "token": token, "name": self._settings.personality.name}
        )

    async def _handle_mcp_status(self, request: web.Request) -> web.Response:
        """Connected MCP servers + their tools (HUD CONFIG tab)."""
        configured = bool(self._settings.mcp.servers)
        servers = self._mcp.status() if self._mcp is not None else []
        return web.json_response(
            {"ok": True, "enabled": self._settings.mcp.enabled,
             "configured": configured, "servers": servers}
        )

    async def _handle_models(self, request: web.Request) -> web.Response:
        """Models offered by the active provider (empty list when offline)."""
        models = await asyncio.to_thread(self._router.llm.list_models)
        return web.json_response({"models": models})

    async def _handle_set_model(self, request: web.Request) -> web.Response:
        """Switch the active LLM model at runtime."""
        try:
            payload = await _json_dict(request)
        except ValueError:
            return _error(400, "body must be JSON like {\"name\": \"…\"}")

        name = str(payload.get("name") or "").strip()
        if not name:
            return _error(400, "missing or empty 'name'")
        self._router.model = name
        self._warm_model(name)
        logger.info("active model set to %r via companion API", name)
        return web.json_response({"ok": True, "model": name})

    def _warm_model(self, model: str | None) -> None:
        """Preload a local model in the background after a switch (no-op online).

        Without this the first question after picking a model pays the full
        cold load (~27 s for the 30B) — which reads as "it doesn't answer".
        """
        warm = getattr(self._router.llm, "warmup", None)
        if warm is not None and model:
            asyncio.create_task(warm(model))

    async def _handle_set_provider(self, request: web.Request) -> web.Response:
        """Switch the LLM provider (and optionally key/base URL/model) live.

        Mutates the shared :class:`~core.config.LLMConfig`, which the client
        reads at call time — so the very next chat goes to the new provider.
        The api_key is write-only: stored, never echoed back or logged.
        """
        try:
            payload = await _json_dict(request)
        except ValueError:
            return _error(400, "body must be JSON like {\"provider\": \"…\"}")

        provider = str(payload.get("provider") or "").strip().lower()
        if provider not in VALID_PROVIDERS:
            return _error(400, "'provider' must be one of: " + ", ".join(VALID_PROVIDERS))

        llm_cfg = self._settings.llm
        llm_cfg.provider = provider
        # The key/base URL land in the field belonging to the chosen provider,
        # so GPT and Claude credentials are remembered independently.
        api_key = payload.get("api_key")
        if api_key is not None:
            if provider == "anthropic":
                llm_cfg.anthropic_api_key = str(api_key)
            else:
                llm_cfg.api_key = str(api_key)
        base_url = str(payload.get("base_url") or "").strip()
        if base_url:
            if provider == "openai":
                llm_cfg.openai_base_url = base_url
            elif provider == "anthropic":
                llm_cfg.anthropic_base_url = base_url
            elif provider == "ollama":
                llm_cfg.host = base_url

        models = await asyncio.to_thread(self._router.llm.list_models)
        model = str(payload.get("model") or "").strip() or (models[0] if models else None)
        self._router.model = model
        self._warm_model(model)  # local switch: load now, not on the first question
        logger.info("provider switched to %r (model %r) via companion API", provider, model)

        if self._persist_secrets:
            # Remember the choice across restarts (git-ignored file). Only write
            # the key when one was supplied this request; leave any stored key
            # untouched otherwise. Merges, so switching to ollama keeps the key.
            save_llm_secrets(
                provider=provider,
                api_key=(llm_cfg.api_key
                         if api_key is not None and provider != "anthropic" else None),
                anthropic_api_key=(llm_cfg.anthropic_api_key
                                   if api_key is not None and provider == "anthropic" else None),
                openai_base_url=(llm_cfg.openai_base_url if base_url and provider == "openai" else None),
                anthropic_base_url=(llm_cfg.anthropic_base_url if base_url and provider == "anthropic" else None),
                default_model=model,
            )

        return web.json_response(
            {"ok": True, "provider": provider, "model": model, "models": models}
        )

    async def _handle_wake(self, request: web.Request) -> web.Response:
        """Start a voice turn without the wake word (HUD "wake" button).

        Frees the assistant from STANDING BY when the pretrained wake phrase
        doesn't match what the user says. 409 if voice mode isn't running.
        """
        if self._wake_event is None:
            return _error(409, "voice mode is not running")
        self._wake_event.set()
        logger.info("manual wake requested via companion API")
        return web.json_response({"ok": True})

    async def _handle_audio_devices(self, request: web.Request) -> web.Response:
        """List input devices (for the HUD mic picker) + the current selection."""
        from voice.audio import list_input_devices

        devices = await asyncio.to_thread(list_input_devices)
        return web.json_response(
            {"ok": True, "devices": devices, "current": self._settings.audio.input_device}
        )

    async def _handle_set_audio_input(self, request: web.Request) -> web.Response:
        """Choose the microphone the wake word listens on, at runtime.

        Accepts an int index, a name substring, or null (system default). The
        voice loop's wake wait polls this setting every frame (~80 ms) and
        reopens the mic itself — no wake/interrupt signal is abused for it, so
        switching mics never cuts a reply or triggers a phantom listen.
        """
        try:
            payload = await _json_dict(request)
        except ValueError:
            return _error(400, "body must be JSON like {\"device\": 4}")

        device = payload.get("device")
        if device is not None and not isinstance(device, (int, str)):
            return _error(400, "'device' must be a number, a name, or null")
        if isinstance(device, str) and not device.strip():
            device = None

        # Persist (and apply) the device NAME, not the raw index: PortAudio
        # re-numbers devices across reboots, so a stored index silently lands
        # on a different microphone next session. (Live-verified: a stored "1"
        # pointed at a virtual cable one boot later.)
        if isinstance(device, int):
            from voice.audio import list_input_devices

            devices = await asyncio.to_thread(list_input_devices)
            name = next((d["name"] for d in devices if d["index"] == device), None)
            if name:
                device = name

        self._settings.audio.input_device = device
        if self._persist_secrets:
            from core.config import save_audio_input

            save_audio_input(device)
        logger.info("audio input device set to %r via companion API", device)
        return web.json_response({"ok": True, "device": device})

    async def _handle_set_languages(self, request: web.Request) -> web.Response:
        """Set the active languages / primary / detection (S4).

        Validated by LanguagesConfig — more than 2 active, an unknown code, or a
        primary outside the active set all return 400 with the reason. The change
        reloads the STT model, so it takes effect on the next restart (the picker
        surfaces ``restart_required``).
        """
        try:
            payload = await _json_dict(request)
        except ValueError:
            return _error(400, "body must be JSON like "
                               "{\"active\": [\"en\", \"mk\"], \"primary\": \"en\"}")
        active = payload.get("active")
        if not isinstance(active, list) or not active:
            return _error(400, "'active' must be a non-empty list of language codes")

        from core.config import LanguagesConfig, save_languages

        try:
            if self._persist_secrets:
                cfg = save_languages(active, payload.get("primary"),
                                     payload.get("detection"))
            else:  # tests: validate without touching the overrides file
                cfg = LanguagesConfig(active=active, primary=payload.get("primary"),
                                      detection=payload.get("detection") or "auto_pair")
        except Exception as exc:
            msg = ("; ".join(e["msg"] for e in exc.errors())
                   if hasattr(exc, "errors") else str(exc))
            return _error(400, (msg or "invalid language selection")[:200])
        logger.info("active languages set to %s via companion API (restart to apply)",
                    cfg.active)
        return web.json_response({"ok": True, "active": list(cfg.active),
                                  "primary": cfg.primary, "detection": cfg.detection,
                                  "restart_required": True})

    async def _handle_dirs(self, request: web.Request) -> web.Response:
        """Folder shortcuts for the HUD sphere dots (labels + resolved paths).

        Rescans this machine live — the HUD's "SCAN THIS PC" button uses it to
        replace beacons from a page rendered elsewhere. Same dot cap as the page.
        """
        return web.json_response({"dirs": await asyncio.to_thread(
            dirs.sphere_dirs, self._settings.hud.max_dir_dots)})

    async def _handle_open(self, request: web.Request) -> web.Response:
        """Open a whitelisted folder shortcut in the OS file explorer.

        ``key`` must name an entry from :func:`core.dirs.user_dirs` — arbitrary
        paths are never opened, so a stray request can't launch anything else.
        """
        try:
            payload = await _json_dict(request)
        except ValueError:
            return _error(400, "body must be JSON like {\"key\": \"downloads\"}")

        key = str(payload.get("key") or "").strip()
        path_str = str(payload.get("path") or "").strip()
        # to_thread: both validators hit the disk (exists / cold whitelist BFS).
        if path_str:  # a search result: validate it's under an allowed root
            path = await asyncio.to_thread(dirs.resolve_path, path_str)
            if path is None:
                return _error(403, "path not allowed or does not exist")
        else:
            path = await asyncio.to_thread(dirs.resolve, key)
            if path is None:
                return _error(404, f"unknown folder shortcut {key!r}")
        opened = await asyncio.to_thread(dirs.open_path, path)
        if not opened:
            return _error(500, f"could not open {path}")
        logger.info("opened %s via companion API", path)
        return web.json_response({"ok": True, "path": str(path)})

    async def _handle_interrupt(self, request: web.Request) -> web.Response:
        """Barge-in: stop any TTS playback now, and (by default) start listening.

        ``sounddevice.stop()`` is global, so it cuts the current Piper utterance
        wherever the voice loop is blocked on playback. ``{"listen": false}`` just
        silences it without opening the mic.
        """
        try:
            import sounddevice as sd

            await asyncio.to_thread(sd.stop)
        except Exception:
            logger.debug("interrupt: could not stop audio", exc_info=True)
        listen = True
        with contextlib.suppress(ValueError):
            listen = bool((await _json_dict(request)).get("listen", True))
        if listen and self._wake_event is not None:
            self._wake_event.set()
        logger.info("interrupt requested (listen=%s)", listen)
        return web.json_response({"ok": True, "listening": listen and self._wake_event is not None})

    async def _handle_pc_control(self, request: web.Request) -> web.Response:
        """The HUD's PC CONTROL switch: may MEDO act on this computer?

        Gates every ``controls_pc`` skill (typing, apps, websites, files,
        power…) at the router. Persisted per machine so the choice survives
        restarts. Sensing skills are never affected.
        """
        body = await _json_dict(request)
        if "on" not in body:
            return _error(400, "body must be JSON like {\"on\": true}")
        enabled = bool(body.get("on"))
        self._settings.safety.pc_control_enabled = enabled
        if self._persist_secrets:
            from core.config import save_pc_control

            save_pc_control(enabled)
        logger.info("PC control switched %s via companion API",
                    "ON" if enabled else "OFF")
        return web.json_response({"ok": True, "on": enabled})

    async def _handle_lion(self, request: web.Request) -> web.Response:
        """MEDO LION MODE (the defensive-security profile) from the HUD.

        Runtime-only on purpose — it resets on restart, because a profile is a
        thing you enter for a task. It changes presentation and which skills
        are surfaced; it does NOT touch any safety rule (see docs/Decisions.md).
        """
        body = await _json_dict(request)
        if "on" not in body:
            return _error(400, "body must be JSON like {\"on\": true}")
        enabled = bool(body.get("on"))
        self._settings.mode.lion = enabled
        logger.info("LION MODE %s via companion API — defensive-security "
                    "profile %s (safety gate unchanged)",
                    "ON" if enabled else "OFF",
                    "surfaced" if enabled else "hidden")
        return web.json_response({"ok": True, "on": enabled})

    async def _handle_council(self, request: web.Request) -> web.Response:
        """The specialist roster and which of them are switched on."""
        from core.council import load_council

        disabled = {d.lower() for d in self._settings.council.disabled}
        members = [
            {"key": s.key, "title": s.title, "on": s.key not in disabled,
             "wants_tools": s.wants_tools}
            for s in load_council(self._settings.council.extra)
        ]
        return web.json_response({
            "ok": True, "enabled": self._settings.council.enabled,
            "max_members": self._settings.council.max_members,
            "members": members,
        })

    async def _handle_council_toggle(self, request: web.Request) -> web.Response:
        """Switch one specialist on/off, or the whole council.

        Body: ``{"agent": "law", "on": false}`` or ``{"enabled": false}``.
        Runtime-only, like the session modes — the skills read
        ``settings.council`` live, so the next question sees the change.
        """
        try:
            payload = await _json_dict(request)
        except (ValueError, TypeError):
            return _error(400, "body must be JSON")
        if "enabled" in payload:
            self._settings.council.enabled = bool(payload["enabled"])
        agent = str(payload.get("agent") or "").strip().lower()
        if agent:
            from core.council import find_specialist, load_council

            council = load_council(self._settings.council.extra)
            if find_specialist(agent, council) is None:
                return _error(404, f"no specialist named {agent!r}")
            disabled = [d.lower() for d in self._settings.council.disabled]
            if bool(payload.get("on", True)):
                disabled = [d for d in disabled if d != agent]
            elif agent not in disabled:
                disabled.append(agent)
            self._settings.council.disabled = disabled
        return await self._handle_council(request)

    async def _handle_modes(self, request: web.Request) -> web.Response:
        """Flip a session mode from the HUD: continuous conversation / interpreter.

        Body ``{"mode": "continuous"|"interpreter", "on": bool}``. The voice
        loop reads the shared SessionModes live, so the change takes effect on
        the next turn. Runtime-only (not persisted) — a mode is a per-session
        choice, not a machine setting.
        """
        try:
            payload = await _json_dict(request)
            mode = str(payload["mode"])
            on = bool(payload["on"])
        except (ValueError, KeyError, TypeError):
            return _error(400, "body must be JSON like {\"mode\": \"continuous\", \"on\": true}")
        if mode == "continuous":
            self._modes.continuous = on
        elif mode == "interpreter":
            self._modes.interpreter = on
        else:
            return _error(400, "mode must be 'continuous' or 'interpreter'")
        logger.info("session mode %r set to %s via companion API", mode, on)
        return web.json_response({"ok": True, "mode": mode, "on": on})

    # --- MEDO Link (M9): manifest-driven device layer -----------------------

    async def _handle_link_register(self, request: web.Request) -> web.Response:
        """A device uploads its manifest; capabilities become tools + patterns."""
        if self._link is None:
            return _error(503, "MEDO Link is not enabled")
        try:
            manifest = await _json_dict(request)
        except ValueError:
            return _error(400, "body must be the device manifest as JSON")
        # to_thread: register persists to sqlite.
        errors = await asyncio.to_thread(self._link.register, manifest)
        if errors:
            return web.json_response({"ok": False, "errors": errors}, status=422)
        return web.json_response({
            "ok": True,
            "device_id": manifest["device_id"],
            "capabilities": len(manifest["capabilities"]),
        })

    async def _handle_link_devices(self, request: web.Request) -> web.Response:
        """Registered devices for the HUD: name, online, capability count."""
        devices = self._link.devices() if self._link is not None else []
        return web.json_response({"ok": True, "devices": devices})

    async def _handle_link_commands(self, request: web.Request) -> web.Response:
        """http_poll transport: a device fetches its queued commands (heartbeat)."""
        if self._link is None:
            return _error(503, "MEDO Link is not enabled")
        device_id = request.match_info["device_id"]
        if not self._link.known(device_id):
            return _error(404, f"unknown device {device_id!r} — register first")
        return web.json_response(
            {"ok": True, "commands": self._link.drain_commands(device_id)})

    async def _handle_link_result(self, request: web.Request) -> web.Response:
        """A device reports one command's outcome; wakes the waiting dispatch."""
        if self._link is None:
            return _error(503, "MEDO Link is not enabled")
        try:
            payload = await _json_dict(request)
            device_id = str(payload["device_id"])
            command_id = str(payload["id"])
        except (ValueError, KeyError, TypeError):
            return _error(400, "body must be JSON like {\"device_id\": …, \"id\": …,"
                               " \"ok\": true, \"message\": \"…\"}")
        if not self._link.known(device_id):
            return _error(404, f"unknown device {device_id!r}")
        matched = self._link.resolve(
            device_id, command_id,
            bool(payload.get("ok", True)), str(payload.get("message") or ""))
        return web.json_response({"ok": True, "matched": matched})

    async def _handle_link_ws(self, request: web.Request) -> web.StreamResponse:
        """websocket transport: commands pushed down, results come back up.

        ``?device_id=…&token=…`` — the query token matters here because many
        embedded websocket clients can't set request headers.
        """
        if self._link is None:
            return _error(503, "MEDO Link is not enabled")
        device_id = request.query.get("device_id", "")
        if not self._link.known(device_id):
            return _error(404, f"unknown device {device_id!r} — register first")
        ws = web.WebSocketResponse(heartbeat=15.0)
        await ws.prepare(request)
        self._link.attach_ws(device_id, ws)
        logger.info("link device %r connected over websocket", device_id)
        try:
            async for msg in ws:
                if msg.type != web.WSMsgType.TEXT:
                    continue
                with contextlib.suppress(ValueError, TypeError, KeyError):
                    data = json.loads(msg.data)
                    self._link.resolve(
                        device_id, str(data.get("id")),
                        bool(data.get("ok", True)),
                        str(data.get("message") or ""))
        finally:
            self._link.detach_ws(device_id)
            logger.info("link device %r websocket closed", device_id)
        return ws

    async def _handle_facts_list(self, request: web.Request) -> web.Response:
        """Remembered facts for the HUD memory manager."""
        facts = await asyncio.to_thread(self._router.facts.list_all)
        return web.json_response({"ok": True, "facts": facts})

    async def _handle_facts_add(self, request: web.Request) -> web.Response:
        """Remember a fact typed into the HUD."""
        try:
            payload = await _json_dict(request)
        except ValueError:
            return _error(400, "body must be JSON like {\"fact\": \"…\"}")
        fact = str(payload.get("fact") or "").strip()
        if not fact:
            return _error(400, "missing or empty 'fact'")
        added = await asyncio.to_thread(self._router.facts.add, fact)
        return web.json_response({"ok": True, "added": added})

    async def _handle_facts_delete(self, request: web.Request) -> web.Response:
        """Forget one fact by id (HUD memory manager delete button)."""
        try:
            payload = await _json_dict(request)
            fact_id = int(payload.get("id"))
        except (ValueError, TypeError):
            return _error(400, "body must be JSON like {\"id\": 3}")
        deleted = await asyncio.to_thread(self._router.facts.delete, fact_id)
        if not deleted:
            return _error(404, f"no fact with id {fact_id}")
        return web.json_response({"ok": True})

    async def _handle_docs_stats(self, request: web.Request) -> web.Response:
        """Documents-RAG index size (files/chunks) for UIs."""
        if self._doc_index is None:
            return web.json_response({"ok": True, "enabled": False, "files": 0, "chunks": 0})
        stats = await asyncio.to_thread(self._doc_index.stats)
        return web.json_response({"ok": True, **stats})

    async def _handle_docs_reindex(self, request: web.Request) -> web.Response:
        """Kick a background reindex of the user's documents."""
        if self._doc_index is None:
            return _error(409, "document indexing is disabled (no embed model)")
        asyncio.create_task(asyncio.to_thread(self._doc_index.reindex))
        return web.json_response({"ok": True, "started": True})

    async def _handle_search_files(self, request: web.Request) -> web.Response:
        """Filename/-folder substring search under the user's folders."""
        query = request.query.get("q", "").strip()
        results = await asyncio.to_thread(dirs.search_files, query, 40) if query else []
        return web.json_response({"ok": True, "results": results})

    async def _handle_search_web(self, request: web.Request) -> web.Response:
        """Keyless DuckDuckGo web search (server-side, so the HUD dodges CORS)."""
        query = request.query.get("q", "").strip()
        if not query:
            return web.json_response({"ok": True, "results": []})
        results = await asyncio.to_thread(_web_search, query, 8)
        if results is None:
            return web.json_response(
                {"ok": False, "results": [], "error": "web search unavailable (offline?)"}
            )
        return web.json_response({"ok": True, "results": results})

    async def _handle_ask(self, request: web.Request) -> web.Response:
        """Route one utterance and return the reply as JSON."""
        try:
            payload = await _json_dict(request)
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


def _web_search(query: str, max_results: int = 8) -> list[dict] | None:
    """DuckDuckGo results as ``[{"title","url","snippet"}]`` (None if unreachable).

    Delegates to the web_search skill's shared ddgs accessor — one place to fix
    when the ddgs API changes — and just maps fields for the HUD.
    """
    from skills.websearch import ddg_text_search

    results = ddg_text_search(query, max_results)
    if results is None:
        return None
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", "") or r.get("url", ""),
            "snippet": r.get("body", ""),
        }
        for r in results
    ]


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"ok": False, "error": message}, status=status)


async def _json_dict(request: web.Request) -> dict:
    """A JSON OBJECT body, or ``{}`` for anything that isn't one.

    A body that is valid JSON but not an object (``5``, ``[1,2]``, ``"x"``)
    used to reach ``payload.get(...)`` and raise ``AttributeError`` → a 500 on
    the request path (including the unauthenticated ``/pair/confirm``). Coercing
    to ``{}`` here means each handler falls into its own "missing field" 400
    instead — never a 500 — while a genuine object is passed through untouched.
    """
    try:
        data = await request.json()
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}
