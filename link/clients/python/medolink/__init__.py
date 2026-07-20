"""medolink — EXPERIMENTAL Python client for MEDO Link (Raspberry Pi etc.).

Register a manifest, receive commands, post results. One protocol, ~100
lines: HTTP polling by default (works everywhere), and every request carries
the bearer token from MEDO's ``secrets.local.yaml`` (``remote.token``).

    import asyncio
    from medolink import MedoLink

    MANIFEST = {
        "device_id": "lamp",
        "name": "the desk lamp",
        "capabilities": [
            {"name": "on",  "description": "Turn the desk lamp on.",
             "fast_patterns": ["\\\\blamp on\\\\b"]},
            {"name": "off", "description": "Turn the desk lamp off.",
             "fast_patterns": ["\\\\blamp off\\\\b"]},
        ],
    }

    link = MedoLink("192.168.1.20:8710", token="...", manifest=MANIFEST)

    @link.on("on")
    async def lamp_on(params):
        return "Lamp is on."       # spoken by MEDO

    @link.on("off")
    async def lamp_off(params):
        return "Lamp is off."

    asyncio.run(link.run())
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import aiohttp

__all__ = ["MedoLink"]

logger = logging.getLogger("medolink")

Handler = Callable[[dict], Awaitable[str | None]]


class MedoLink:
    """Polling MEDO Link client: register once, then poll + answer commands."""

    def __init__(self, address: str, token: str, manifest: dict,
                 poll_interval_s: float = 1.5) -> None:
        self._base = f"http://{address}"
        self._token = token
        self._manifest = manifest
        self._poll_interval = poll_interval_s
        self._handlers: dict[str, Handler] = {}

    def on(self, capability: str) -> Callable[[Handler], Handler]:
        """Decorator binding a capability name to an async handler."""
        def bind(handler: Handler) -> Handler:
            self._handlers[capability] = handler
            return handler
        return bind

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    async def run(self) -> None:
        """Register, then poll for commands forever (reconnects on errors)."""
        device_id = self._manifest["device_id"]
        async with aiohttp.ClientSession(headers=self._headers) as http:
            async with http.post(f"{self._base}/link/register",
                                 json=self._manifest) as resp:
                body = await resp.json()
                if resp.status != 200:
                    raise RuntimeError(f"registration failed: {body}")
            logger.info("registered %r (%s capabilities)", device_id,
                        body.get("capabilities"))
            while True:
                try:
                    async with http.get(
                            f"{self._base}/link/commands/{device_id}") as resp:
                        commands = (await resp.json()).get("commands", [])
                    for command in commands:
                        await self._handle(http, device_id, command)
                except aiohttp.ClientError as exc:
                    logger.warning("MEDO unreachable (%s); retrying", exc)
                await asyncio.sleep(self._poll_interval)

    async def _handle(self, http: aiohttp.ClientSession, device_id: str,
                      command: dict) -> None:
        handler = self._handlers.get(command.get("capability", ""))
        ok, message = True, ""
        if handler is None:
            ok, message = False, f"no handler for {command.get('capability')!r}"
        else:
            try:
                message = await handler(command.get("params") or {}) or ""
            except Exception as exc:  # a crashing handler still answers MEDO
                logger.exception("handler failed")
                ok, message = False, f"device error: {exc}"
        await http.post(f"{self._base}/link/result", json={
            "device_id": device_id, "id": command.get("id"),
            "ok": ok, "message": message,
        })
