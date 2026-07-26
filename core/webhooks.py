"""Outbound event webhooks — POST a JSON payload to a URL when MEDO does things.

Configured in ``config.yaml`` under ``webhooks:`` (or added from the HUD CONFIG
tab, which persists them to the git-ignored overrides so their auth tokens are
never committed). Each :class:`~core.config.WebhookConfig` binds one of four
lifecycle events to a URL:

======================  ================  ==================================
event                   fires on          payload
======================  ================  ==================================
``wake``                a turn started    ``{event, state, at}``
``transcript``          something heard   ``{event, text, at}``
``routed``              a command handled ``{event, skill, path, speech, at}``
``reply``               MEDO answered     ``{event, text, at}``
======================  ================  ==================================

This lets MEDO drive external automations (n8n, Home Assistant, a logger,
Discord) with zero per-integration code. Design notes:

* Firing is **fire-and-forget**: the bus subscriber schedules the POST as a
  task and returns immediately, so a slow or dead endpoint never stalls a turn.
* A failed POST is logged at debug and dropped — a webhook is never fatal.
* ``aiohttp`` is imported lazily; if it's missing, webhooks degrade to a warning.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from core.config import WebhookConfig
from core.events import AssistantState, Event, EventBus, EventType
from core.router import RouteResult

logger = logging.getLogger(__name__)

#: Don't let a hung endpoint hold a connection open forever.
_POST_TIMEOUT_S = 8.0

#: HUD-friendly event label -> the EventBus type it subscribes to.
_EVENT_TYPES = {
    "wake": EventType.STATE_CHANGED,
    "transcript": EventType.TRANSCRIPT,
    "routed": EventType.ROUTED,
    "reply": EventType.RESPONSE,
}


class WebhookManager:
    """Subscribes to the bus and POSTs matching webhooks when events fire."""

    def __init__(self, webhooks: list[WebhookConfig]) -> None:
        # Only enabled hooks with a URL and a known event are ever fired.
        self._hooks = [
            h for h in (webhooks or [])
            if h.enabled and h.url and h.event in _EVENT_TYPES
        ]
        self._session: Any = None
        self._tasks: set[asyncio.Task] = set()

    @property
    def active(self) -> bool:
        return bool(self._hooks)

    def subscribe(self, bus: EventBus) -> int:
        """Wire this manager onto ``bus``. Returns the number of live webhooks."""
        if not self._hooks:
            return 0
        # Subscribe once per distinct EventType actually in use.
        for event_type in {_EVENT_TYPES[h.event] for h in self._hooks}:
            bus.subscribe(event_type, self._on_event)
        logger.info("webhooks: %d live (%s)", len(self._hooks),
                    ", ".join(sorted({h.event for h in self._hooks})))
        return len(self._hooks)

    def _on_event(self, event: Event) -> None:
        """Bus subscriber (sync, non-blocking): schedule POSTs, never await them."""
        payload = self._payload(event)
        if payload is None:
            return
        label = payload["event"]
        for hook in self._hooks:
            if hook.event == label:
                task = asyncio.ensure_future(self._post(hook, payload))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

    def _payload(self, event: Event) -> dict[str, Any] | None:
        """Build the JSON body for an event, or None if it shouldn't fire."""
        at = datetime.now().isoformat(timespec="seconds")
        if event.type is EventType.STATE_CHANGED:
            # A turn "wakes" on the transition INTO listening — not every change.
            new = event.payload[1] if isinstance(event.payload, tuple) else None
            if new is not AssistantState.LISTENING:
                return None
            return {"event": "wake", "state": AssistantState.LISTENING.value, "at": at}
        if event.type is EventType.TRANSCRIPT:
            return {"event": "transcript", "text": str(event.payload or ""), "at": at}
        if event.type is EventType.RESPONSE:
            return {"event": "reply", "text": str(event.payload or ""), "at": at}
        if event.type is EventType.ROUTED:
            r = event.payload
            if not isinstance(r, RouteResult):
                return None
            return {
                "event": "routed",
                "skill": r.skill_name or "",
                "path": r.path.value if r.path else "",
                "speech": r.speech or "",
                "at": at,
            }
        return None

    async def _post(self, hook: WebhookConfig, payload: dict[str, Any]) -> None:
        """POST one payload to one hook. Best-effort: logs and swallows failures."""
        try:
            session = await self._ensure_session()
            if session is None:
                return
            headers = ({"Authorization": f"Bearer {hook.auth_token}"}
                       if hook.auth_token else None)
            async with session.post(hook.url, json=payload, headers=headers,
                                    timeout=_POST_TIMEOUT_S) as resp:
                if resp.status >= 400:
                    logger.debug("webhook %r -> HTTP %s", hook.name, resp.status)
        except Exception as exc:  # dead endpoint, DNS, timeout — never fatal
            logger.debug("webhook %r failed: %s", hook.name, exc)

    async def _ensure_session(self) -> Any:
        if self._session is None:
            try:
                import aiohttp
            except ImportError:
                logger.warning("webhooks configured but 'aiohttp' is not installed")
                self._hooks = []       # stop trying
                return None
            self._session = aiohttp.ClientSession()
        return self._session

    async def stop(self) -> None:
        """Cancel in-flight POSTs and close the shared session (idempotent)."""
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                logger.debug("error closing webhook session", exc_info=True)
            self._session = None

    def status(self) -> list[dict[str, Any]]:
        """Summary for the HUD CONFIG tab."""
        return [{"name": h.name, "event": h.event, "url": h.url,
                 "has_auth": bool(h.auth_token)} for h in self._hooks]
