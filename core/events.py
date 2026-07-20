"""Event-driven core: assistant states, routing paths, and a tiny async bus.

The pipeline is coordinated by state transitions:

    IDLE -> LISTENING -> THINKING -> SPEAKING -> IDLE

Every subsystem (router now; wake word / STT / TTS later) publishes events on the
:class:`EventBus`. The console UI (M5) subscribes to visualize state and routing.
Keeping this decoupled means no component imports the UI directly.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AssistantState(str, Enum):
    """Where the assistant is in the wake -> listen -> think -> speak cycle."""

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


class RoutePath(str, Enum):
    """Which brain handled a request. Logged for the README routing stats."""

    FAST = "FAST"   # deterministic rule-based skill, no LLM
    LLM = "LLM"     # Ollama conversation / tool calling


class EventType(str, Enum):
    STATE_CHANGED = "state_changed"
    TRANSCRIPT = "transcript"          # user text (typed or from STT)
    RESPONSE = "response"              # assistant text (before/for TTS)
    ROUTED = "routed"                  # a RouteResult was produced
    CAPTION = "caption"                # interpreter mode: source + translation
    ERROR = "error"


@dataclass
class Event:
    type: EventType
    payload: Any = None


Subscriber = Callable[[Event], Awaitable[None] | None]


class EventBus:
    """Minimal async publish/subscribe hub.

    Subscribers may be sync or async callables. ``emit`` awaits async subscribers
    and never lets one subscriber's failure break the others.
    """

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[Subscriber]] = {}

    def subscribe(self, event_type: EventType, handler: Subscriber) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)

    async def emit(self, event: Event) -> None:
        for handler in self._subscribers.get(event.type, []):
            try:
                result = handler(event)
                if isinstance(result, Awaitable):
                    await result
            except Exception:  # a broken subscriber must not crash the pipeline
                logger.exception("event subscriber failed for %s", event.type)


@dataclass
class StateMachine:
    """Holds the current :class:`AssistantState` and announces transitions."""

    bus: EventBus
    state: AssistantState = AssistantState.IDLE
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def transition(self, new_state: AssistantState) -> None:
        async with self._lock:
            if new_state == self.state:
                return
            old, self.state = self.state, new_state
        logger.debug("state: %s -> %s", old.value, new_state.value)
        await self.bus.emit(Event(EventType.STATE_CHANGED, (old, new_state)))
