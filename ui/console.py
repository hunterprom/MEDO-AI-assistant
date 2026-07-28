"""Rich terminal UI: state machine, routing path, transcript, and latency.

Subscribes to the :class:`~core.events.EventBus` for state transitions (so the
`IDLE→LISTENING→THINKING→SPEAKING` cycle is visible) and renders a compact panel
for each turn showing which path handled it and the per-stage timings.

Deliberately print-based (not a full-screen ``Live`` dashboard) so it coexists
with the REPL's ``input()`` prompt and scrolls as a readable transcript.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.events import AssistantState, Event, EventBus, EventType, RoutePath
from core.metrics import BUDGETS, LatencyLog, TurnTimings
from core.router import RouteResult

_STATE_COLOR = {
    AssistantState.IDLE: "grey50",
    AssistantState.LISTENING: "cyan",
    AssistantState.THINKING: "yellow",
    AssistantState.SPEAKING: "green",
}


class ConsoleUI:
    """Renders assistant state and per-turn routing/latency to the terminal."""

    def __init__(self, console: Console, bus: EventBus, name: str, show_states: bool) -> None:
        self.console = console
        self.name = name
        self._show_states = show_states
        bus.subscribe(EventType.STATE_CHANGED, self._on_state)

    def banner(self, mode: str, model: str | None) -> None:
        body = Text.assemble(
            (f"{self.name} v2", "bold green"),
            (f"  ·  {mode}", "white"),
            (f"  ·  model: {model or 'offline'}", "dim"),
        )
        self.console.print(Panel(body, expand=False, border_style="green"))

    async def _on_state(self, event: Event) -> None:
        if not self._show_states:
            return
        _old, new = event.payload
        color = _STATE_COLOR[new]
        # A single compact chip; the state track reads as it cycles.
        self.console.print(f"[{color}]● {new.value.lower()}[/{color}]", highlight=False)

    def transcript(self, text: str) -> None:
        self.console.print(Text.assemble(("you ", "bold cyan"), (text, "white")))

    def turn(self, result: RouteResult, timings: TurnTimings | None = None) -> None:
        """Render one reply: routing badge, spoken text, and latency line."""
        is_llm = result.path is RoutePath.LLM
        badge_style = "magenta" if is_llm else "cyan"
        header = Text.assemble(
            (f" {result.path.value} ", f"reverse {badge_style}"),
            ("  ", ""),
            (result.skill_name or ("conversation" if is_llm else "—"), "dim"),
        )
        body = Text(result.speech, style="white")
        subtitle = self._latency_line(result, timings)
        self.console.print(
            Panel(body, title=header, subtitle=subtitle, title_align="left",
                  subtitle_align="right", border_style=badge_style, expand=True)
        )

    def _latency_line(self, result: RouteResult, timings: TurnTimings | None) -> str:
        parts: list[str] = []
        if timings and timings.wake_to_listen_ms is not None:
            parts.append(f"wake→listen {timings.wake_to_listen_ms:.0f}ms")
        if timings and timings.stt_ms is not None:
            parts.append(f"STT {timings.stt_ms:.0f}ms")
        parts.append(f"route {result.latency_ms:.0f}ms")
        if timings and timings.tts_ms is not None:
            parts.append(f"TTS {timings.tts_ms:.0f}ms")
        return "  ".join(parts)

    def latency_table(self, log: LatencyLog) -> None:
        """Print averaged latencies against the budgets (for /latency)."""
        s = log.summary()
        table = Table(title=f"Latency over {int(s['count'])} turns", title_style="bold")
        table.add_column("stage")
        table.add_column("avg", justify="right")
        table.add_column("budget", justify="right")

        def row(label: str, value: float | None, budget_key: str | None) -> None:
            budget = BUDGETS.get(budget_key) if budget_key else None
            avg = "—" if value is None else f"{value:.0f} ms"
            within = "" if (value is None or budget is None) else (
                " ✓" if value <= budget else " ✗")
            table.add_row(label, avg + within, "—" if budget is None else f"{budget:.0f} ms")

        row("wake → listen", s["wake_to_listen_ms"], "wake_to_listen_ms")
        row("fast-path route", s["fast_route_ms"], "fast_route_ms")
        row("STT", s["stt_ms"], None)
        row("TTS synth", s["tts_ms"], None)
        row("LLM route", s["llm_route_ms"], None)
        self.console.print(table)
