"""MEDO v2 — entry point.

The text REPL (M0) and the hands-free voice pipeline (M1) share one wiring:
registry, Intent Router, event bus, and state machine. Text mode lets you type;
voice mode listens for the wake word, transcribes, routes, and speaks the reply.

Usage:
    python main.py            # text REPL
    python main.py --voice    # hands-free: wake word -> STT -> route -> TTS
    python main.py --hud      # also serve the M.E.D.O. web HUD
    python main.py --once "what time is it"   # single request, then exit
    python main.py --serve    # also run the companion API (watch app + vision)

Hand gestures run as a separate process (their own venv): see vision/run.py.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from rich.console import Console

from collections.abc import Awaitable

from core.config import PROJECT_ROOT, Settings, load_settings
from core.events import AssistantState, EventBus, RoutePath, StateMachine
from core.memory import NoteStore, ReminderStore
from core.metrics import LatencyLog, TurnTimings
from core.router import Router
from core.safety import PathWhitelist
from ui.console import ConsoleUI
from ui.hud import HudServer
from llm.client import LLMUnavailableError, OllamaClient
from remote.server import RemoteServer
from skills.apps import AppsSkill
from skills.base import SkillRegistry
from skills.datetime_skill import DateTimeSkill
from skills.files import FilesSkill
from skills.media import MediaSkill
from skills.news import NewsSkill
from skills.notes import NotesSkill
from skills.system import PowerSkill, ScreenshotSkill, SystemInfoSkill, VolumeSkill
from skills.timers import TimerSkill
from skills.weather import WeatherSkill
from skills.websearch import WebSearchSkill

console = Console()


class Announcer:
    """Delivers async messages (e.g. a timer firing) to whatever output is live.

    Prints to the console always; if voice mode registers a ``speak`` coroutine,
    it speaks too. Injected into skills that fire on their own schedule.
    """

    def __init__(self) -> None:
        self.speak: "callable[[str], Awaitable[None]] | None" = None

    async def __call__(self, text: str) -> None:
        console.print(f"[yellow]🔔 {text}[/yellow]")
        if self.speak is not None:
            await self.speak(text)

STATE_STYLES = {
    AssistantState.IDLE: "dim",
    AssistantState.LISTENING: "cyan",
    AssistantState.THINKING: "yellow",
    AssistantState.SPEAKING: "green",
}


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Keep routing logs (core.router) but silence chatty third-party request logs.
    for noisy in ("httpx", "httpcore", "openwakeword", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_registry(
    settings: Settings,
    announcer: Announcer,
    summarize: "callable[[str, str], Awaitable[str]] | None" = None,
    reminder_store: ReminderStore | None = None,
) -> SkillRegistry:
    """Register every skill. Order sets fast-path precedence on overlaps.

    Timers/notes come before power/system so "remind me … to sleep" and "note
    about memory" aren't stolen by the broad ``sleep`` / ``memory`` patterns.
    Web skills come last (broad "search for …"). ``summarize`` (LLM) powers the
    fast-path web-search summary; None => it returns raw results.
    """
    registry = SkillRegistry()
    apps_table = settings.skills.get("apps", {})
    whitelist = PathWhitelist(settings.safety.whitelist_dirs)
    notes = NoteStore(settings.memory.db_path)
    shots = PROJECT_ROOT / "screenshots"

    registry.register(DateTimeSkill())
    registry.register(TimerSkill(announcer, reminder_store))
    registry.register(NotesSkill(notes))
    registry.register(AppsSkill(apps_table))
    registry.register(FilesSkill(whitelist))
    registry.register(VolumeSkill())
    registry.register(MediaSkill())
    registry.register(SystemInfoSkill())
    registry.register(ScreenshotSkill(shots))
    registry.register(PowerSkill())
    # Web skills (network; degrade gracefully offline).
    registry.register(WeatherSkill(settings.weather))
    registry.register(NewsSkill(settings.news))
    registry.register(WebSearchSkill(summarize))
    return registry


def select_startup_model(llm: OllamaClient, settings: Settings) -> str | None:
    """Pick the LLM model to start with (runtime model selection)."""
    if not llm.is_available():
        console.print(
            "[yellow]Ollama not reachable — LLM path disabled. Fast-path skills "
            "still work. Start Ollama and restart, or use /model later.[/yellow]"
        )
        return None

    installed = llm.list_models()
    configured = settings.llm.default_model
    if configured and configured in installed:
        return configured
    if configured and configured not in installed:
        console.print(
            f"[yellow]Configured model {configured!r} is not installed "
            f"(have: {', '.join(installed) or 'none'}).[/yellow]"
        )
    if installed:
        chosen = installed[0]
        console.print(
            f"[dim]Using model [bold]{chosen}[/bold]. "
            f"Type /models to list, /model <name> to switch.[/dim]"
        )
        return chosen
    console.print(
        "[yellow]No Ollama models installed. Run e.g. "
        "`ollama pull qwen2.5:7b`, then /model to select.[/yellow]"
    )
    return None


HELP = """[bold]Commands[/bold]
  /models         list installed Ollama models
  /model <name>   switch the active LLM model
  /stats          show fast-path vs LLM-path counts
  /latency        average per-stage latency vs budgets
  /help           this help
  /quit           exit
Anything else is routed to MEDO."""


async def handle_command(
    cmd: str, router: Router, llm: OllamaClient, log: LatencyLog, ui: ConsoleUI
) -> bool:
    """Handle a /command. Returns False to signal exit, True otherwise."""
    parts = cmd.split(maxsplit=1)
    name = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if name in ("/quit", "/exit"):
        return False
    if name == "/help":
        console.print(HELP)
    elif name == "/models":
        models = llm.list_models()
        console.print("Installed models: " + (", ".join(models) or "[dim]none[/dim]"))
    elif name == "/model":
        if not arg:
            console.print("Usage: /model <name>. Current: " + (router.model or "none"))
        else:
            router.model = arg
            console.print(f"Active model set to [bold]{arg}[/bold].")
    elif name == "/stats":
        s = router.stats
        console.print(f"Routing — FAST: {s[RoutePath.FAST]}  LLM: {s[RoutePath.LLM]}")
    elif name == "/latency":
        ui.latency_table(log)
    else:
        console.print(f"[red]Unknown command {name!r}. Try /help.[/red]")
    return True


async def respond(
    router: Router, sm: StateMachine, ui: ConsoleUI, log: LatencyLog, text: str
) -> None:
    """Route one utterance and render the reply with its path and latency."""
    await sm.transition(AssistantState.THINKING)
    result = await router.route(text)
    await sm.transition(AssistantState.SPEAKING)

    timings = TurnTimings(path=result.path.value, route_ms=result.latency_ms)
    log.record(timings)
    ui.turn(result, timings)
    await sm.transition(AssistantState.IDLE)


async def run_repl(
    router: Router, sm: StateMachine, llm: OllamaClient, ui: ConsoleUI, log: LatencyLog
) -> None:
    while True:
        await sm.transition(AssistantState.IDLE)
        try:
            text = (await asyncio.to_thread(input, "you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            return
        if not text:
            continue
        if text.startswith("/"):
            if not await handle_command(text, router, llm, log, ui):
                console.print("[dim]Goodbye.[/dim]")
                return
            continue
        await respond(router, sm, ui, log, text)


async def run_voice(
    settings: Settings, router: Router, sm: StateMachine, announcer: Announcer,
    ui: ConsoleUI, log: LatencyLog,
) -> None:
    """Hands-free loop: wake word -> record -> STT -> route -> TTS.

    Blocking audio/model calls run in worker threads so the asyncio loop (and the
    companion API, if serving) stays responsive. After a reply that asks for
    confirmation, we record again *without* the wake word so "yes" works
    naturally. Every stage is timed into a :class:`TurnTimings`.
    """
    import time

    from voice.audio import Microphone, Speaker, record_until_silence
    from voice.stt import Transcriber
    from voice.tts import TextToSpeech
    from voice.wakeword import WakeWord

    with console.status("[dim]loading voice models…[/dim]"):
        wake = WakeWord(settings.wakeword)
        stt = Transcriber(settings.stt)
        # A missing/broken Piper voice shouldn't sink the whole session — degrade
        # to showing replies without speaking them (the HUD still renders them).
        try:
            tts: TextToSpeech | None = TextToSpeech(settings.tts)
        except Exception as exc:
            tts = None
            console.print(
                f"[yellow]TTS voice unavailable ({exc}); replies will be shown, "
                f"not spoken. Run run.bat to fetch the Piper voice.[/yellow]"
            )
    speaker = Speaker(settings.audio.output_device)

    def wait_for_wake(mic: Microphone) -> None:
        wake.reset()
        while not wake.triggered(mic.read_frame()):
            pass

    async def speak(text: str) -> float:
        """Synthesize and play; returns synth time (ms) for instrumentation."""
        if tts is None:  # TTS unavailable — reply is shown, not spoken.
            return 0.0
        t0 = time.perf_counter()
        wav, sr = await asyncio.to_thread(tts.synthesize, text)
        tts_ms = (time.perf_counter() - t0) * 1000
        await asyncio.to_thread(speaker.play, wav, sr)
        return tts_ms

    async def capture(require_wake: bool) -> tuple[np.ndarray, float]:
        """Open the mic, optionally wait for the wake word, record a phrase.

        Returns the audio plus the wake→listen latency (ms)."""
        mic = Microphone(settings.audio.sample_rate, device=settings.audio.input_device)
        try:
            mic.open()
            if require_wake:
                await asyncio.to_thread(wait_for_wake, mic)
            t_wake = time.perf_counter()
            await sm.transition(AssistantState.LISTENING)
            wake_to_listen_ms = (time.perf_counter() - t_wake) * 1000
            audio = await asyncio.to_thread(
                record_until_silence,
                mic,
                silence_threshold=settings.audio.silence_threshold,
                silence_duration_s=settings.audio.silence_duration_s,
            )
            return audio, wake_to_listen_ms
        finally:
            mic.close()

    # Timer/reminder announcements should be spoken, not just printed.
    announcer.speak = speak

    ui.banner("voice mode — say “%s”" % settings.wakeword.phrase.replace("_", " "), router.model)
    require_wake = True
    while True:
        if require_wake:
            await sm.transition(AssistantState.IDLE)
        audio, wake_to_listen_ms = await capture(require_wake)
        if audio.size == 0:
            if require_wake:
                console.print("[dim](heard nothing — back to sleep)[/dim]")
            require_wake = True
            continue

        await sm.transition(AssistantState.THINKING)
        t0 = time.perf_counter()
        text = await asyncio.to_thread(stt.transcribe, audio)
        stt_ms = (time.perf_counter() - t0) * 1000
        if not text.strip():
            console.print("[dim](couldn't make that out)[/dim]")
            require_wake = True
            continue
        ui.transcript(text)

        result = await router.route(text)

        await sm.transition(AssistantState.SPEAKING)
        tts_ms = await speak(result.speech)
        log.record(TurnTimings(
            path=result.path.value,
            wake_to_listen_ms=wake_to_listen_ms if require_wake else None,
            stt_ms=stt_ms, route_ms=result.latency_ms, tts_ms=tts_ms,
        ))
        ui.turn(result, log.turns[-1])
        # If MEDO just asked "are you sure?", listen for the yes/no without a wake.
        require_wake = not router.awaiting_confirmation


async def async_main(once: str | None, serve: bool, voice: bool, hud: bool) -> None:
    settings = load_settings()
    setup_logging(settings.logging.level)

    announcer = Announcer()
    llm = OllamaClient(settings.llm)

    async def summarize(query: str, results_text: str) -> str:
        """Summarize web results into a couple of spoken sentences (fast path)."""
        if not router.model:
            return "I found some results, but I can't summarize them offline."
        prompt = [
            {"role": "system", "content": (
                "Summarize these web search results into two or three spoken "
                "sentences. Plain text, no markdown, no lists.")},
            {"role": "user", "content": f"Query: {query}\n\nResults:\n{results_text}"},
        ]
        try:
            message = await llm.chat(router.model, prompt)
        except LLMUnavailableError:
            return "I found results, but my summarizer is offline."
        return (message.get("content") or "").strip() or "I couldn't summarize that."

    reminders = ReminderStore(settings.memory.db_path)
    registry = build_registry(settings, announcer, summarize, reminders)
    bus = EventBus()
    sm = StateMachine(bus)
    router = Router(settings, registry, llm, bus)
    router.model = select_startup_model(llm, settings)

    log = LatencyLog()
    # State chips are shown in voice mode; in the REPL they'd clutter the prompt.
    ui = ConsoleUI(console, bus, settings.personality.name, show_states=voice)

    if once is not None:
        await respond(router, sm, ui, log, once)
        return

    remote: RemoteServer | None = None
    # The vision sidecar POSTs gestures to the companion API, so enabling vision
    # (or the HUD, which reads the same events) implies serving it.
    if serve or settings.remote.enabled or settings.vision.enabled:
        remote = RemoteServer(settings, router, sm)
        await remote.start()
        console.print(
            f"[dim]Companion API on port {settings.remote.port} — "
            f"watch app + vision sidecar connect here.[/dim]"
        )

    hud_server: HudServer | None = None
    if hud or settings.hud.enabled:
        hud_server = HudServer(settings, bus)
        await hud_server.start()
        console.print(
            f"[bold cyan]HUD:[/bold cyan] open "
            f"[underline]http://localhost:{settings.hud.port}[/underline] in a browser."
        )

    try:
        if voice:
            # Voice pipeline; the companion API (if started) serves alongside it.
            try:
                await run_voice(settings, router, sm, announcer, ui, log)
            except (KeyboardInterrupt, asyncio.CancelledError):
                raise
            except Exception as exc:
                # Don't let a broken mic/voice stack take the HUD + gestures down
                # with it — keep serving them (or fall back to the text REPL).
                logging.getLogger("main").error("voice mode failed", exc_info=exc)
                console.print(f"[red]Voice mode couldn't start: {exc}[/red]")
                if remote is not None or hud_server is not None:
                    console.print(
                        "[yellow]HUD + companion API stay up — use the HUD command "
                        "box or hand gestures. Ctrl-C to quit.[/yellow]"
                    )
                    await asyncio.Event().wait()
                else:
                    console.print("[yellow]Falling back to text mode.[/yellow]")
                    ui.banner("text mode — /help for commands", router.model)
                    await run_repl(router, sm, llm, ui, log)
        elif remote is not None and not sys.stdin.isatty():
            # Headless (e.g. launched as a service): serve until interrupted.
            await asyncio.Event().wait()
        else:
            ui.banner("text mode — /help for commands", router.model)
            await run_repl(router, sm, llm, ui, log)
    finally:
        if remote is not None:
            await remote.stop()
        if hud_server is not None:
            await hud_server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="MEDO v2 local voice assistant")
    parser.add_argument("--once", metavar="TEXT", help="route one request and exit")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="also start the companion API for the watch app (see config: remote)",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="hands-free voice mode: wake word -> STT -> route -> TTS",
    )
    parser.add_argument(
        "--hud",
        action="store_true",
        help="serve the M.E.D.O. web HUD (arc-reactor front end)",
    )
    args = parser.parse_args()
    try:
        asyncio.run(async_main(args.once, args.serve, args.voice, args.hud))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
