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
import threading

from rich.console import Console

from collections.abc import Awaitable

from core.config import PROJECT_ROOT, Settings, apply_local_secrets, load_settings
from core.events import AssistantState, EventBus, RoutePath, StateMachine
from core.facts import FactsStore
from core.memory import NoteStore, ReminderStore
from core.metrics import LatencyLog, TurnTimings
from core.router import Router
from core.safety import PathWhitelist
from ui.console import ConsoleUI
from ui.hud import HudServer
from llm.client import LLMUnavailableError, OllamaClient
from remote.server import RemoteServer
from voice.loop import VoiceLoop
from skills.apps import AppsSkill
from skills.base import SkillRegistry
from skills.datetime_skill import DateTimeSkill
from skills.desktop import (
    BrightnessSkill,
    ClipboardSkill,
    PressKeysSkill,
    TypeTextSkill,
    WindowActionSkill,
)
from skills.files import FilesSkill
from skills.media import MediaSkill
from skills.memory_skill import ForgetFactSkill, RecallFactsSkill, RememberFactSkill
from skills.news import NewsSkill
from skills.notes import NotesSkill
from skills.system import (
    PointerControlSkill,
    PowerSkill,
    ScreenshotSkill,
    SystemInfoSkill,
    VolumeSkill,
)
from skills.timers import TimerSkill
from skills.vision_skill import SeeCameraSkill, SeeScreenSkill
from skills.weather import WeatherSkill
from skills.websearch import WebSearchSkill

# Never let a pretty glyph kill the app: on legacy/cp1252 consoles (and
# redirected stdout) rich's output hits charmap encoding, and one un-encodable
# character (the "●" state chip took down a whole voice session) raises
# UnicodeEncodeError mid-print. errors="replace" renders those as "?" instead.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

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
    doc_index=None,
    briefing_rewrite=None,
) -> SkillRegistry:
    """Register every skill. Order sets fast-path precedence on overlaps.

    Timers/notes come before power/system so "remind me … to sleep" and "note
    about memory" aren't stolen by the broad ``sleep`` / ``memory`` patterns.
    Web skills come last (broad "search for …"). ``summarize`` (LLM) powers the
    fast-path web-search summary; None => it returns raw results.
    ``briefing_rewrite`` is the LLM pass that turns briefing sections into one
    flowing spoken paragraph; None => the briefing speaks its raw sections.
    """
    registry = SkillRegistry()
    apps_table = settings.skills.get("apps", {})
    whitelist = PathWhitelist(settings.safety.whitelist_dirs)
    notes = NoteStore(settings.memory.db_path)
    facts = FactsStore(settings.memory.db_path)
    shots = PROJECT_ROOT / "screenshots"
    # Weather/news are registered LAST (broad patterns) but constructed here so
    # the briefing chains the very same instances instead of duplicating them.
    weather_skill = WeatherSkill(settings.weather)
    news_skill = NewsSkill(settings.news)

    registry.register(DateTimeSkill())
    registry.register(TimerSkill(announcer, reminder_store))
    # Morning briefing (M8): chains weather/news/reminders/facts; registered
    # early so "brief me" can't be stolen by broader patterns.
    from skills.briefing import BriefingSkill

    registry.register(BriefingSkill(
        settings.briefing.sections, weather_skill, news_skill,
        reminders=reminder_store, facts=facts, rewrite=briefing_rewrite,
    ))
    registry.register(NotesSkill(notes))
    # Long-term facts. Recall/forget register before remember so "what do you
    # remember" is answered, never stored.
    registry.register(RecallFactsSkill(facts, settings.memory.max_facts))
    registry.register(ForgetFactSkill(facts))
    registry.register(RememberFactSkill(facts))
    # Window actions before apps: "close the window" is not "close <app>".
    registry.register(WindowActionSkill())
    registry.register(AppsSkill(apps_table))
    registry.register(SeeCameraSkill(settings))
    registry.register(SeeScreenSkill(settings))
    # M11 deictic pointing: "what is this?" crops around the mouse cursor.
    from skills.vision_skill import PointAtSkill

    registry.register(PointAtSkill(settings))
    # Documents RAG before FilesSkill/WebSearch so "search my documents for X"
    # isn't stolen by the filename search or the broad web "search for …".
    if doc_index is not None:
        from skills.documents import DocumentsSkill

        registry.register(DocumentsSkill(doc_index))
    registry.register(FilesSkill(whitelist))
    registry.register(VolumeSkill())
    registry.register(MediaSkill())
    registry.register(TypeTextSkill())
    registry.register(PressKeysSkill())
    registry.register(ClipboardSkill())
    registry.register(BrightnessSkill())
    registry.register(SystemInfoSkill())
    registry.register(ScreenshotSkill(shots))
    registry.register(PointerControlSkill(settings.vision.stream_port))
    registry.register(PowerSkill())
    # Bench camera (M10): on-demand part identification; the sidecar opens
    # camera 2 per request. Registered late so core skills keep precedence
    # over its broad "do i have any …" inventory pattern.
    from skills.bench import BenchInventory, BenchSkill

    bench_embedder = None
    if settings.memory.embed_model:
        from core.embeddings import embed_texts as _bench_embed

        _bm, _bh = settings.memory.embed_model, settings.llm.host

        def bench_embedder(texts):  # noqa: E731 - tiny closure over config
            return _bench_embed(texts, _bm, _bh)

    registry.register(BenchSkill(
        settings, BenchInventory(settings.memory.db_path, bench_embedder)))
    # Drop-in plugins load after the core skills (which keep fast-path
    # precedence) but BEFORE the broad web skills — otherwise web_search's
    # greedy "search … for …" pattern steals plugin triggers like
    # "search obsidian for …". A broken plugin is skipped, never fatal.
    from core.plugins import load_plugins

    load_plugins(registry, {
        "settings": settings,
        "announcer": announcer,
        "summarize": summarize,
        "reminders": reminder_store,
        "doc_index": doc_index,
    })

    # Web skills (network; degrade gracefully offline; broad patterns last).
    registry.register(weather_skill)
    registry.register(news_skill)
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


def _trust_os_certificates() -> None:
    """Make Python's TLS trust the Windows certificate store.

    This network intercepts TLS with its own root CA; httpx/requests ship the
    certifi bundle which doesn't contain it, so every https call to an online
    LLM API (Groq/OpenAI/...), and even weather/news feeds, dies with
    CERTIFICATE_VERIFY_FAILED. truststore patches ssl to use the OS store —
    where that root actually lives — fixing all of them at once.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:  # pragma: no cover - best effort; local Ollama unaffected
        logging.getLogger("main").debug("truststore unavailable", exc_info=True)


async def async_main(once: str | None, serve: bool, voice: bool, hud: bool) -> None:
    _trust_os_certificates()
    settings = load_settings()
    # Overlay any online API key / model saved via the HUD (git-ignored file), so
    # an online provider chosen last session is restored without touching config.yaml.
    apply_local_secrets(settings)
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

    async def briefing_rewrite(macedonian: bool, raw_sections: str) -> str:
        """One LLM pass: briefing sections -> a flowing spoken paragraph (M8).

        Persona-aware (M7 fragment) and language-matched. Returns "" when no
        model is available — the briefing then speaks its raw sections.
        """
        if not router.model:
            return ""
        from core.persona import Persona

        language = "Macedonian" if macedonian else "English"
        prompt = [
            {"role": "system", "content": (
                f"You are {settings.personality.name}, a voice assistant. "
                f"{Persona(settings.personality).prompt_fragment()} "
                f"Rewrite the user's briefing sections into ONE flowing spoken "
                f"morning briefing in {language}, 30 to 60 seconds when read "
                f"aloud. Plain text only — no markdown, no lists, no headings. "
                f"Keep every fact; invent nothing.")},
            {"role": "user", "content": raw_sections},
        ]
        try:
            message = await llm.chat(router.model, prompt)
        except LLMUnavailableError:
            return ""
        return (message.get("content") or "").strip()

    reminders = ReminderStore(settings.memory.db_path)

    # Documents RAG index: same local embedder as semantic facts, same safety
    # whitelist as the files skill. None when embeddings are disabled.
    doc_index = None
    if settings.memory.embed_model:
        from core.docindex import DocumentIndex
        from core.embeddings import embed_texts

        _em, _eh = settings.memory.embed_model, settings.llm.host
        # The Obsidian vault (docs/) is indexed FIRST: notes are the densest,
        # most-asked-about content, and the whitelist walk could take a while
        # (or hit the chunk cap) before reaching it otherwise.
        _roots = [p for p in [PROJECT_ROOT / "docs"] if p.exists()]
        _roots += PathWhitelist(settings.safety.whitelist_dirs).roots
        doc_index = DocumentIndex(
            settings.memory.db_path,
            lambda texts: embed_texts(texts, _em, _eh),
            _roots,
        )

    registry = build_registry(settings, announcer, summarize, reminders, doc_index,
                              briefing_rewrite=briefing_rewrite)

    # MCP: connect configured servers and register their tools as skills, so
    # any application that speaks the Model Context Protocol becomes callable
    # by the LLM path. A bad server (or missing `mcp` package) never blocks
    # startup — it's logged and skipped.
    from core.mcp import MCPManager

    mcp_manager = MCPManager(settings.mcp)
    mcp_tools = await mcp_manager.start(registry)
    if mcp_tools:
        console.print(f"[dim]MCP: {mcp_tools} tool(s) from connected servers.[/dim]")

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

    # Preload the local model in the background: a cold 30B costs ~27 s on the
    # first question otherwise, which users read as "it doesn't answer".
    if settings.llm.provider == "ollama" and router.model:
        asyncio.create_task(llm.warmup(router.model))
    # Index the user's documents in the background so "what do my documents
    # say about X" has something to search (incremental; skips unchanged files).
    if doc_index is not None:
        asyncio.create_task(asyncio.to_thread(doc_index.reindex))
    # Proactive routines (morning briefing etc.): answers are announced —
    # spoken in voice mode, printed otherwise, visible in the HUD either way.
    if settings.routines:
        from core.routines import RoutineScheduler

        asyncio.create_task(
            RoutineScheduler(settings.routines, router, announcer).run_forever()
        )

    # Manual-wake signal: POST /wake sets it and the voice loop's wake-word wait
    # returns immediately — so you can start a turn from the HUD without saying
    # the "hey jarvis" phrase (fixes being stuck in STANDING BY). Only handed to
    # the server when voice mode will actually consume it, so /wake correctly
    # 409s in text-only sessions instead of pretending to listen.
    wake_event = threading.Event()

    remote: RemoteServer | None = None
    # The vision sidecar POSTs gestures to the companion API, so enabling vision
    # (or the HUD, which reads the same events) implies serving it.
    if serve or settings.remote.enabled or settings.vision.enabled:
        # First serve mints the LAN auth token into secrets.local.yaml; later
        # runs just load it. Localhost clients (HUD, sidecar) never need it.
        if settings.remote.auth_enabled:
            from core.config import ensure_remote_token

            ensure_remote_token(settings)
        # MEDO Link (M9): restore persisted device manifests so a reboot
        # doesn't forget the fleet; their tools re-register immediately.
        from link.registry import LinkRegistry

        link = LinkRegistry(registry, settings.memory.db_path)
        restored = await asyncio.to_thread(link.load_persisted)
        if restored:
            console.print(f"[dim]MEDO Link: {restored} device manifest(s) restored.[/dim]")
        remote = RemoteServer(settings, router, sm, persist_secrets=True,
                              wake_event=wake_event if voice else None,
                              doc_index=doc_index, mcp_manager=mcp_manager,
                              link=link)
        try:
            await remote.start()
        except OSError as exc:
            # Port already bound = a previous MEDO is still running. Without
            # this message the new launch just dies and the user keeps talking
            # to the old build, wondering why nothing they changed works.
            console.print(
                f"[red]Another MEDO is already running (port "
                f"{settings.remote.port} is busy): {exc}[/red]\n"
                f"[yellow]Close the old MEDO window (or re-run run.bat, which "
                f"now replaces it automatically) and try again.[/yellow]"
            )
            return
        console.print(
            f"[dim]Companion API on port {settings.remote.port} — "
            f"watch app + vision sidecar connect here.[/dim]"
        )
        if settings.remote.auth_enabled:
            console.print(
                "[dim]LAN clients need the token from secrets.local.yaml "
                "(remote.token); localhost is exempt.[/dim]"
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
            # Voice pipeline (voice/loop.py); the companion API (if started)
            # serves alongside it.
            try:
                await VoiceLoop(settings, router, sm, announcer, ui, log,
                                wake_event=wake_event).run()
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
        await mcp_manager.stop()


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
