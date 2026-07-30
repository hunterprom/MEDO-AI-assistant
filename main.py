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
import re
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
    QuitSkill,
    ScreenshotSkill,
    SystemInfoSkill,
    VolumeSkill,
)
from skills.timers import TimerSkill
from skills.vision_skill import SeeCameraSkill, SeeScreenSkill
from skills.weather import WeatherSkill
from skills.webfetch import WebFetchSkill
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


def _use_os_trust_store() -> None:
    """Make Python's TLS verify against the OPERATING-SYSTEM trust store.

    On managed/corporate networks the HTTPS connection to Microsoft's edge-tts
    voices is TLS-intercepted by a proxy whose CA lives in the OS/browser trust
    store but NOT in Python's bundled ``certifi`` list. Python then rejects it
    with "unable to get local issuer certificate", edge-tts fails, and every
    non-English reply falls back to the English Piper voice — reading Macedonian,
    Russian, Japanese, etc. with English phonemes (gibberish). ``truststore``
    points Python at the SAME certificates the browser trusts (the secure fix —
    it still verifies every connection, it just uses the OS's CAs). Best-effort:
    if the package is missing we log and carry on with certifi.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
        logging.getLogger("main").info("TLS: verifying against the OS trust store")
    except Exception as exc:  # never let a TLS-config nicety stop startup
        logging.getLogger("main").debug(
            "truststore unavailable (%s); using certifi bundle", exc)


def _start_overlay(settings: Settings):
    """Spawn the desktop presence sphere (``ui/overlay.py``) as its own process.

    A separate process on purpose: Tk wants to own a real main loop, and a UI
    crash must never take the assistant down with it. It talks to MEDO only over
    HTTP (the HUD's /events for state, the companion API for click-to-talk), so
    it degrades to a dim sphere when those aren't up. Returns the handle, or
    None when disabled/unavailable, so the caller can stop it on shutdown.
    """
    if not settings.ui.overlay.enabled:
        return None
    import subprocess

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "ui.overlay"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logging.getLogger("main").info("desktop sphere started (pid %s)", proc.pid)
        return proc
    except Exception as exc:          # no display, no Tk, whatever — never fatal
        logging.getLogger("main").warning("desktop sphere didn't start: %s", exc)
        return None


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


#: Long-lived background tasks are kept here so the event loop can't GC them
#: mid-flight (the create_task footgun), and each logs its own failure instead
#: of dying silently ("Task exception was never retrieved" only at GC).
_BG_TASKS: set = set()


def _spawn(coro, what: str):
    """Fire-and-forget a background task that LOGS if it dies, and is retained."""
    task = asyncio.ensure_future(coro)
    _BG_TASKS.add(task)

    def _done(t: "asyncio.Task") -> None:
        _BG_TASKS.discard(t)
        if not t.cancelled():
            exc = t.exception()
            if exc is not None:
                logging.getLogger(__name__).warning(
                    "background task %s failed: %s", what, exc, exc_info=exc)

    task.add_done_callback(_done)
    return task


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
    modes=None,
    browser_think=None,
    expert=None,
    synthesize_council=None,
    compose=None,
    plan=None,
) -> SkillRegistry:
    """Register every skill. Order sets fast-path precedence on overlaps.

    Timers/notes come before power/system so "remind me … to sleep" and "note
    about memory" aren't stolen by the broad ``sleep`` / ``memory`` patterns.
    Web skills come last (broad "search for …"). ``summarize`` (LLM) powers the
    fast-path web-search summary; None => it returns raw results.
    ``briefing_rewrite`` is the LLM pass that turns briefing sections into one
    flowing spoken paragraph; None => the briefing speaks its raw sections.
    ``browser_think`` is the LLM pass that picks the next action in a
    multi-step web task; None => that skill declines instead of guessing.
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
    # "say that again" — replays the last reply verbatim from ConversationMemory
    # (a small LLM would re-generate and drift). Early so its phrases can't be
    # stolen; it reads context['last_reply'], which the router supplies.
    from skills.repeat_skill import RepeatSkill

    registry.register(RepeatSkill())

    # "Close yourself" / "quit MEDO": actually end the assistant session (the LLM
    # used to just narrate a shutdown that never happened). Registered early so
    # its self-directed phrases beat PowerSkill's bare "shut down" (the machine)
    # and the window/app-close skills.
    registry.register(QuitSkill())

    # Self-programming: MEDO proposes changes to its OWN code (core/self_dev.py),
    # built in an isolated worktree and gated by the tests. Registered ONLY when
    # opted in (self_dev.enabled), and it SURFACES only in Lion mode (its match()
    # checks settings.mode.lion) — you enter the Lion profile to let MEDO work on
    # its own code. Registered early so "fix your code" wins over broader patterns.
    if settings.self_dev.enabled:
        from core.self_dev import SelfDevEngine
        from skills.self_dev_skill import SelfDevSkill

        registry.register(SelfDevSkill(
            SelfDevEngine(settings.self_dev, PROJECT_ROOT), announcer, settings))
    # Session-mode toggles (continuous conversation / interpreter). Registered
    # early so its phrases can't be stolen; flips the shared SessionModes the
    # voice loop reads. A no-op holder when modes weren't provided (tests).
    if modes is not None:
        from skills.modes_skill import ModesSkill

        registry.register(ModesSkill(modes))
        # Dictation mode: everything heard is written down until "stop
        # dictation". Registered with the other mode toggles so its phrases
        # ("write this down") can't be stolen by the note or file skills.
        from skills.dictate import DictateSkill

        registry.register(DictateSkill(settings, modes))
    # MEDO Lion Mode: registered first so "lion mode on" can never be taken
    # for anything else — the one command that must always be reachable.
    from skills.lion import LionModeSkill

    registry.register(LionModeSkill(settings))
    # Lion mode's defensive-security skills — read-only, local, advisory. They
    # surface themselves only while mode.lion is on (their match() checks it),
    # so registering them always is harmless. They borrow the council's LLM
    # helper for the prose in their advice; None => they fall back to facts.
    from skills.security import build_security_skills

    for _sec in build_security_skills(settings, whitelist, explain=expert):
        registry.register(_sec)
    # Resistor colour-band decoder — deterministic maker arithmetic. Before the
    # council/circuit block so "what does brown black red gold mean" claims the
    # fast path instead of falling to the electrical-engineer prose (or the LLM,
    # which swaps the multiplier and tolerance bands).
    from skills.resistor import ResistorSkill

    registry.register(ResistorSkill())
    # The specialist council + the wiring helper that borrows its electrical
    # engineer. Before the broad web/search skills, whose "how do I ..." and
    # "ask ..." patterns would otherwise swallow them.
    if settings.council.enabled:
        from skills.circuit import CircuitSkill
        from skills.council import (
            AskSpecialistSkill,
            ConveneCouncilSkill,
            CouncilRosterSkill,
        )

        # "second opinion" / "are you sure?" — the council red-teams MEDO's OWN
        # last answer and returns a deterministic, fail-safe verdict. Registered
        # first so its specific patterns win over the broader council/LLM path.
        if settings.council.second_opinion:
            from skills.second_opinion import SecondOpinionSkill

            registry.register(SecondOpinionSkill(settings, expert))
        registry.register(ConveneCouncilSkill(settings, expert, synthesize_council))
        registry.register(AskSpecialistSkill(settings, expert))
        # "who's on the council" / "do you have a lawyer" — discovery, no LLM.
        registry.register(CouncilRosterSkill(settings))
        registry.register(CircuitSkill(settings, expert, expert))
    # Morning briefing (M8): chains weather/news/reminders/facts; registered
    # early so "brief me" can't be stolen by broader patterns.
    from skills.briefing import BriefingSkill

    registry.register(BriefingSkill(
        settings.briefing.sections, weather_skill, news_skill,
        reminders=reminder_store, facts=facts, rewrite=briefing_rewrite,
    ))
    registry.register(NotesSkill(notes))
    # Project organization — create projects, add/complete tasks, status.
    from core.projects import ProjectStore
    from skills.projects_skill import ProjectsSkill

    registry.register(ProjectsSkill(ProjectStore(settings.memory.db_path), plan))
    # Long-term facts. Recall/forget register before remember so "what do you
    # remember" is answered, never stored.
    registry.register(RecallFactsSkill(facts, settings.memory.max_facts))
    registry.register(ForgetFactSkill(facts))
    registry.register(RememberFactSkill(facts))
    # Window actions before apps: "close the window" is not "close <app>".
    registry.register(WindowActionSkill())
    # "open notepad AND type this in notepad" has to beat AppsSkill, which
    # matches the leading "open notepad" and stops at launching it — dropping
    # the typing half of the request silently (live transcript: MEDO answered
    # "Opening notepad." and wrote nothing). WriteInApp only claims utterances
    # that ALSO carry a write verb naming a configured app, so a bare
    # "open notepad" still falls straight through to AppsSkill below.
    from skills.file_edit import WriteInAppSkill

    registry.register(WriteInAppSkill(apps_table))
    registry.register(AppsSkill(apps_table))
    # Websites right after apps: "open chrome" stays an app launch, while
    # "open tinkercad.com" (dotted) opens the browser — before FilesSkill so
    # a domain never reads as a filename.
    from skills.web_open import OpenWebsiteSkill
    from skills.sites import PlaySkill, SiteSearchSkill

    # The controlled browser (Playwright). One session shared by the two
    # browser skills AND used as the opener for the two open/search skills, so
    # "search X on youtube" then "click the first video" act on one window.
    # Off => opener stays webbrowser.open and today's behaviour is unchanged.
    browser_session = None
    site_opener = None
    if settings.browser.enabled:
        from skills.browser import BrowserSession, browser_opener

        browser_session = BrowserSession(settings.browser)
        if settings.browser.route_opens:
            site_opener = browser_opener(browser_session)

    # Site-scoped search BEFORE the plain opener: "search cats on youtube" is
    # strictly more specific than "open youtube", and its patterns all demand
    # both a search verb and a known site, so a bare "open youtube" still falls
    # through to OpenWebsiteSkill below.
    # "play X on youtube" before the search skills: playing is a search plus a
    # click, and landing on a results page is the wrong answer to "play me".
    registry.register(PlaySkill(opener=site_opener, session=browser_session,
                                extra_sites=settings.skills.get("sites")))
    registry.register(SiteSearchSkill(opener=site_opener,
                                      extra_sites=settings.skills.get("sites")))
    registry.register(OpenWebsiteSkill(opener=site_opener))
    # Page interaction. Before TypeTextSkill/PressKeysSkill so "type medo into
    # the search box" reaches the page rather than the focused window, while a
    # bare "type hello" (no field) still falls through to TypeTextSkill.
    # Control before agent: "what's on the page" is a read, not a task.
    if browser_session is not None:
        from skills.browser import BrowserAgentSkill, BrowserControlSkill

        registry.register(BrowserControlSkill(settings.browser, browser_session))
        registry.register(BrowserAgentSkill(settings.browser, browser_session,
                                            think=browser_think))
    registry.register(SeeCameraSkill(settings))
    registry.register(SeeScreenSkill(settings))
    # M11 deictic pointing: "what is this?" crops around the mouse cursor.
    from skills.vision_skill import PointAtSkill

    registry.register(PointAtSkill(settings))
    # Screen agent (experimental): "do this for me: …" clicks/types on screen.
    # controls_pc + requires_confirmation; after the sensing vision skills so
    # "what's on my screen" stays a description, not an action.
    from skills.screen_agent import ScreenAgentSkill

    registry.register(ScreenAgentSkill(settings))
    # Documents RAG before FilesSkill/WebSearch so "search my documents for X"
    # isn't stolen by the filename search or the broad web "search for …".
    if doc_index is not None:
        from skills.documents import DocumentsSkill

        registry.register(DocumentsSkill(doc_index))
    # "import this file" — takes ONE file into the managed store and pushes it
    # through that same index (or, for a picture, its vision description).
    # Registered before the file skills so "import ~/Downloads/spec.pdf" isn't
    # read as an open/find. Works without doc_index: it still copies and
    # describes, and says plainly that it couldn't index.
    from skills.importer import ImportFileSkill

    registry.register(ImportFileSkill(settings, whitelist, doc_index))
    # Editing before finding: "edit notes.md" is a specific action, while
    # FilesSkill's verbs (find/search/open) don't overlap with it. Both are
    # confined to the same whitelist.
    from skills.file_edit import FileEditSkill, OpenInEditorSkill

    # (WriteInAppSkill is registered further up, ahead of AppsSkill — it has to
    # win "open notepad and type this in notepad". It still comes before the two
    # below: otherwise FileEditSkill reads "напиши го ова во нотепад" as a write
    # to a FILE called "нотепад", and TypeTextSkill's bare "type …" would
    # swallow the app name into the text.)
    registry.register(FileEditSkill(whitelist))
    registry.register(OpenInEditorSkill(whitelist, apps_table))
    registry.register(FilesSkill(whitelist))
    # Make a document/presentation about a topic — MEDO writes the content (via
    # `compose`, the LLM) and renders it to Word/PowerPoint/HTML (core/documents).
    from skills.documents_gen import MakeDocumentSkill

    registry.register(MakeDocumentSkill(compose))   # saves to ~/Documents/MEDO
    # Build a standalone app from a description (core/app_builder.py). Opt-in, in
    # its own folder — the outward sibling of self-dev.
    if settings.app_builder.enabled:
        from core.app_builder import AppBuilder
        from skills.app_builder_skill import MakeAppSkill

        registry.register(MakeAppSkill(AppBuilder(settings.app_builder), announcer))
    # Application discovery. LocateApp answers "do I have X" (it only looks);
    # InstallApp is gated on a spoken yes AFTER naming the resolved package.
    from skills.appfinder import (
        InstallAppSkill,
        LocateAppSkill,
        OpenDiscoveredAppSkill,
    )

    registry.register(LocateAppSkill())
    registry.register(InstallAppSkill())
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

    _plugin_store = None
    if settings.security.plugin_approval:
        from security.plugins import PluginApprovalStore

        _plugin_store = PluginApprovalStore()
    load_plugins(registry, {
        "settings": settings,
        "announcer": announcer,
        "summarize": summarize,
        "reminders": reminder_store,
        "doc_index": doc_index,
    }, require_approval=settings.security.plugin_approval,
       approval_store=_plugin_store)

    # "open <anything installed>" — deliberately last of the openers: its
    # pattern is broad by necessity, so it only ever sees what the configured
    # launcher, the website opener and the file skills did not want.
    registry.register(OpenDiscoveredAppSkill())

    # Web skills (network; degrade gracefully offline; broad patterns last).
    registry.register(weather_skill)
    registry.register(news_skill)
    # Reading a page BEFORE searching for one: web_fetch's patterns all demand
    # either a URL or an explicit "page/страница" noun, so they are narrow
    # enough to sit ahead of web_search's greedy "search … for …" — and
    # "summarize <url>" must not be read as a search for the word "summarize".
    if settings.web_fetch.enabled:
        registry.register(WebFetchSkill(settings.web_fetch, summarize))
    registry.register(WebSearchSkill(summarize))

    # Software connectors (control local apps by command; NOT MCP). Off by
    # default — flipping software.enabled on registers each connector action as a
    # skill, gated by the policy engine like everything else. Guarded so a missing
    # desktop dep never breaks startup.
    if settings.software.enabled:
        try:
            from security.policy import PolicyEngine
            from software.mechanisms import Mechanisms
            from software.registry import register_software
            from software.win32_backend import Win32Backend

            # PathWhitelist is already imported at module scope (used above).
            policy = PolicyEngine(settings.security, settings.safety,
                                  PathWhitelist(settings.safety.whitelist_dirs))
            mech = Mechanisms(Win32Backend(), policy,
                              exe_paths=settings.software.exe_paths)
            n = register_software(registry, settings, mech)
            logging.getLogger(__name__).info("software connectors: %d actions", n)
        except Exception:
            logging.getLogger(__name__).warning(
                "software connectors failed to load", exc_info=True)

    # Last, once every skill exists: the user's own trigger phrases from
    # config.yaml (skills.triggers). They are appended to the very same
    # `patterns` list the built-ins use, so nothing downstream — the router,
    # the registry, the tests — has to know a phrase came from config.
    from core.triggers import apply_triggers

    apply_triggers(registry, settings.skills.get("triggers", {}))
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
        console.print(
            f"Routing — FAST: {s[RoutePath.FAST]}  "
            f"SEMANTIC: {s.get(RoutePath.SEMANTIC, 0)}  LLM: {s[RoutePath.LLM]}")
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


def _migrate_db_name(db_path: str) -> None:
    """One-time rename jarvis.db -> the configured db (default medo.db).

    Best-effort: only when the new file doesn't exist yet and the old one does,
    so a user's facts/metrics carry across the jarvis->medo rebrand untouched.
    """
    from pathlib import Path

    new = Path(db_path)
    old = new.with_name("jarvis.db")
    if old.name == new.name or new.exists() or not old.exists():
        return
    try:
        old.rename(new)
        logging.getLogger("main").info("migrated %s -> %s", old.name, new.name)
    except OSError:
        logging.getLogger("main").debug("db migration skipped", exc_info=True)


async def async_main(once: str | None, serve: bool, voice: bool, hud: bool) -> None:
    _trust_os_certificates()
    settings = load_settings()
    # Overlay any online API key / model saved via the HUD (git-ignored file), so
    # an online provider chosen last session is restored without touching config.yaml.
    apply_local_secrets(settings)
    # Rebrand carry-over: the assistant DB was 'jarvis.db'. Rename an existing
    # one to the new default so remembered facts / metrics survive the rename.
    _migrate_db_name(settings.memory.db_path)
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

    async def compose(instruction: str) -> str:
        """Write document/presentation content as Markdown (for MakeDocumentSkill).

        Returns "" when there's no model, so the skill declines cleanly rather
        than saving an empty file.
        """
        if not router.model:
            return ""
        prompt = [
            {"role": "system", "content": (
                "You are a writing assistant. Produce ONLY the requested content "
                "as clean Markdown: '## ' for section or slide titles, short "
                "paragraphs, and '- ' bullet lists. No code fences, no preamble, "
                "no closing remarks.")},
            {"role": "user", "content": instruction},
        ]
        try:
            message = await llm.chat(router.model, prompt)
        except LLMUnavailableError:
            return ""
        return (message.get("content") or "").strip()

    async def plan(goal: str) -> list[str]:
        """Break a goal into concrete tasks (for ProjectsSkill). [] with no model."""
        if not router.model:
            return []
        prompt = [
            {"role": "system", "content": (
                "Break the user's goal into 5 to 10 concrete, ordered, actionable "
                "tasks. Reply with ONLY the tasks, one per line, no numbering, no "
                "headings, no preamble.")},
            {"role": "user", "content": goal},
        ]
        try:
            message = await llm.chat(router.model, prompt)
        except LLMUnavailableError:
            return []
        lines = (message.get("content") or "").splitlines()
        tasks = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip()
                 for ln in lines]
        return [t for t in tasks if t][:12]

    async def expert(system: str, user: str) -> str:
        """One specialist round-trip: a role prompt plus the question.

        Shared by the council and the wiring helper. Runs on the council brain
        (a fast local model when the selected brain is a CLI agent — see
        Router.council_brain), so convening several experts in parallel doesn't
        spawn a pile of slow CLI processes. Returns "" when no model is
        available, so a caller degrades instead of raising at the user.
        """
        client, model = router.council_brain()
        if not model:
            return ""
        try:
            message = await client.chat(model, [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
        except LLMUnavailableError:
            return ""
        return (message.get("content") or "").strip()

    async def synthesize_council(question: str, notes: str,
                                 macedonian: bool = False) -> str:
        """MEDO's architect pass: several specialist notes -> one spoken answer."""
        language = "Macedonian" if macedonian else "English"
        system = (
            f"You are {settings.personality.name}, the architect coordinating a "
            f"panel of specialists. Combine their notes into ONE spoken answer "
            f"in {language}, two to four sentences, plain text with no markdown. "
            f"Keep every concrete number. Where they disagree, say so plainly "
            f"and say which you would follow and why. Invent nothing."
        )
        user = f"Question: {question}\n\nSpecialist notes:\n{notes}"
        return await expert(system, user)

    async def browser_think(prompt: str) -> str:
        """One LLM pass: page elements + task -> the next browser action (JSON).

        Text-only, so the main brain drives it — the vision model never sees a
        web page. Temperature stays at the provider default; the prompt already
        pins the output shape and a wrong guess costs a click.
        """
        if not router.model:
            return ""
        try:
            message = await llm.chat(router.model, [{"role": "user", "content": prompt}])
        except LLMUnavailableError:
            return ""
        return (message.get("content") or "").strip()

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

    # Shared runtime toggles (continuous conversation, interpreter mode): one
    # instance seen by the modes skill, the voice loop, and the companion API.
    from core.modes import SessionModes

    modes = SessionModes(continuous=settings.conversation.continuous)

    registry = build_registry(settings, announcer, summarize, reminders, doc_index,
                              briefing_rewrite=briefing_rewrite, modes=modes,
                              browser_think=browser_think, expert=expert,
                              synthesize_council=synthesize_council, compose=compose,
                              plan=plan)

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

    # Event webhooks: POST a JSON payload to external URLs when MEDO wakes,
    # hears, routes, or replies. Subscribed here so it sees every bus event;
    # fire-and-forget, so a dead endpoint never stalls a turn.
    webhook_manager = None
    if settings.webhooks:
        from core.webhooks import WebhookManager

        webhook_manager = WebhookManager(settings.webhooks)
        live = webhook_manager.subscribe(bus)
        if live:
            console.print(f"[dim]Webhooks: {live} live.[/dim]")

    log = LatencyLog()
    # State chips are shown in voice mode; in the REPL they'd clutter the prompt.
    ui = ConsoleUI(console, bus, settings.personality.name, show_states=voice)

    if once is not None:
        # Even a one-shot must unwind the MCP servers it connected above —
        # otherwise every `--once` orphans their subprocesses (not reaped on
        # Windows), leaking a process per invocation.
        try:
            await respond(router, sm, ui, log, once)
        finally:
            await mcp_manager.stop()
        return

    # Preload the local model in the background: a cold 30B costs ~27 s on the
    # first question otherwise, which users read as "it doesn't answer".
    if settings.llm.provider == "ollama" and router.model:
        _spawn(llm.warmup(router.model), "model warmup")
    # Index the user's documents in the background so "what do my documents
    # say about X" has something to search (incremental; skips unchanged files).
    if doc_index is not None:
        _spawn(asyncio.to_thread(doc_index.reindex), "document reindex")
    # Proactive routines (morning briefing etc.): answers are announced —
    # spoken in voice mode, printed otherwise, visible in the HUD either way.
    if settings.routines:
        from core.routines import RoutineScheduler

        _spawn(RoutineScheduler(settings.routines, router, announcer).run_forever(),
               "routine scheduler")

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
                              link=link, modes=modes)
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
            await mcp_manager.stop()   # unwind the servers we connected above
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
        hud_server = HudServer(settings, bus, registry=registry)
        await hud_server.start()
        console.print(
            f"[bold cyan]HUD:[/bold cyan] open "
            f"[underline]http://localhost:{settings.hud.port}[/underline] in a browser."
        )

    # The corner presence sphere — visible while you work in other apps.
    overlay_proc = _start_overlay(settings)

    try:
        if voice:
            # Voice pipeline (voice/loop.py); the companion API (if started)
            # serves alongside it.
            try:
                await VoiceLoop(settings, router, sm, announcer, ui, log,
                                wake_event=wake_event, modes=modes).run()
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
        _slog = logging.getLogger(__name__)

        async def _quietly(what: str, coro) -> None:
            # Each shutdown step guarded on its own: one failing (e.g. aiohttp
            # cleanup erroring on an in-flight SSE stream) must NOT skip the rest
            # — the MCP stdio children (not reaped on Windows) and the browser
            # profile lock both have to be released regardless.
            try:
                await coro
            except Exception:
                _slog.warning("shutdown: %s failed", what, exc_info=True)

        if overlay_proc is not None:
            try:
                overlay_proc.terminate()      # the sphere follows MEDO down
            except Exception:
                _slog.warning("shutdown: overlay terminate failed", exc_info=True)
        # Cancel the long-lived background tasks (routine scheduler run_forever,
        # model warmup, doc reindex) so they can't fire against a half-torn-down
        # router or leave "Task was destroyed but it is pending" noise.
        pending = list(_BG_TASKS)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if remote is not None:
            await _quietly("remote.stop", remote.stop())
        if hud_server is not None:
            await _quietly("hud.stop", hud_server.stop())
        if webhook_manager is not None:
            await _quietly("webhooks.stop", webhook_manager.stop())
        await _quietly("mcp.stop", mcp_manager.stop())
        # Close the controlled browser if one was ever launched, so Chrome
        # doesn't outlive MEDO holding a lock on the profile directory.
        browser_skill = registry.get("browser_control")
        if browser_skill is not None:
            await _quietly("browser.close", browser_skill.session.close())


def main() -> None:
    # Before anything opens a TLS connection: use the OS trust store so edge-tts
    # (the multilingual neural voice) works behind a corporate CA / TLS proxy.
    _use_os_trust_store()
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
