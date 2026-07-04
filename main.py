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
    facts = FactsStore(settings.memory.db_path)
    shots = PROJECT_ROOT / "screenshots"

    registry.register(DateTimeSkill())
    registry.register(TimerSkill(announcer, reminder_store))
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
    ui: ConsoleUI, log: LatencyLog, wake_event: threading.Event | None = None,
) -> None:
    """Hands-free loop: wake word -> record -> STT -> route -> TTS.

    Blocking audio/model calls run in worker threads so the asyncio loop (and the
    companion API, if serving) stays responsive. After a reply that asks for
    confirmation, we record again *without* the wake word so "yes" works
    naturally. Every stage is timed into a :class:`TurnTimings`.

    ``wake_event`` (set by ``POST /wake``) is an escape hatch: while waiting for
    the wake word we also poll it, so the HUD can start a turn without the phrase
    — which is how you break out of STANDING BY when the pretrained wake word
    doesn't match what you say.
    """
    import time

    from voice.audio import (
        Microphone,
        frame_rms,
        normalize_peak,
        record_until_silence,
        resolve_input_device,
    )
    from voice.stt import Transcriber
    from voice.tts import TextToSpeech
    from voice.wakeword import WakeWord

    vlog = logging.getLogger("voice")

    # Plain ASCII print, no rich Live/status spinner: with stdout redirected to
    # a file (service/log launches) the spinner's buffer flush dies on cp1252
    # encoding and takes ALL of voice mode down with it.
    console.print("[dim]loading voice models...[/dim]")
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
    def wait_for_wake(mic: Microphone, opened_device) -> str:
        """Block until something should end the wait; returns why.

        ``"go"``     — wake word heard, or a manual /wake / barge-in carried over.
        ``"reopen"`` — the input device changed in settings (HUD mic picker), or
                       a higher-priority device in the configured chain came or
                       went (e.g. Bluetooth headphones connected) — the caller
                       must reopen the mic on the newly-resolved device.

        NB: wake_event is NOT cleared on entry — a /wake or /interrupt fired
        during the previous turn should carry over and start listening now.
        """
        wake.reset()
        # Report the peak wake score + mic level every few seconds so it's obvious
        # whether the mic is even hearing you and how close the phrase gets to the
        # threshold — the two things that keep it "stuck on STANDING BY".
        peak_score = peak_rms = 0.0
        last_report = time.monotonic()
        frames = 0
        while True:
            if settings.audio.input_device != opened_device:
                vlog.info("input device changed — reopening the microphone")
                return "reopen"
            frames += 1
            if frames % 25 == 0:  # ~2 s: hot-swap when the chain resolves elsewhere
                try:
                    if resolve_input_device(settings.audio.input_device) != mic.resolved_index:
                        vlog.info("a preferred microphone (dis)connected — switching")
                        return "reopen"
                except Exception:
                    pass  # nothing usable right now — keep the mic we have
            if wake_event is not None and wake_event.is_set():
                wake_event.clear()
                console.print("[dim](woken from the HUD)[/dim]")
                return "go"
            frame = mic.read_frame()               # ~80 ms/frame, so /wake lands fast
            score = wake.predict(frame)
            if score >= wake.threshold:
                vlog.info("wake word detected (score %.2f)", score)
                return "go"
            peak_score = max(peak_score, score)
            peak_rms = max(peak_rms, frame_rms(frame))
            now = time.monotonic()
            if now - last_report >= 4.0:
                vlog.info("listening for wake word — peak score %.2f (need %.2f), mic level %.3f",
                          peak_score, wake.threshold, peak_rms)
                if peak_rms < 0.004:
                    console.print("[yellow](microphone seems silent — check the input "
                                  "device/mute, or pick a mic in the HUD CONFIG tab)[/yellow]")
                peak_score = peak_rms = 0.0
                last_report = now

    # Set by the interruptible player when the user barges in mid-reply; the main
    # loop reads it to skip the wake word and listen immediately.
    barge_in = {"hit": False}

    # Barge-in tuning (per the latency/leakage analysis): TTS leaking from the
    # speakers into the open mic crushes the wake score, so 0.4 rarely fires
    # while MEDO talks. During playback we run a LOWER wake threshold (barging
    # in is low-risk — worst case the reply stops), plus an energy gate: mic
    # level holding well above the playback's own leakage baseline means
    # someone is talking over MEDO.
    BARGE_WAKE_THRESHOLD = 0.25   # vs 0.4 when idle
    BARGE_RMS_RATIO = 3.0         # mic level vs playback-leakage baseline
    BARGE_MIN_RMS = 0.02          # absolute floor so silence can't ratio-trip
    BARGE_HOLD_FRAMES = 4         # ~0.3 s sustained before it counts

    def play_interruptible(wav, sr: int) -> bool:
        """Play a reply while watching the mic; True if the user barged in.

        Interrupts on any of: the HUD/watch interrupt signal (``wake_event``),
        the wake phrase at a playback-lowered threshold, or sustained mic
        energy well above the reply's own speaker-leakage baseline (measured
        live during the first frames of playback). Falls back to blocking
        playback if the mic can't be opened.
        """
        import sounddevice as sd

        sd.play(wav, samplerate=sr, device=settings.audio.output_device)
        stream = sd.get_stream()
        interrupted = False
        why = ""
        try:
            wake.reset()
            baseline: float | None = None   # EMA of mic RMS incl. TTS leakage
            loud_run = 0
            frames = 0
            with Microphone(settings.audio.sample_rate,
                            device=settings.audio.input_device) as m:
                while stream.active:
                    frame = m.read_frame()  # 80 ms cadence paces this loop
                    frames += 1
                    if wake_event is not None and wake_event.is_set():
                        wake_event.clear()
                        interrupted, why = True, "interrupt signal"
                        break
                    if wake.predict(frame) >= BARGE_WAKE_THRESHOLD:
                        interrupted, why = True, "wake word over playback"
                        break
                    rms = frame_rms(frame)
                    if baseline is None:
                        baseline = rms
                    elif frames <= 6 or rms < baseline * 1.5:
                        # Track the reply's own loudness only while nothing
                        # shouts over it, so a talking user can't raise the bar.
                        baseline += 0.2 * (rms - baseline)
                    loud = rms >= max(baseline * BARGE_RMS_RATIO, BARGE_MIN_RMS)
                    # Ignore the first frames: the baseline is still settling.
                    loud_run = loud_run + 1 if (loud and frames > 6) else 0
                    if loud_run >= BARGE_HOLD_FRAMES:
                        interrupted, why = True, "voice over playback"
                        break
        except Exception:  # mic busy/unavailable — degrade to plain playback
            logging.getLogger("voice").debug("barge-in watcher failed", exc_info=True)
            sd.wait()
            return False
        finally:
            wake.reset()
        if interrupted:
            sd.stop()
            vlog.info("barge-in (%s): reply interrupted, listening", why)
        return interrupted

    async def speak(text: str) -> float:
        """Synthesize and play (interruptibly); returns synth time (ms)."""
        if tts is None:  # TTS unavailable — reply is shown, not spoken.
            return 0.0
        t0 = time.perf_counter()
        wav, sr = await asyncio.to_thread(tts.synthesize, text)
        tts_ms = (time.perf_counter() - t0) * 1000
        if await asyncio.to_thread(play_interruptible, wav, sr):
            barge_in["hit"] = True
        return tts_ms

    async def capture(require_wake: bool) -> tuple[np.ndarray, float]:
        """Open the mic, optionally wait for the wake word, record a phrase.

        Returns the audio plus the wake→listen latency (ms). Reopens on the new
        device when the HUD mic picker changes settings mid-wait, and falls back
        to the system default when a picked device can't be opened — so a bad
        pick degrades instead of killing voice mode.
        """
        while True:
            device = settings.audio.input_device
            mic = Microphone(settings.audio.sample_rate, device=device)
            try:
                mic.open()
            except Exception as exc:
                if device is None:
                    raise  # even the system default is broken — that's fatal
                vlog.warning("mic %r failed (%s) — falling back to system default",
                             device, exc)
                settings.audio.input_device = None
                continue
            try:
                if require_wake:
                    why = await asyncio.to_thread(wait_for_wake, mic, device)
                    if why == "reopen":
                        continue  # device changed under us — reopen on the new one
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
        # Quiet mics (webcam/onboard) record speech peaking at a few percent;
        # normalizing before STT noticeably improves Whisper on those clips.
        text = await asyncio.to_thread(stt.transcribe, normalize_peak(audio))
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
        # Listen again immediately (no wake word) when MEDO asked "are you sure?"
        # or when the user just interrupted the reply — they clearly want to talk.
        require_wake = not (router.awaiting_confirmation or barge_in["hit"])
        barge_in["hit"] = False


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

    # Preload the local model in the background: a cold 30B costs ~27 s on the
    # first question otherwise, which users read as "it doesn't answer".
    if settings.llm.provider == "ollama" and router.model:
        asyncio.create_task(llm.warmup(router.model))

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
        remote = RemoteServer(settings, router, sm, persist_secrets=True,
                              wake_event=wake_event if voice else None)
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
                await run_voice(settings, router, sm, announcer, ui, log, wake_event=wake_event)
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
