"""The hands-free voice pipeline: wake word -> record -> STT -> route -> TTS.

Extracted verbatim from ``main.py`` (which had grown a ~335-line ``run_voice``)
into a class with one named method per phase:

    run                  the standby/turn loop
    _load_models         wake word + STT + TTS (graceful when a voice is missing)
    _wait_for_wake       standby: score frames until the phrase / a manual wake
    _capture             listen: open the mic (hot-swap aware), record a phrase
    _run_turn            transcribe -> route (sentence-streaming TTS) -> speak
    _speak               synthesize one utterance (mk/en voice pick) and play it
    _play_interruptible  playback while watching the mic for barge-in

Pure mechanical refactor — behavior, log lines, and timings are unchanged.

Blocking audio/model calls run in worker threads so the asyncio loop (and the
companion API, if serving) stays responsive. After a reply that asks for
confirmation, we record again *without* the wake word so "yes" works
naturally. Every stage is timed into a :class:`~core.metrics.TurnTimings`.

``wake_event`` (set by ``POST /wake``) is an escape hatch: while waiting for
the wake word we also poll it, so the HUD can start a turn without the phrase
— which is how you break out of STANDING BY when the pretrained wake word
doesn't match what you say.

The heavy engines (sounddevice, openwakeword, faster-whisper, piper) are
imported lazily by the voice modules themselves, so importing this module —
and constructing :class:`VoiceLoop` — is cheap and dependency-free (tests).
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from typing import Any

from rich.console import Console

from core.config import Settings
from core.events import AssistantState, StateMachine
from core.metrics import LatencyLog, TurnTimings
from core.router import Router
from ui.console import ConsoleUI
from voice.audio import (
    Microphone,
    frame_rms,
    normalize_peak,
    record_until_silence,
    resolve_input_device,
)
from voice.stt import Transcriber
from voice.tts import EdgeTTS, TextToSpeech, contains_cyrillic, drain_sentences
from voice.wakeword import WakeWord
from voice.wakeword import display_phrase as _wake_display

logger = logging.getLogger("voice")
console = Console()


class VoiceLoop:
    """Owns one voice session; see the module docstring for the phase map."""

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

    def __init__(
        self,
        settings: Settings,
        router: Router,
        sm: StateMachine,
        announcer: Any,           # main.Announcer — anything with a .speak slot
        ui: ConsoleUI,
        log: LatencyLog,
        wake_event: threading.Event | None = None,
        modes: Any = None,
    ) -> None:
        self._settings = settings
        self._router = router
        self._sm = sm
        self._announcer = announcer
        self._ui = ui
        self._log = log
        self._wake_event = wake_event
        # Shared runtime toggles (continuous conversation, interpreter mode).
        # Defaults to a fresh SessionModes so the loop works without one.
        if modes is None:
            from core.modes import SessionModes

            modes = SessionModes(continuous=settings.conversation.continuous)
        self._modes = modes
        # Loaded by run() -> _load_models(); None until then.
        self._wakeword: WakeWord | None = None
        self._stt: Transcriber | None = None
        self._tts: TextToSpeech | None = None
        self._edge: EdgeTTS | None = None
        # Set by the interruptible player when the user barges in mid-reply;
        # the turn loop reads it to skip the wake word and listen immediately.
        self._barged_in = False

    # --- model loading -------------------------------------------------------

    def _load_models(self) -> None:
        # Plain ASCII print, no rich Live/status spinner: with stdout redirected
        # to a file (service/log launches) the spinner's buffer flush dies on
        # cp1252 encoding and takes ALL of voice mode down with it.
        console.print("[dim]loading voice models...[/dim]")
        self._wakeword = WakeWord(self._settings.wakeword)
        self._stt = Transcriber(self._settings.stt)
        # A missing/broken Piper voice shouldn't sink the whole session —
        # degrade to showing replies without speaking them (the HUD renders them).
        try:
            self._tts = TextToSpeech(self._settings.tts)
        except Exception as exc:
            self._tts = None
            console.print(
                f"[yellow]TTS voice unavailable ({exc}); replies will be shown, "
                f"not spoken. Run run.bat to fetch the Piper voice.[/yellow]"
            )
        # Macedonian neural voice (edge-tts, online) for Cyrillic replies; Piper
        # remains the offline voice and the fallback when the network is down.
        self._edge = None
        if self._settings.tts.multilingual:
            try:
                self._edge = EdgeTTS(self._settings.tts.mk_voice)
            except Exception as exc:
                console.print(f"[dim]Macedonian voice unavailable ({exc}); "
                              f"Cyrillic replies will use the English voice.[/dim]")

    # --- standby -------------------------------------------------------------

    def _wait_for_wake(self, mic: Microphone, opened_device) -> str:
        """Block until something should end the wait; returns why.

        ``"go"``     — wake word heard, or a manual /wake / barge-in carried over.
        ``"reopen"`` — the input device changed in settings (HUD mic picker), or
                       a higher-priority device in the configured chain came or
                       went (e.g. Bluetooth headphones connected) — the caller
                       must reopen the mic on the newly-resolved device.

        NB: wake_event is NOT cleared on entry — a /wake or /interrupt fired
        during the previous turn should carry over and start listening now.
        """
        wake = self._wakeword
        assert wake is not None
        wake.reset()
        # Report the peak wake score + mic level every few seconds so it's obvious
        # whether the mic is even hearing you and how close the phrase gets to the
        # threshold — the two things that keep it "stuck on STANDING BY".
        peak_score = peak_rms = 0.0
        last_report = time.monotonic()
        frames = 0
        while True:
            if self._settings.audio.input_device != opened_device:
                logger.info("input device changed — reopening the microphone")
                return "reopen"
            frames += 1
            if frames % 25 == 0:  # ~2 s: hot-swap when the chain resolves elsewhere
                try:
                    if resolve_input_device(self._settings.audio.input_device) != mic.resolved_index:
                        logger.info("a preferred microphone (dis)connected — switching")
                        return "reopen"
                except Exception:
                    pass  # nothing usable right now — keep the mic we have
            if self._wake_event is not None and self._wake_event.is_set():
                self._wake_event.clear()
                console.print("[dim](woken from the HUD)[/dim]")
                return "go"
            frame = mic.read_frame()               # ~80 ms/frame, so /wake lands fast
            score = wake.predict(frame)
            if score >= wake.threshold:
                logger.info("wake word detected (score %.2f)", score)
                return "go"
            peak_score = max(peak_score, score)
            peak_rms = max(peak_rms, frame_rms(frame))
            now = time.monotonic()
            if now - last_report >= 4.0:
                logger.info("listening for wake word — peak score %.2f (need %.2f), mic level %.3f",
                            peak_score, wake.threshold, peak_rms)
                if peak_rms < 0.004:
                    console.print("[yellow](microphone seems silent — check the input "
                                  "device/mute, or pick a mic in the HUD CONFIG tab)[/yellow]")
                peak_score = peak_rms = 0.0
                last_report = now

    # --- barge-in ------------------------------------------------------------

    def _play_interruptible(self, wav, sr: int) -> bool:
        """Play a reply while watching the mic; True if the user barged in.

        Interrupts on any of: the HUD/watch interrupt signal (``wake_event``),
        the wake phrase at a playback-lowered threshold, or sustained mic
        energy well above the reply's own speaker-leakage baseline (measured
        live during the first frames of playback). Falls back to blocking
        playback if the mic can't be opened.
        """
        import sounddevice as sd

        wake = self._wakeword
        assert wake is not None
        sd.play(wav, samplerate=sr, device=self._settings.audio.output_device)
        stream = sd.get_stream()
        interrupted = False
        why = ""
        try:
            wake.reset()
            baseline: float | None = None   # EMA of mic RMS incl. TTS leakage
            loud_run = 0
            frames = 0
            with Microphone(self._settings.audio.sample_rate,
                            device=self._settings.audio.input_device) as m:
                while stream.active:
                    frame = m.read_frame()  # 80 ms cadence paces this loop
                    frames += 1
                    if self._wake_event is not None and self._wake_event.is_set():
                        self._wake_event.clear()
                        interrupted, why = True, "interrupt signal"
                        break
                    if wake.predict(frame) >= self.BARGE_WAKE_THRESHOLD:
                        interrupted, why = True, "wake word over playback"
                        break
                    rms = frame_rms(frame)
                    if baseline is None:
                        baseline = rms
                    elif frames <= 6 or rms < baseline * 1.5:
                        # Track the reply's own loudness only while nothing
                        # shouts over it, so a talking user can't raise the bar.
                        baseline += 0.2 * (rms - baseline)
                    loud = rms >= max(baseline * self.BARGE_RMS_RATIO, self.BARGE_MIN_RMS)
                    # Ignore the first frames: the baseline is still settling.
                    loud_run = loud_run + 1 if (loud and frames > 6) else 0
                    if loud_run >= self.BARGE_HOLD_FRAMES:
                        interrupted, why = True, "voice over playback"
                        break
        except Exception:  # mic busy/unavailable — degrade to plain playback
            logger.debug("barge-in watcher failed", exc_info=True)
            sd.wait()
            return False
        finally:
            wake.reset()
        if interrupted:
            sd.stop()
            logger.info("barge-in (%s): reply interrupted, listening", why)
        return interrupted

    # --- speak ---------------------------------------------------------------

    async def _speak(self, text: str) -> float:
        """Synthesize and play (interruptibly); returns synth time (ms).

        Cyrillic replies go to the Macedonian neural voice; anything else (and
        any edge-tts failure — offline, service hiccup) uses local Piper.
        """
        if self._tts is None and self._edge is None:  # no voice — reply shown only
            return 0.0
        t0 = time.perf_counter()
        wav = None
        sr = 0
        if self._edge is not None and contains_cyrillic(text):
            try:
                wav, sr = await self._edge.synthesize(text)
            except Exception:
                logger.warning("edge-tts failed; falling back to Piper", exc_info=True)
                wav = None
        if wav is None or getattr(wav, "size", 0) == 0:
            if self._tts is None:
                return 0.0
            wav, sr = await asyncio.to_thread(self._tts.synthesize, text)
        tts_ms = (time.perf_counter() - t0) * 1000
        if await asyncio.to_thread(self._play_interruptible, wav, sr):
            self._barged_in = True
        return tts_ms

    # --- listen --------------------------------------------------------------

    async def _capture(self, require_wake: bool):
        """Open the mic, optionally wait for the wake word, record a phrase.

        Returns the audio plus the wake→listen latency (ms). Reopens on the new
        device when the HUD mic picker changes settings mid-wait, and falls back
        to the system default when a picked device can't be opened — so a bad
        pick degrades instead of killing voice mode.

        A wake-less capture (a confirmation follow-up, or continuous mode)
        waits ``followup_window_s`` for speech to begin, then returns empty —
        so a silent follow-up window naturally falls back to standby.
        """
        start_timeout = (self._settings.conversation.followup_window_s
                         if not require_wake else 6.0)
        while True:
            device = self._settings.audio.input_device
            mic = Microphone(self._settings.audio.sample_rate, device=device)
            try:
                mic.open()
            except Exception as exc:
                if device is None:
                    raise  # even the system default is broken — that's fatal
                logger.warning("mic %r failed (%s) — falling back to system default",
                               device, exc)
                self._settings.audio.input_device = None
                continue
            try:
                if require_wake:
                    why = await asyncio.to_thread(self._wait_for_wake, mic, device)
                    if why == "reopen":
                        continue  # device changed under us — reopen on the new one
                t_wake = time.perf_counter()
                await self._sm.transition(AssistantState.LISTENING)
                wake_to_listen_ms = (time.perf_counter() - t_wake) * 1000
                audio = await asyncio.to_thread(
                    record_until_silence,
                    mic,
                    silence_threshold=self._settings.audio.silence_threshold,
                    silence_duration_s=self._settings.audio.silence_duration_s,
                    start_timeout_s=start_timeout,
                )
                return audio, wake_to_listen_ms
            finally:
                mic.close()

    # --- interpreter mode: transcribe -> translate -> speak ------------------

    _LANG_NAME = {"en": "English", "mk": "Macedonian"}
    # Spoken in either language, this leaves interpreter mode (routed normally
    # so the session_mode skill flips the flag off).
    _INTERP_EXIT = re.compile(
        r"\b(?:stop|exit|end|quit)\s+(?:interpreting|translating|interpreter)\b"
        r"|\binterpreter\s+(?:mode\s+)?off\b"
        r"|\bпрекини\s+(?:со\s+)?(?:преведување|толкување)\b"
        r"|\bстоп\s+преведување\b",
        re.IGNORECASE,
    )

    async def _interpret_turn(self, audio) -> bool:
        """Transcribe one utterance; translate + speak it, or exit the mode.

        Returns True to keep interpreting (listen wake-less for the next line),
        False when an exit phrase was heard and routed (back to standby).

        Source language is whatever Whisper detects; MEDO speaks the OTHER of
        the configured pair. Both the source line and the translation go to the
        HUD as a CAPTION event. Bypasses the router except for the exit phrase.
        """
        from core.events import Event, EventType

        assert self._stt is not None
        await self._sm.transition(AssistantState.THINKING)
        text, lang = await asyncio.to_thread(
            self._stt.transcribe_with_language, normalize_peak(audio))
        if not text.strip():
            return True
        if self._INTERP_EXIT.search(text):
            result = await self._router.route(text)   # flips the mode off
            await self._sm.transition(AssistantState.SPEAKING)
            await self._speak(result.speech)
            return False
        a, b = self._modes.interpreter_langs
        # Detected language decides direction; anything unexpected is treated as
        # the first-of-pair so we still translate to the other.
        src = lang if lang in (a, b) else a
        dst = b if src == a else a
        self._ui.transcript(f"[{src}] {text}")
        translated = await self._translate(text, src, dst)
        await self._sm.bus.emit(Event(EventType.CAPTION, {
            "src": src, "src_text": text, "dst": dst, "dst_text": translated,
        }))
        await self._sm.transition(AssistantState.SPEAKING)
        await self._speak(translated)
        return True

    async def _translate(self, text: str, src: str, dst: str) -> str:
        """Translate ``text`` from ``src`` to ``dst`` via the active LLM."""
        model = self._router.model
        if not model:
            return text  # no model — echo rather than drop the line
        src_name = self._LANG_NAME.get(src, src)
        dst_name = self._LANG_NAME.get(dst, dst)
        messages = [
            {"role": "system", "content": (
                f"You are a translation engine. Translate the user's {src_name} "
                f"text into {dst_name}. Output ONLY the translation — no quotes, "
                f"no notes, no transliteration, nothing else.")},
            {"role": "user", "content": text},
        ]
        try:
            reply = await self._router.llm.chat(model, messages)
        except Exception:
            logger.warning("interpreter translation failed", exc_info=True)
            return text
        return (reply.get("content") or "").strip() or text

    # --- one turn: transcribe -> route -> speak ------------------------------

    async def _run_turn(self, audio, wake_to_listen_ms: float, require_wake: bool) -> bool:
        """Handle one recorded utterance; returns next turn's ``require_wake``."""
        assert self._stt is not None
        await self._sm.transition(AssistantState.THINKING)
        t0 = time.perf_counter()
        # Quiet mics (webcam/onboard) record speech peaking at a few percent;
        # normalizing before STT noticeably improves Whisper on those clips.
        text = await asyncio.to_thread(self._stt.transcribe, normalize_peak(audio))
        stt_ms = (time.perf_counter() - t0) * 1000
        if not text.strip():
            console.print("[dim](couldn't make that out)[/dim]")
            return True
        self._ui.transcript(text)

        # Sentence-streaming TTS: LLM content chunks arrive via on_delta,
        # complete sentences are queued, and a speaker task voices them
        # WHILE the rest of the reply is still generating — first audio
        # after the first sentence, not after the whole reply. Fast-path
        # and tool answers never stream (no deltas) and speak as before.
        stream_buf = {"text": ""}
        streamed = {"count": 0}
        stream_q: asyncio.Queue = asyncio.Queue()

        def on_delta(chunk: str) -> None:
            stream_buf["text"] += chunk
            sentences, stream_buf["text"] = drain_sentences(stream_buf["text"])
            for s in sentences:
                streamed["count"] += 1
                stream_q.put_nowait(s)

        async def stream_speaker() -> None:
            while True:
                sentence = await stream_q.get()
                if sentence is None:
                    return
                if self._barged_in:
                    continue  # user cut in — drop the remaining sentences
                await self._sm.transition(AssistantState.SPEAKING)
                await self._speak(sentence)

        speaker_task = asyncio.create_task(stream_speaker())
        try:
            result = await self._router.route(text, on_delta=on_delta)
        finally:
            tail = stream_buf["text"].strip()
            if streamed["count"] and tail:
                streamed["count"] += 1
                stream_q.put_nowait(tail)  # the last, unterminated sentence
            stream_q.put_nowait(None)
            await speaker_task

        tts_ms = None
        if streamed["count"] == 0:  # nothing streamed: speak the reply whole
            await self._sm.transition(AssistantState.SPEAKING)
            tts_ms = await self._speak(result.speech)
        self._log.record(TurnTimings(
            path=result.path.value,
            wake_to_listen_ms=wake_to_listen_ms if require_wake else None,
            stt_ms=stt_ms, route_ms=result.latency_ms, tts_ms=tts_ms,
        ))
        self._ui.turn(result, self._log.turns[-1])
        # Listen again right away (no wake word) when MEDO asked "are you
        # sure?", the user just interrupted, or continuous mode is on — in all
        # three cases they clearly want to keep talking.
        next_require_wake = not (
            self._modes.continuous
            or self._router.awaiting_confirmation
            or self._barged_in
        )
        self._barged_in = False
        return next_require_wake

    # --- the loop ------------------------------------------------------------

    async def run(self) -> None:
        """Load models, then loop: standby -> listen -> turn — forever."""
        self._load_models()
        # Timer/reminder announcements should be spoken, not just printed.
        self._announcer.speak = self._speak

        self._ui.banner(
            "voice mode — say “%s”" % _wake_display(self._settings.wakeword.phrase),
            self._router.model,
        )
        require_wake = True
        while True:
            # Interpreter mode listens continuously (no wake word between lines)
            # so it works like a live interpreter until you say "stop".
            interpreting = self._modes.interpreter
            cap_require_wake = require_wake and not interpreting
            if cap_require_wake:
                await self._sm.transition(AssistantState.IDLE)
            audio, wake_to_listen_ms = await self._capture(cap_require_wake)
            if audio.size == 0:
                if interpreting:
                    continue  # silent gap — keep the interpreter open
                if require_wake:
                    console.print("[dim](heard nothing — back to sleep)[/dim]")
                require_wake = True
                continue

            # One failed turn must NEVER kill voice mode: an exception here used
            # to propagate out of the loop, leaving the HUD stuck on "PROCESSING"
            # forever (state frozen in THINKING, nothing consuming wake/interrupt)
            # while typing kept working — the classic "voice is broken" state.
            try:
                if interpreting:
                    still = await self._interpret_turn(audio)
                    require_wake = not still  # exited -> back to standby
                else:
                    require_wake = await self._run_turn(
                        audio, wake_to_listen_ms, require_wake)
            except (KeyboardInterrupt, asyncio.CancelledError):
                raise
            except Exception:
                logger.exception("voice turn failed; back to standby")
                console.print("[red]That one failed — say the wake word to try again.[/red]")
                require_wake = True
                self._barged_in = False
