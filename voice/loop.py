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
import collections
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
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
from core.speech_text import FenceFilter, strip_markup
from voice.tts import EdgeTTS, TextToSpeech, contains_cyrillic, drain_sentences
from voice.wakeword import WakeWord
from voice.wakeword import display_phrase as _wake_display

logger = logging.getLogger("voice")
console = Console()

#: Hard ceiling on the edge-tts network synth before falling back to local
#: Piper — a hung Microsoft socket must not freeze the turn indefinitely.
_EDGE_TTS_TIMEOUT_S = 20.0

#: Sentences synthesized ahead of what is currently being spoken. One is
#: enough to cover the gap (synthesis is far faster than playback for every
#: voice here), and keeping it small bounds both the RAM a long reply holds
#: and the work thrown away when the user interrupts.
_SYNTH_LOOKAHEAD = 1


class VoiceLoop:
    """Owns one voice session; see the module docstring for the phase map."""

    # Barge-in tuning (per the latency/leakage analysis): TTS leaking from the
    # speakers into the open mic crushes the wake score, so 0.4 rarely fires
    # while MEDO talks. During playback we run a LOWER wake threshold (barging
    # in is low-risk — worst case the reply stops), plus an energy gate: mic
    # level holding well above the playback's own leakage baseline means
    # someone is talking over MEDO.
    # Barge-in tuning lives in audio config now: the right values depend on the
    # room, the speakers and the mic, so they must be tunable without a code
    # edit. These stay as documented fallbacks.
    BARGE_WAKE_THRESHOLD = 0.25   # vs 0.4 when idle
    BARGE_RMS_RATIO = 2.0         # mic level vs playback-leakage baseline
    BARGE_MIN_RMS = 0.02          # absolute floor so silence can't ratio-trip
    BARGE_HOLD_FRAMES = 3         # ~0.24 s sustained before it counts

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
        # Serializes speech: the turn pipeline and the timer/reminder announcer
        # both reach _speak on the same event loop, and each drives the SINGLE
        # global sounddevice stream and the SHARED (non-thread-safe) openWakeWord
        # model inside _play_interruptible. Without this, a timer firing while
        # MEDO is speaking runs two player threads at once — corrupting the wake
        # model's ring buffers and truncating one of the two replies.
        self._speak_lock = asyncio.Lock()
        # Cross-THREAD guard for the single wake model. _speak_lock only
        # serializes the event-loop _speak callers; it does NOT cover the standby
        # loop (_wait_for_wake), which runs in its own worker thread and never
        # takes it. So a timer/reminder announcement firing DURING standby drives
        # _play_interruptible's wake.reset()/predict() on the same non-thread-safe
        # model from a second thread — corrupting its ring buffers so wake
        # detection breaks afterwards. The player sets this flag to claim the
        # model; the standby loop pauses its own predicting while it's set.
        self._model_busy = threading.Event()
        # 'Let me think' fillers for slow LLM answers (core/filler.py). LLM path
        # only — armed on the router's on_llm_start hook in _run_turn.
        from core.filler import Filler

        self._filler = Filler(settings.filler,
                              active=settings.active_languages(),
                              primary=settings.primary_language())
        # Wake-word diagnosis (off unless wakeword.debug_capture): records every
        # activation's score + audio so a threshold can be set from real data.
        from core.config import PROJECT_ROOT
        from voice.wake_debug import WakeCaptureLog

        self._wake_capture = WakeCaptureLog(
            PROJECT_ROOT / "logs" / "wake_captures",
            enabled=settings.wakeword.debug_capture,
            sample_rate=settings.audio.sample_rate,
        )
        #: Language of the utterance being handled, so _speak picks its voice.
        self._turn_language: str | None = None

    # --- model loading -------------------------------------------------------

    def _load_models(self) -> None:
        # Plain ASCII print, no rich Live/status spinner: with stdout redirected
        # to a file (service/log launches) the spinner's buffer flush dies on
        # cp1252 encoding and takes ALL of voice mode down with it.
        console.print("[dim]loading voice models...[/dim]")
        self._wakeword = WakeWord(self._settings.wakeword)
        self._stt = Transcriber(self._settings.stt, self._settings.languages)
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
        # Neural voices (edge-tts, online) for non-Piper languages; Piper remains
        # the offline English voice and the fallback when the network is down.
        # The edge fallback voice is the PRIMARY active language's voice (S3),
        # not a hardcoded Macedonian one — the per-utterance voice still follows
        # the detected language via EdgeTTS.voice_for.
        self._edge = None
        if self._settings.tts.multilingual:
            from core import languages as _languages

            _primary = self._settings.primary_language()
            _fallback_voice = (_languages.voice_for(_primary)
                               or self._settings.tts.mk_voice)
            try:
                self._edge = EdgeTTS(_fallback_voice)
            except Exception as exc:
                console.print(f"[dim]Macedonian voice unavailable ({exc}); "
                              f"Cyrillic replies will use the English voice.[/dim]")
        self._voicebox = self._build_voicebox()

    def _build_voicebox(self) -> Any:
        """The provider chain that speaks a reply — see voice/providers.py.

        Order is the whole design: local voices first for the fifteen languages
        that have one, the cloud only for a language that has none (Macedonian),
        and the old direct-Piper call last as a safety net for English.
        """
        from pathlib import Path

        from core.config import PROJECT_ROOT
        from voice.providers import (
            EdgeProvider,
            LegacyPiperProvider,
            SherpaProvider,
            VoiceBox,
        )

        cfg = self._settings.tts
        chain: list[Any] = []
        if cfg.local_voices:
            root = Path(cfg.voices_dir).expanduser()
            if not root.is_absolute():
                root = PROJECT_ROOT / root
            chain.append(SherpaProvider(
                root, max_loaded=cfg.max_loaded_voices,
                num_threads=cfg.num_threads,
                auto_download=cfg.auto_download,
                use_fallback_voices=cfg.offline_fallback_voices))
        chain.append(EdgeProvider(self._edge, allow_cloud=cfg.allow_cloud))
        chain.append(LegacyPiperProvider(self._tts))
        box = VoiceBox(chain)
        logger.info("voice providers: %s",
                    ", ".join(p.name for p in box.providers))
        return box

    # --- standby -------------------------------------------------------------

    def _confirm_wake(self, audio) -> bool:
        """True if the buffered trigger audio really contains the wake phrase.

        Fail-OPEN: if STT itself errors, allow the wake — a broken transcriber
        must never make MEDO unwakeable, and the sustained-frame gate already
        rejected the obvious transients.
        """
        from voice.wakeword import wake_phrase_confirmed

        try:
            text = self._stt.transcribe(normalize_peak(audio))
        except Exception:
            logger.debug("STT wake-confirm failed; allowing the wake", exc_info=True)
            return True
        # Confirm against the phrase the LOADED model listens for (not the
        # configured one), so the bundled fallback isn't rejected forever.
        phrase = getattr(self._wakeword, "phrase", None) or self._settings.wakeword.phrase
        ok = wake_phrase_confirmed(text, phrase)
        if not ok:
            logger.info("wake discarded by STT confirm — heard %r, not the phrase",
                        (text or "")[:60])
        return ok

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
        # An announcement speaking right now owns the model on another thread;
        # wait for it before we touch the model, so the two never race.
        while self._model_busy.is_set():
            time.sleep(0.05)
        wake.reset()
        # Report the peak wake score + mic level every few seconds so it's obvious
        # whether the mic is even hearing you and how close the phrase gets to the
        # threshold — the two things that keep it "stuck on STANDING BY".
        peak_score = peak_rms = 0.0
        last_report = time.monotonic()
        frames = 0
        run = 0                 # consecutive over-threshold frames (anti-noise)
        need = getattr(wake, "trigger_frames", 1)
        # STT confirmation: re-transcribe the trigger buffer and require the wake
        # phrase before actually waking — the strongest loud-noise defence.
        confirm = self._settings.wakeword.stt_confirm and self._stt is not None
        # Rolling ~1.5 s of frames, kept when we need the audio that triggered
        # an activation (for the STT confirm or a debug capture).
        recent = (collections.deque(maxlen=20)
                  if (self._wake_capture.enabled or confirm) else None)
        while True:
            if self._model_busy.is_set():
                # A timer/reminder announcement is speaking and owns the model;
                # pause predicting until it's done rather than corrupt it.
                time.sleep(0.05)
                continue
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
            if recent is not None:
                recent.append(frame)
            score = wake.predict(frame)
            # Require the phrase to hold above threshold for `need` frames: a
            # transient noise spikes for one frame, a spoken "hey medo" doesn't.
            run = run + 1 if score >= wake.threshold else 0
            if run >= need:
                audio = np.concatenate(list(recent)) if recent else frame
                if self._wake_capture.enabled:
                    # Save what actually triggered this (diagnosis mode).
                    self._wake_capture.record(score, frame_rms(frame), audio)
                # Verify the WORDS: a loud transient scores high but transcribes
                # to nothing, so it's discarded here and MEDO keeps sleeping.
                if confirm and not self._confirm_wake(audio):
                    run = 0
                    wake.reset()
                    continue
                logger.info("wake word detected (score %.2f, %d frames)", score, run)
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

    @staticmethod
    def watch_for_barge(frames, wake, audio_cfg, wake_event=None) -> str | None:
        """Decide whether to interrupt playback, over an iterable of mic frames.

        The interrupt triggers depend on ``audio_cfg.barge_mode``:

        * ``"wake"``  — ONLY the wake word ("medo") over the reply, or the
          explicit HUD/watch interrupt signal. Random words and background
          noise are ignored. (Default.)
        * ``"voice"`` — additionally, sustained mic energy well above the
          reply's own speaker-leakage baseline (measured live over the first
          frames). The older "talk over MEDO" behaviour.
        * ``"off"``   — only the explicit interrupt signal; the mic never
          interrupts.

        Pure of audio hardware so it can be unit-tested: ``frames`` is any
        iterable of PCM frames and ``wake`` any object with
        ``predict(frame) -> float``. Returns the reason string, or ``None`` if
        the frames ran out with no barge-in.
        """
        mode = getattr(audio_cfg, "barge_mode", "wake")
        wake_enabled = mode in ("wake", "voice")
        voice_enabled = mode == "voice"
        # A barge-in needs its OWN sustained-frame count, NOT the idle
        # trigger_frames: standby can be single-frame twitchy without harm, but
        # that same twitchiness while MEDO speaks lets its speaker-leakage / a
        # stray blip cut the reply off. barge_wake_frames keeps interruption
        # deliberate even when waking is hyper-sensitive.
        need = max(1, getattr(audio_cfg, "barge_wake_frames", 3))
        baseline: float | None = None   # EMA of mic RMS incl. TTS leakage
        loud_run = 0
        wake_run = 0
        n = 0
        for frame in frames:
            n += 1
            if wake_event is not None and wake_event.is_set():
                wake_event.clear()
                return "interrupt signal"
            if wake_enabled:
                # Same sustained-phrase rule as standby: one loud frame that
                # happens to score high must not count as "medo".
                wake_run = (wake_run + 1
                            if wake.predict(frame) >= audio_cfg.barge_wake_threshold
                            else 0)
                if wake_run >= need:
                    return "wake word over playback"
            if not voice_enabled:
                continue          # mode "wake"/"off": energy never interrupts
            rms = frame_rms(frame)
            if baseline is None:
                baseline = rms
            elif n <= 6 or rms < baseline * 1.5:
                # Track the reply's own loudness only while nothing shouts over
                # it, so a talking user can't raise the bar.
                baseline += 0.2 * (rms - baseline)
            loud = rms >= max(baseline * audio_cfg.barge_rms_ratio,
                              audio_cfg.barge_min_rms)
            # Ignore the first frames: the baseline is still settling.
            loud_run = loud_run + 1 if (loud and n > 6) else 0
            if loud_run >= audio_cfg.barge_hold_frames:
                return "voice over playback"
        return None

    def _play_interruptible(self, wav, sr: int) -> bool:
        """Play a reply while watching the mic; True if the user barged in.

        The barge triggers themselves live in :meth:`watch_for_barge`; this
        wires it to the live mic and speaker. ``barge_mode: "off"`` still lets
        the explicit HUD/watch signal through. Falls back to blocking playback
        if the mic can't be opened.
        """
        import sounddevice as sd

        wake = self._wakeword
        assert wake is not None
        audio_cfg = self._settings.audio
        # Claim the wake model so the standby loop (another thread) pauses its
        # own predicting for the duration — see __init__/_model_busy.
        self._model_busy.set()
        try:
            sd.play(wav, samplerate=sr, device=self._settings.audio.output_device)
            stream = sd.get_stream()
            why: str | None = None
            try:
                wake.reset()
                with Microphone(self._settings.audio.sample_rate,
                                device=self._settings.audio.input_device) as m:
                    def live_frames():
                        while stream.active:
                            yield m.read_frame()   # 80 ms cadence paces this loop
                    why = self.watch_for_barge(
                        live_frames(), wake, audio_cfg, self._wake_event)
            except Exception:  # mic busy/unavailable — degrade to plain playback
                logger.debug("barge-in watcher failed", exc_info=True)
                sd.wait()
                return False
            finally:
                wake.reset()
            if why:
                sd.stop()
                logger.info("barge-in (%s): reply interrupted, listening", why)
            return why is not None
        finally:
            self._model_busy.clear()

    # --- speak ---------------------------------------------------------------

    async def _speak(self, text: str, language: str | None = None) -> float:
        """Synthesize and play (interruptibly); returns synth time (ms).

        The voice comes from the language of the turn — the code Whisper
        reported for what the user just said — so MEDO answers in the language
        it was addressed in. Falls back to the script test when no language was
        detected (an announcement, a typed command), and to local Piper on any
        edge-tts failure, so going offline costs the accent, not the voice.
        """
        if not self._voicebox.providers:      # no voice at all — reply shown only
            return 0.0
        # One speaker at a time: a timer announcement must not drive the global
        # audio stream / wake model concurrently with a turn's reply.
        async with self._speak_lock:
            return await self._synthesize_and_play(text, language)

    async def _synthesize_and_play(self, text: str, language: str | None) -> float:
        t0 = time.perf_counter()
        audio = await self._synthesize(text, language)
        tts_ms = (time.perf_counter() - t0) * 1000
        if audio is not None:
            await self._play(*audio)
        return tts_ms

    async def _synthesize(self, text: str,
                          language: str | None) -> tuple[Any, int] | None:
        """Text -> (waveform, sample_rate), or None when nothing can speak it.

        Split out from playback so the streaming path can synthesize the NEXT
        sentence while the current one is still being heard. Doing both in one
        call meant every sentence boundary cost a full synthesis of silence —
        150-300 ms for a Piper voice, and 2.8 s for Japanese on Kokoro.
        """
        from core import languages

        spoken_lang = language or self._turn_language
        if spoken_lang is None:
            # Nothing detected this turn (typed input, an announcement, a
            # skill's own words). Fall back to the script of what we are about
            # to say: speaking Japanese text with the English voice makes it
            # read the characters out one by one.
            spoken_lang = languages.detect_script(text)
        # One call, whatever engine ends up serving it. The chain handles the
        # local-first ordering, the per-language voice, and falling through a
        # provider that fails; the timeout still bounds the cloud one, because
        # a Microsoft socket that accepts and then never streams is a hang, not
        # an exception, and no fallback fires on a hang.
        try:
            wav, sr = await asyncio.wait_for(
                self._voicebox.synthesize(text, spoken_lang),
                timeout=_EDGE_TTS_TIMEOUT_S)
        except Exception:
            logger.warning("synthesis failed for %r", spoken_lang, exc_info=True)
            return None
        if wav is None or getattr(wav, "size", 0) == 0:
            # Nothing could speak this language. SHOW the reply — never hand it
            # to a voice trained on another language, which is the exact
            # "reads Japanese out character by character" bug.
            logger.warning("no voice could speak %r — reply shown, not spoken",
                           spoken_lang)
            console.print(f"[dim](no voice available for {spoken_lang} — "
                          f"reply shown, not spoken)[/dim]")
            return None
        return wav, sr

    async def _play(self, wav, sr: int) -> None:
        """Play a waveform, watching the mic for a barge-in."""
        if await asyncio.to_thread(self._play_interruptible, wav, sr):
            self._barged_in = True

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
        conv = self._settings.conversation
        start_timeout = conv.followup_window_s if not require_wake else 6.0
        # Dictation is composed, not commanded: people pause mid-sentence, so
        # it gets a longer silence gate and a much higher ceiling than a
        # one-line request.
        if self._modes.dictating:
            silence_s = conv.dictation_silence_s
            max_s = conv.dictation_max_utterance_s
        else:
            silence_s = self._settings.audio.silence_duration_s
            max_s = self._settings.audio.max_utterance_s
        while True:
            device = self._settings.audio.input_device
            mic = Microphone(self._settings.audio.sample_rate, device=device)
            try:
                mic.open()
            except Exception as exc:
                # open() may have constructed the PortAudio stream before
                # start() raised — close it so a repeated device-fallback loop
                # doesn't leak a stream on every pass.
                try:
                    mic.close()
                except Exception:
                    pass
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
                    silence_duration_s=silence_s,
                    max_seconds=max_s,
                    start_timeout_s=start_timeout,
                )
                return audio, wake_to_listen_ms
            finally:
                mic.close()

    # --- dictation mode: transcribe -> append to file ------------------------

    async def _dictate_turn(self, audio) -> bool:
        """Transcribe one utterance and append it to the dictation file.

        Returns True to keep dictating. Nothing is routed and nothing is
        spoken back except the exit confirmation — reading every sentence back
        would make composing anything longer than a note unbearable.
        """
        from skills.dictate import STOP_DICTATION

        text = (await asyncio.to_thread(
            self._stt.transcribe, normalize_peak(audio)) or "").strip()
        if not text:
            return True
        if STOP_DICTATION.search(text):
            path = Path(self._modes.dictation_path)
            self._modes.dictating = False
            self._modes.dictation_path = ""
            await self._sm.transition(AssistantState.SPEAKING)
            speak_mk = contains_cyrillic(text)
            await self._speak(f"Запишано во {path.name}." if speak_mk
                              else f"Saved to {path.name}.")
            return False
        from core.events import Event, EventType

        path = Path(self._modes.dictation_path)

        def _append() -> None:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(text.rstrip() + "\n")

        try:
            # to_thread: this runs once per spoken line, and the loop it would
            # block is the one streaming the next sentence to the speakers.
            await asyncio.to_thread(_append)
        except OSError:
            logger.exception("dictation write failed")
            await self._speak("I couldn't write that down.")
            return False
        await self._sm.bus.emit(Event(EventType.TRANSCRIPT, text))
        console.print(f"[dim]dictated:[/dim] {text}")
        return True

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
        # The reply is spoken in the language of the question.
        self._turn_language = lang
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
        # Voice the translation in the DESTINATION language, not self._turn_language
        # (the SOURCE code Whisper detected) — otherwise the Cyrillic reply is read
        # by the English-only Piper voice.
        await self._speak(translated, language=dst)
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
        # transcribe_with_language, not transcribe: the detected language is
        # what picks the reply voice and tells the brain which language to
        # answer in. Taking the text alone here is what made MEDO read a
        # Japanese reply aloud in the English voice, one character at a time.
        text, lang = await asyncio.to_thread(
            self._stt.transcribe_with_language, normalize_peak(audio))
        self._turn_language = lang
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
        # A code fence spans many chunks, so it can only be caught with state:
        # without this the speaker voices the inside of the block long before
        # the closing ``` arrives (MEDO once read out a whole JSON tool schema).
        fence = FenceFilter()

        def on_delta(chunk: str) -> None:
            stream_buf["text"] += fence.feed(chunk)
            sentences, stream_buf["text"] = drain_sentences(stream_buf["text"])
            for s in sentences:
                spoken = strip_markup(s)
                if not spoken:
                    continue          # the sentence was pure markup
                streamed["count"] += 1
                stream_q.put_nowait(spoken)

        async def stream_speaker() -> None:
            """Voice queued sentences, synthesizing one AHEAD of playback.

            Two stages, not one. Synthesizing and playing in the same step
            meant the next sentence only started synthesizing after the
            current one had finished being heard, so every sentence boundary
            was a silent gap the length of a full synthesis — 150-300 ms on a
            Piper voice and ~2.8 s on Japanese. Now the gap is only whatever
            synthesis takes LONGER than the audio already playing, which for
            every language but Japanese is nothing at all.

            The queue is bounded: run ahead without limit and a long reply
            synthesizes itself entirely into RAM, and every buffered sentence
            is one more that has to be thrown away on a barge-in.
            """
            audio_q: asyncio.Queue = asyncio.Queue(maxsize=_SYNTH_LOOKAHEAD)

            async def synthesize() -> None:
                while True:
                    sentence = await stream_q.get()
                    if sentence is None or self._barged_in:
                        await audio_q.put(None)
                        return
                    audio = await self._synthesize(sentence, None)
                    if audio is not None:
                        await audio_q.put(audio)

            async def play() -> None:
                while True:
                    audio = await audio_q.get()
                    if audio is None:
                        return
                    if self._barged_in:
                        continue    # user cut in — drop what's still queued
                    await self._sm.transition(AssistantState.SPEAKING)
                    # The lock still serializes against the filler and timer
                    # announcements; it is held per SENTENCE, not for the whole
                    # reply, so a barge-in doesn't wait for the queue to drain.
                    async with self._speak_lock:
                        await self._play(*audio)

            await asyncio.gather(synthesize(), play())

        speaker_task = asyncio.create_task(stream_speaker())

        # Filler: bridge dead air on a SLOW LLM answer. Armed ONLY when the
        # router commits to the LLM path (on_llm_start) — the fast path never
        # fires it. A big question gets a quick "let me think"; anything still
        # silent after the delay gets one; a very long wait gets ONE follow-up.
        # It never talks over the reply: _speak is serialized, each step
        # re-checks that nothing has been voiced (streamed["count"]), and when
        # the reply lands we SIGNAL the filler to wind down rather than cancel
        # it — cancelling mid-_speak would release the speak-lock early and let
        # the reply's audio overlap the filler's.
        llm_started = asyncio.Event()
        route_done = asyncio.Event()

        async def _sleep_or_done(seconds: float) -> bool:
            """Sleep, waking early (True) the moment the reply has arrived."""
            try:
                await asyncio.wait_for(route_done.wait(), seconds)
                return True
            except asyncio.TimeoutError:
                return False

        async def filler_task() -> None:
            fill = self._filler
            if fill is None or not fill.enabled:
                return
            # Wait for the LLM commit — but the FAST and SEMANTIC paths never
            # fire on_llm_start, so ALSO wake when the whole turn has already
            # finished, and bail. Otherwise the finally's `await filler`
            # deadlocks: the turn hangs in THINKING, its answer shown in the HUD
            # (via the ROUTED event) but never spoken, and MEDO sits on
            # "processing" forever after answering.
            waiters = [asyncio.ensure_future(llm_started.wait()),
                       asyncio.ensure_future(route_done.wait())]
            try:
                await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for w in waiters:
                    w.cancel()
            if not llm_started.is_set():
                return          # turn finished without an LLM — no filler needed
            if await _sleep_or_done(fill.opening_delay(text)):
                return
            if streamed["count"] or self._barged_in:
                return
            phrase = fill.opening(self._turn_language)
            if phrase:
                await self._speak(phrase)          # finishes; lock released clean
            if await _sleep_or_done(fill.followup_delay):
                return
            if streamed["count"] or self._barged_in:
                return
            phrase = fill.waiting(self._turn_language)
            if phrase:
                await self._speak(phrase)

        filler = asyncio.create_task(filler_task())
        try:
            result = await self._router.route(
                text, context={"language": self._turn_language},
                on_delta=on_delta, on_llm_start=llm_started.set)
        finally:
            route_done.set()      # tell the filler to stop BEFORE its next line
            await filler          # let any in-progress filler audio finish first
            tail = strip_markup(stream_buf["text"] + fence.flush())
            if streamed["count"] and tail:
                streamed["count"] += 1
                stream_q.put_nowait(tail)  # the last, unterminated sentence
            stream_q.put_nowait(None)
            await speaker_task

        tts_ms = None
        # Speak result.speech UNLESS it was already voiced by the streamer.
        # It was voiced only when sentences actually drained (count > 0) AND the
        # router says this speech is the streamed reply. A tool answer (or a
        # confirmation prompt) leaves streamed_reply False, so it is spoken here
        # even if a filler preamble streamed first — otherwise the real answer
        # was silently dropped. A short reply with no sentence terminator never
        # drains (count == 0), so it is still spoken whole.
        already_voiced = streamed["count"] > 0 and result.streamed_reply
        if not already_voiced and result.speech:
            await self._sm.transition(AssistantState.SPEAKING)
            tts_ms = await self._speak(result.speech)
        self._log.record(TurnTimings(
            path=result.path.value,
            wake_to_listen_ms=wake_to_listen_ms if require_wake else None,
            stt_ms=stt_ms, route_ms=result.latency_ms, tts_ms=tts_ms,
        ))
        self._ui.turn(result, self._log.turns[-1])
        # A self-close skill (QuitSkill) asked MEDO to end the session. The
        # farewell has now been spoken, so exit cleanly — reusing the Ctrl-C path
        # so main() unwinds the remote API, HUD and MCP children in its finally.
        # Raise (rather than return) to leave the forever-loop regardless of what
        # require_wake would have been.
        if result.data.get("exit"):
            raise KeyboardInterrupt
        next_require_wake = not self._should_relisten(self._barged_in)
        self._barged_in = False
        # The turn's language voiced its reply above; clear it so a later
        # timer/reminder announcement (or the dictation-exit line) — spoken via
        # _speak with no language — falls back to detect_script(text) instead of
        # this turn's voice, rather than reusing the stale prior-turn language.
        self._turn_language = None
        return next_require_wake

    def _should_relisten(self, barged_in: bool) -> bool:
        """Whether to open the mic again WITHOUT the wake word after this turn.

        Yes when MEDO just asked "are you sure?", asked a one-word tie-break
        ("did you mean the weather, or the news?"), or continuous mode is on —
        the user is clearly mid-conversation. A barge-in is different: with
        wake-word interruption, saying "medo" to cut MEDO off does NOT mean a
        command is coming, so by default we return to standby instead of
        sitting in LISTENING recording the silence (or noise) that follows.
        Set conversation.listen_after_barge to keep the old follow-up window.
        """
        if (self._router.awaiting_confirmation or self._router.awaiting_choice
                or self._modes.continuous):
            return True
        return barged_in and self._settings.conversation.listen_after_barge

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
            dictating = self._modes.dictating
            # Both modes stream: no wake word between lines until you stop.
            cap_require_wake = require_wake and not (interpreting or dictating)
            if cap_require_wake:
                await self._sm.transition(AssistantState.IDLE)
            # _capture handles a mic that won't OPEN (device fallback), but a
            # read error MID-stream (USB/Bluetooth mic unplugged while listening)
            # used to propagate out of run() and kill voice mode for the whole
            # session. Catch it here: return to standby and let the next _capture
            # re-resolve the device, so a reconnect just works.
            try:
                audio, wake_to_listen_ms = await self._capture(cap_require_wake)
            except (KeyboardInterrupt, asyncio.CancelledError):
                raise
            except Exception:
                logger.exception("audio capture failed; back to standby")
                console.print("[red]Lost the microphone — retrying…[/red]")
                require_wake = True
                await asyncio.sleep(0.5)      # don't spin if the device stays gone
                continue
            if audio.size == 0:
                if interpreting or dictating:
                    continue  # silent gap — keep the mode open
                if require_wake:
                    console.print("[dim](heard nothing — back to sleep)[/dim]")
                require_wake = True
                continue

            # One failed turn must NEVER kill voice mode: an exception here used
            # to propagate out of the loop, leaving the HUD stuck on "PROCESSING"
            # forever (state frozen in THINKING, nothing consuming wake/interrupt)
            # while typing kept working — the classic "voice is broken" state.
            try:
                if dictating:
                    still = await self._dictate_turn(audio)
                    require_wake = not still  # exited -> back to standby
                elif interpreting:
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
