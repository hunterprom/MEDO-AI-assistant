"""Typed configuration loaded from ``config.yaml``.

The whole app reads settings through :class:`Settings`. Values come from the YAML
file, but any field can be overridden by an environment variable (e.g.
``MEDO_LLM__DEFAULT_MODEL=qwen2.5:7b``) thanks to pydantic-settings — handy for
tests and CI without editing the file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# Resolved once so every module agrees on where the project root is.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Online API keys live here, NOT in config.yaml (which is committed to git). This
# file is git-ignored; it is written by the HUD's "connect online model" flow and
# read at startup by apply_local_secrets(). Fields mirror LLMConfig.
SECRETS_PATH = PROJECT_ROOT / "secrets.local.yaml"

# Only these LLM fields may be persisted to / loaded from the secrets file.
_SECRET_LLM_FIELDS = (
    "provider",
    "api_key",
    "openai_base_url",
    "anthropic_api_key",
    "anthropic_base_url",
    "default_model",
)


def expand_path(path: str | Path) -> Path:
    """Expand ``~`` and environment variables, returning an absolute path."""
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


class LLMConfig(BaseModel):
    # ollama       — local Ollama server
    # openai       — any OpenAI-compatible cloud API (GPT, Groq, …)
    # anthropic    — the Claude API (api.anthropic.com)
    # claude-code  — the locally installed `claude` CLI agent (Claude Code)
    # codex        — the locally installed `codex` CLI agent (OpenAI Codex)
    provider: Literal["ollama", "openai", "anthropic", "claude-code", "codex"] = "ollama"
    host: str = "http://localhost:11434"
    api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    # Claude API: separate key so switching between GPT and Claude keeps both.
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    # Claude's /v1/messages requires an output cap (also a sane spoken-reply cap).
    anthropic_max_tokens: int = 1024
    # CLI agents: the executable to run (absolute path or on $PATH).
    claude_cmd: str = "claude"
    codex_cmd: str = "codex"
    cli_timeout_s: float = 180.0
    default_model: str | None = None
    fallback_model: str = "llama3.2:3b"
    # Auto tool-brain: the CLI agents (claude-code/codex) can't use MEDO's tools,
    # so a query that needs live/online info (search, "latest…", "who won…") is
    # run on this local Ollama model for that ONE turn — it has web_search and
    # the other tools — then MEDO reverts to your selected brain. "" disables.
    tool_brain_model: str = "qwen3:30b"
    temperature: float = 0.6
    num_ctx: int = 4096
    request_timeout_s: float = 120.0
    # Ollama keep-alive: how long the model stays loaded after a request.
    keep_alive: str = "30m"


class RouterConfig(BaseModel):
    fast_path_enabled: bool = True


class ConversationConfig(BaseModel):
    """Continuous-conversation mode (voice/loop.py + core/modes.py)."""

    # When on, MEDO keeps listening after each reply for a natural follow-up
    # without the wake word. Toggle live by voice ("continuous mode on") or the
    # HUD; this is just the boot default.
    continuous: bool = False
    # How long the mic stays open for a follow-up before falling back to
    # standby on silence.
    followup_window_s: float = 8.0
    # After you interrupt MEDO (say "medo" over its reply), should it keep
    # listening for a follow-up without the wake word? Off by default: cutting
    # MEDO off usually means "stop", not "here comes another command", and the
    # old behaviour left MEDO recording the silence/noise after an interrupt.
    # Turn on to chain "medo — actually, do X" in one breath.
    listen_after_barge: bool = False
    # --- dictation (skills/dictate.py + voice/loop.py) ----------------------
    #: Where dictated text lands when no file is named.
    dictation_dir: str = "~/Documents"
    dictation_file: str = "dictation.md"
    #: Dictation gets a longer leash than a command: people pause mid-sentence
    #: while composing, and a 1.4 s gate would chop every thought in half.
    dictation_silence_s: float = 2.5
    dictation_max_utterance_s: float = 90.0


class PersonalityConfig(BaseModel):
    name: str = "MEDO"
    address_user_as: str = "sir"
    # The Jarvis charm (core/persona.py). style sets the LLM manner fragment
    # AND gates fast-path quips; wit_level is the per-reply quip probability
    # (0 = never, 1 = always; fast path only). quips_language "match" answers
    # Macedonian input with Macedonian quips; or pin "en"/"mk".
    style: Literal["dry_wit", "professional", "minimal"] = "dry_wit"
    wit_level: float = 0.3
    quips_language: str = "match"


class RemoteConfig(BaseModel):
    """Companion API (watch app) — see remote/server.py."""

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8710
    # Bearer-token auth for LAN clients (localhost is always exempt so the HUD
    # keeps its zero-config startup). False restores the old open behavior —
    # documented as unsafe outside a trusted network.
    auth_enabled: bool = True
    # The token itself never lives in config.yaml (committed); it is generated
    # on first serve and persisted to the git-ignored secrets.local.yaml.
    token: str = ""
    # Answer UDP "who is MEDO?" broadcasts so the watch app can find this
    # machine and start pairing without typing an IP (same port, UDP).
    discovery_enabled: bool = True


class PointerConfig(BaseModel):
    """Gesture mouse control (pointer mode) — see vision/engine.py.

    ``enabled`` gates the feature; the runtime toggle always starts OFF and is
    flipped by voice ("pointer on"), the HUD switch, or POST :8731/pointer.
    """

    enabled: bool = True
    sensitivity: float = 2.5            # hand range → screen range amplification
    ema_alpha: float = 0.4              # 0..1 smoothing (higher = snappier)
    click_debounce_ms: int = 600        # min gap between gesture right-clicks
    hold_frames: int = 3                # frames a right-click pose must hold to fire
    scroll_gain: float = 45.0           # hand vertical motion → wheel notches (victory)
    zoom_gain: float = 25.0             # hand vertical motion → ctrl+wheel zoom (rock)
    volume_interval_ms: int = 180       # min gap between volume steps while held
    exit_hold_frames: int = 18          # frames a fist must HOLD to exit (~1.2 s @15fps)
    # Gesture geometry (vision/gestures.py). Finger state is a joint angle, so
    # there is a deliberate dead band between "curled" and "extended": with
    # strict_gestures on, a hand in transit (mid-fist) classifies as unknown
    # instead of flashing through a real pose and firing an action.
    strict_gestures: bool = True        # reject any hand with an in-between finger
    extended_min_deg: float = 160.0     # PIP angle at/above this = finger extended
    curled_max_deg: float = 100.0       # PIP angle at/below this = finger curled
    zoom_min_spread_deg: float = 50.0   # min index<->pinky splay for the rock/zoom pose
    l_shape_tolerance_deg: float = 25.0
    # Window/tab navigation poses (L = next tab, four = taskbar). Held longer
    # and debounced harder than a click: these jump you between windows, so a
    # single misread frame must never fire one.
    # --- click snapping (vision/snap.py) ------------------------------------
    #: A pinch pulls the fingertip down as the thumb comes up, so the click
    #: lands just below the button. Snap presses the nearest clickable element
    #: instead — after showing which one.
    snap_enabled: bool = True
    snap_radius_px: int = 100
    #: Preview delay. The chosen target is outlined on screen and the click
    #: waits this long, so a wrong pick is visible and you can pull back.
    #: 0 = click immediately (still outlines it).
    snap_confirm_ms: int = 250
    #: Distance discount for targets ABOVE the cursor — the drift is downward,
    #: so an equally near target above is the one you were reaching for.
    snap_above_bias_px: float = 20.0
    snap_highlight: bool = True
    nav_hold_frames: int = 5
    nav_debounce_ms: int = 900
    #: pose -> action overrides, e.g. {"open_palm": "switch_window"}. Actions:
    #: move, drag, scroll, zoom, right_click, volume_up, volume_down,
    #: next_tab, switch_window, taskbar, idle.
    pose_actions: dict[str, str] = Field(default_factory=dict)  # how far off 90 deg an L (thumb+index) may be


class GestureRecognitionConfig(BaseModel):
    """Exact-signature gesture matching — see vision/gestures.py.

    Named ``gesture_recognition`` and not ``gestures``: ``vision.gestures`` is
    already the gesture-to-utterance map, and reusing it would silently break
    that block.
    """

    #: Minimum depth-inside-band across a signature's required digits. A finger
    #: sitting exactly on its threshold scores 0, dead-straight scores 1, and a
    #: gesture scores the MINIMUM of the digits it requires — so this is "every
    #: required finger is comfortably inside its band", not an average.
    min_confidence: float = 0.85
    #: Consecutive frames the same signature must hold before it fires.
    hold_frames: int = 3
    #: After firing, the hand must leave the pose before it can fire again.
    #: Off means a held pose repeats every cooldown.
    hysteresis: bool = True


class BenchConfig(BaseModel):
    """Second workbench camera (M10, skills/bench.py) — on-demand only.

    The sidecar opens this camera per request (``GET /bench.jpg``) and
    releases it immediately; it is never streamed and never joins the
    pointer pipeline (privacy + VRAM).
    """

    enabled: bool = False
    camera_index: int = 1


class VisionConfig(BaseModel):
    """Camera hand-gesture control — see vision/gestures.py."""

    enabled: bool = False
    camera_index: int = 0
    stream_port: int = 8731                 # MJPEG stream for the HUD to embed
    flip: bool = True                       # mirror the selfie view
    # Draw the coloured hand skeleton (MediaPipe landmarks) onto the streamed
    # feed? OFF by default: the overlay is drawn on the SAME frame the vision
    # model looks at, so leaving it on makes "what do you see" describe coloured
    # lines on your hands. Gesture tracking still runs either way — this only
    # controls whether the overlay is visible.
    show_hand_tracking: bool = False
    max_fps: int = 15                       # throttle inference on CPU
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    stability_frames: int = 6               # consecutive frames to confirm a gesture
    cooldown_s: float = 2.0                 # min gap before re-firing a gesture
    pointer: PointerConfig = Field(default_factory=PointerConfig)
    gesture_recognition: GestureRecognitionConfig = Field(
        default_factory=GestureRecognitionConfig)
    bench: BenchConfig = Field(default_factory=BenchConfig)
    # Screen agent (skills/screen_agent.py): max actions per task before it
    # stops itself. Gated by the PC-control switch + a spoken confirmation.
    agent_enabled: bool = True
    agent_max_steps: int = 6
    # Recognized gesture -> the utterance routed through the Intent Router.
    gestures: dict[str, str] = Field(
        default_factory=lambda: {
            "thumbs_up": "yes",
            "open_palm": "no",
            "victory": "take a screenshot",
            "point_up": "volume up",
            "three": "volume down",
            "fist": "mute",
            "pinch": "lock the screen",
            "rock": "play music",
        }
    )


class SpheresConfig(BaseModel):
    """The agent cluster-sphere view (a HUD screen) — see ui/web/index.html.

    Purely visual: a sphere per capability domain, a star per registered skill,
    built at page load from the live skill registry (:mod:`core.agents`).
    """

    enabled: bool = True
    #: Each cluster sphere renders as a rotating cosmic-web (nodes + filaments +
    #: dust). ``high`` = full dust shell + glow sprites; ``medium`` = fewer dust
    #: particles; ``low`` = no dust and slower rotation, for weak GPUs. This is
    #: the ceiling — the HUD may still drop lower if it measures a slow frame.
    quality: Literal["low", "medium", "high"] = "high"


class UIConfig(BaseModel):
    """HUD look: theme + composable effect layers. Purely cosmetic — none of
    this changes what a skill does or relaxes a safety gate.

    ``theme`` sets the base skin; ``effects`` are overlays stacked on top of
    it. They compose where it makes sense (scanlines over glass, say), and the
    HUD refuses combinations that would hurt the transcript — an effect layer
    never captures pointer events and never sits over the chat text.
    """

    #: Base skin. ``deep_red`` is Lion mode's identity (see skills/lion.py).
    theme: Literal["arc", "glass", "minimal", "maximal", "deep_red"] = "arc"
    #: Overlays: any of ``scanlines``, ``glitch``, ``glass``, ``grid``. The
    #: browser also persists the user's own pick in localStorage, so this only
    #: seeds a first visit.
    effects: list[str] = Field(default_factory=lambda: ["scanlines"])
    spheres: SpheresConfig = Field(default_factory=SpheresConfig)


class HudConfig(BaseModel):
    """M.E.D.O. web HUD (arc-reactor front end) — see ui/hud.py."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8730
    max_dir_dots: int = 300             # how many folder dots to scatter on the orb
    #: INTERFACE language, picked in the HUD's CONFIG tab. Independent of what
    #: you speak: the language you read a screen in and the language you talk
    #: in are usually different decisions.
    language: str = "en"


class STTConfig(BaseModel):
    engine: str = "faster-whisper"
    model: str = "small"                 # small is noticeably more accurate than base
    device: str = "auto"
    compute_type: str = "auto"
    # null/None = per-utterance auto-detect (enables Macedonian + English).
    language: str | None = "en"
    #: How the spoken language is chosen. "auto" detects per utterance; any
    #: code from core/languages.py forces it — Whisper skips detection, decodes
    #: as that language, and replies come back in it.
    #:
    #: This is the SAME lever as ``language`` above, not a second one: the
    #: validator below folds it into ``language``, which is the only field the
    #: transcriber reads. ``language_mode`` is just the spelling that says out
    #: loud that "auto" is a legal value.
    #:
    #: Force it when auto-detect keeps guessing wrong — one-word commands carry
    #: very little signal, and languages with close neighbours (mk/bg/sr,
    #: es/pt, hi/ur) are where detection actually fails.
    language_mode: str = "auto"
    # With auto-detect on, clamp detection to these languages: Whisper often
    # mistakes spoken Macedonian for Bulgarian/Serbian, which garbles the
    # decode. Outside-the-set detections are re-transcribed forced to the
    # first non-English entry. Empty = no clamp.
    allowed_languages: list[str] = Field(default_factory=list)
    #: Which of core/languages.py MEDO listens for and speaks. Empty => all of
    #: them. Trimming this tightens Whisper's clamp, which is the single
    #: biggest lever on accuracy for languages that resemble each other.
    spoken_languages: list[str] = Field(default_factory=list)
    beam_size: int = 5                   # >1 = beam search; more accurate than greedy
    vad_filter: bool = True              # drop non-speech the recorder let through
    condition_on_previous_text: bool = False  # avoids runaway repeats on short clips
    filter_hallucinations: bool = True   # drop low-confidence/junk Whisper segments
    no_speech_threshold: float = 0.6     # passed through to faster-whisper transcribe()
    # Biases the decoder toward the words this assistant actually hears.
    # Leave null when language auto-detect is on — an English prompt skews
    # Macedonian decodes.
    initial_prompt: str | None = (
        "Commands for a voice assistant: what time is it, open chrome, volume up, "
        "take a note, set a timer, take a screenshot, weather, the news, shut down."
    )

    @model_validator(mode="after")
    def _apply_language_mode(self) -> STTConfig:
        """Fold ``language_mode`` into ``language`` — one lever, one field.

        Whichever of the two the user set, the transcriber sees the result in
        ``language`` and nothing downstream has to know both names exist. An
        unsupported code falls back to auto-detect with a warning rather than
        forcing Whisper into a language MEDO has no voice for.
        """
        from core import languages

        mode = (self.language_mode or "auto").strip().lower()
        if mode in ("", "auto"):
            # Only auto-detect if the older field didn't already force one;
            # "auto" is the default of a field the user may never have touched.
            if "language_mode" in self.model_fields_set:
                self.language = None
            self.language_mode = "auto"
            return self
        if languages.get(mode) is None:
            logging.getLogger(__name__).warning(
                "stt.language_mode=%r is not a language MEDO speaks — using "
                "auto-detect. Supported: %s", mode, ", ".join(languages.codes()))
            self.language_mode, self.language = "auto", None
            return self
        self.language_mode = mode
        self.language = mode
        return self


class TTSConfig(BaseModel):
    engine: str = "piper"
    voice_model: str = ""
    speed: float = 1.0
    # Cyrillic replies are spoken with a Macedonian neural voice via edge-tts
    # (free, online); Piper stays the offline voice for everything else.
    multilingual: bool = True
    mk_voice: str = "mk-MK-MarijaNeural"


class WakeWordConfig(BaseModel):
    engine: str = "openwakeword"
    phrase: str = "hey_jarvis"
    threshold: float = 0.5
    #: Consecutive ~80 ms frames the phrase must stay above ``threshold`` before
    #: MEDO wakes. 1 = the old single-frame trigger, which is twitchy: a clap, a
    #: door, or a stray word spikes the score for one frame and wakes it. A real
    #: "hey medo" SUSTAINS across several frames, so requiring 2–3 is the main
    #: defence against random-noise false wakes — no model retraining needed.
    #: Raise it if MEDO still wakes to noise; lower it (toward 1) if a genuine
    #: "hey medo" is being missed.
    trigger_frames: int = 3
    #: openWakeWord's built-in speech gate (0 = off). Raise toward ~0.5 so only
    #: sounds that are actually SPEECH can wake MEDO — music, bangs and keyboard
    #: clatter are ignored outright. Needs openWakeWord's VAD model available;
    #: if it can't load, MEDO logs a warning and runs without it.
    vad_threshold: float = 0.0
    #: DIAGNOSIS mode (off by default). When on, EVERY wake activation — real or
    #: false — is logged (time, score, audio RMS) and the ~1.5 s buffer that
    #: triggered it is saved to logs/wake_captures/ as a .wav. Turn on, slam
    #: some doors and say the wake word for a day, then run
    #: ``python -m voice.wakeword --report`` to see where the noise scores vs
    #: the real wake word — so the threshold is set from data, not guessed.
    #: The captures are your own voice: logs/ is git-ignored.
    debug_capture: bool = False
    #: STRONGEST loud-noise defence, on by default. After the score crosses the
    #: threshold, MEDO re-transcribes the ~1.5 s that triggered it and only
    #: actually wakes if the wake phrase is really in the audio. A door slam,
    #: music or a clap transcribes to nothing, so it is discarded silently —
    #: this verifies WORDS, not just a score, which is what a loud transient
    #: can't fake. Cost: one short STT pass per trigger (only on triggers). Set
    #: false only if a genuine "hey medo" is being wrongly rejected; then tune
    #: threshold/vad from the --report data instead.
    stt_confirm: bool = True


class AudioConfig(BaseModel):
    # int index, a case-insensitive name substring (e.g. "FHD Webcam") that
    # survives device re-indexing across reboots, or a PRIORITY LIST of those —
    # the first currently-available entry wins and the voice loop hot-swaps
    # (~2 s) when a higher-priority device connects. null = system default.
    input_device: int | str | list[int | str] | None = None
    output_device: int | None = None
    sample_rate: int = 16000
    silence_threshold: float = 0.015
    #: Quiet needed to end an utterance. Short values cut people off mid-thought,
    #: which reads as "it stopped listening" rather than "I paused".
    silence_duration_s: float = 1.4
    #: Hard cap on one utterance. Was 12 s hard-coded, which truncated any
    #: question longer than a sentence or two.
    max_utterance_s: float = 30.0

    # --- barge-in (voice/loop.py) --------------------------------------------
    #: What may interrupt MEDO mid-reply:
    #:   "wake"  — ONLY the wake word ("medo"), or the HUD/watch interrupt
    #:             button. A random word or background noise will not cut it
    #:             off. This is the default.
    #:   "voice" — also any sustained speech/noise over the reply (the older,
    #:             twitchier behaviour). Restore this if you want to talk over
    #:             MEDO without saying its name.
    #:   "off"   — only the explicit HUD/watch interrupt button; the mic never
    #:             interrupts playback.
    barge_mode: Literal["wake", "voice", "off"] = "wake"
    #: Wake-word score that interrupts a reply. Lower than the idle threshold —
    #: you're speaking over MEDO's own voice, so the model scores lower.
    barge_wake_threshold: float = 0.25
    #: How far above MEDO's own speaker-leakage your voice must sit. Measured on
    #: this machine: leakage ~0.011 rms median, 0.035 peak, so 3.0 put the bar
    #: above MEDO's own loud moments and barge-in effectively never fired.
    barge_rms_ratio: float = 2.0
    #: Absolute floor so a silent room can't ratio-trip.
    barge_min_rms: float = 0.02
    #: Frames (~80 ms each) of sustained speech before it counts.
    barge_hold_frames: int = 3


class ModeConfig(BaseModel):
    """Expert PROFILES — presentation + which skills are surfaced. A profile
    is not a safety switch: none of these relax the confirmation gate, the
    path whitelist, or the PC-control switch. See docs/Decisions.md.
    """

    # MEDO LION MODE — the defensive-security profile. Off by default; a
    # runtime toggle that resets on restart (a mode for a task, not a setting).
    # When on it reskins the HUD deep red, shows a LION indicator, and SURFACES
    # a group of read-only, local-machine, advisory security skills (port
    # audit, firewall audit, process/permission explainers, update check).
    #
    # It is DEFENSIVE-ONLY and changes nothing about safety: MEDO still asks
    # before every destructive action, still refuses paths outside the
    # whitelist, and still honours the PC-control switch. It will not scan or
    # touch any other machine, and it refuses to produce exploits, malware,
    # credential-cracking, or ways around its own guards. An "unrestricted
    # mode" is a liability; "surface more, restrict nothing less" is the design.
    lion: bool = False


class SafetyConfig(BaseModel):
    confirm_destructive: bool = True
    whitelist_dirs: list[str] = Field(default_factory=list)
    # Master switch: may MEDO ACT on this computer (type, click, launch apps,
    # open websites, move files, power)? Toggled live from the HUD CONFIG tab
    # (PC CONTROL) and persisted per-machine in secrets.local.yaml. Sensing
    # (vision, weather, questions) is never affected.
    pc_control_enabled: bool = True

    def resolved_whitelist(self) -> list[Path]:
        """Whitelist directories as absolute, expanded paths."""
        return [expand_path(d) for d in self.whitelist_dirs]


class BrowserConfig(BaseModel):
    """Controlled browser — skills/browser.py (Playwright driving real Chrome).

    Separate from ``webbrowser.open``: this is a browser MEDO can *read and
    click*, not just launch. Off until ``pip install playwright`` has been run;
    every browser skill degrades to a spoken "not installed" message, so a
    missing dependency can never break startup.
    """

    enabled: bool = False
    #: Playwright browser channel. "chrome"/"msedge" drive the copy already
    #: installed on the machine (no 150 MB Chromium download, and it looks like
    #: the browser the user knows); "" falls back to Playwright's own Chromium.
    channel: str = "chrome"
    #: Headless hides the window. Default off on purpose — when MEDO clicks
    #: things on your behalf you should be able to watch it happen.
    headless: bool = False
    #: Persistent profile dir (relative to the project root). Persistent so you
    #: log into a site once and MEDO is still logged in tomorrow. It holds live
    #: session cookies, so it is git-ignored like any other secret.
    profile_dir: str = "browser-profile"
    #: Cap on autonomous actions in one "do X on the site" task.
    max_steps: int = 8
    timeout_s: float = 20.0
    #: Hosts MEDO must never drive. Substring match on the hostname, so
    #: "bank" covers "mybank.com". Checked on navigation AND before each action.
    blocked_domains: list[str] = Field(default_factory=list)
    #: When on, "open youtube" / "search X on youtube" land in this controlled
    #: browser instead of the system default — one window, and MEDO can then
    #: act on what it just opened.
    route_opens: bool = True


class CouncilConfig(BaseModel):
    """The specialist council — see core/council.py."""

    enabled: bool = True
    #: Who answers when a question matches no field.
    default_agent: str = "software"
    #: Most specialists consulted for one "convene the council" question. Each
    #: is a full model round-trip, so this is a latency dial as much as a
    #: quality one.
    max_members: int = 3
    #: Specialist keys to switch off (HUD/config toggle).
    disabled: list[str] = Field(default_factory=list)
    #: Extra specialists, keyed by name: {title, prompt, triggers, wants_tools}.
    extra: dict = Field(default_factory=dict)


class WeatherConfig(BaseModel):
    default_city: str = "Skopje"
    latitude: float = 41.9973
    longitude: float = 21.4280


class NewsConfig(BaseModel):
    feeds: list[str] = Field(default_factory=list)
    #: Macedonian-language sources, used when the question was asked in
    #: Macedonian. Empty => ``feeds`` is used for both languages.
    feeds_mk: list[str] = Field(default_factory=list)


class WebFetchConfig(BaseModel):
    """Reading a page out loud — skills/webfetch.py.

    Every field here is a *bound*, not a preference: this skill pulls bytes
    from an address someone else controls, so the timeout, the byte cap and
    the redirect cap are what stop a hostile or broken page from hanging the
    assistant. Raising them widens that exposure.
    """

    enabled: bool = True
    timeout_s: float = 15.0
    #: Hard cap on the body. Enforced WHILE streaming (the read is aborted),
    #: never after — the point is to not have the bytes.
    max_bytes: int = 2_000_000
    #: Enough for http -> https -> www; a longer chain is a tracker or a loop.
    max_redirects: int = 3
    #: Sent as-is. Plain and honest: some sites 403 an unknown client, and
    #: pretending to be a browser is both a lie and a fragile one.
    user_agent: str = "Mozilla/5.0 (compatible; MEDO/2.0; local voice assistant)"


class VisionLLMConfig(BaseModel):
    """Local vision model for 'what do you see' / 'read my screen'."""

    model: str = "moondream"
    timeout_s: float = 60.0


class MemoryConfig(BaseModel):
    db_path: str = "medo.db"
    max_turns: int = 10
    max_facts: int = 20                 # facts injected into the system prompt
    # Local Ollama embedding model for semantic fact recall ("dentist" finds
    # "my dentist is Dr. ..."). Empty string disables -> newest-N as before.
    embed_model: str = "nomic-embed-text"
    # Where "import this file" copies documents and pictures. Under Documents
    # by default so it sits inside the whitelist AND inside the document
    # index's roots — imports stay searchable after a plain reindex.
    import_dir: str = "~/Documents/MEDO/Imports"


class MCPServerConfig(BaseModel):
    """One MCP server MEDO connects to (see core/mcp.py).

    Two transports: a local process (``command`` + ``args``, stdio) or a remote
    endpoint (``url``, Streamable HTTP/SSE). Exactly one of command/url should
    be set; each of the server's tools becomes a MEDO skill the LLM can call.
    """

    enabled: bool = True
    command: str = ""                   # e.g. "npx" (stdio transport)
    args: list[str] = Field(default_factory=list)  # e.g. ["-y", "@modelcontextprotocol/server-filesystem", "~"]
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""                       # e.g. "http://localhost:3000/mcp" (HTTP transport)


class MCPConfig(BaseModel):
    """Model Context Protocol client — connect any app that speaks MCP."""

    enabled: bool = True
    connect_timeout_s: float = 15.0
    call_timeout_s: float = 60.0
    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class RoutineItem(BaseModel):
    """One proactive routine (see core/routines.py)."""

    name: str = "routine"
    at: str = "08:00"                   # HH:MM local time
    days: list[str] = Field(default_factory=list)  # empty = daily; [mon, tue, ...]
    ask: list[str] = Field(default_factory=list)   # utterances routed + announced
    enabled: bool = True


class BriefingConfig(BaseModel):
    """Morning briefing sections (M8, skills/briefing.py) — order matters."""

    sections: list[str] = Field(
        default_factory=lambda: ["weather", "news", "reminders", "upcoming"]
    )


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    routing_stats: bool = True


class Settings(BaseSettings):
    """Root settings object — one instance per process."""

    model_config = SettingsConfigDict(
        env_prefix="MEDO_",
        env_nested_delimiter="__",
        extra="ignore",
        yaml_file=str(DEFAULT_CONFIG_PATH),
        yaml_file_encoding="utf-8",
    )

    llm: LLMConfig = Field(default_factory=LLMConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)
    remote: RemoteConfig = Field(default_factory=RemoteConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    council: CouncilConfig = Field(default_factory=CouncilConfig)
    hud: HudConfig = Field(default_factory=HudConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    wakeword: WakeWordConfig = Field(default_factory=WakeWordConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    mode: ModeConfig = Field(default_factory=ModeConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)
    web_fetch: WebFetchConfig = Field(default_factory=WebFetchConfig)
    vision_llm: VisionLLMConfig = Field(default_factory=VisionLLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    routines: list[RoutineItem] = Field(default_factory=list)
    briefing: BriefingConfig = Field(default_factory=BriefingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    # Raw per-platform app launch table; interpreted by skills/apps.py (M2).
    skills: dict = Field(default_factory=dict)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence: explicit kwargs > environment variables > config.yaml.
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
        )


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings, optionally from a non-default YAML path (used by tests).

    ``yaml_file`` lives in ``model_config`` — a class-level value that the
    ``_yaml_file`` init kwarg silently fails to override on the installed
    pydantic-settings, so the alternate file was never actually read. A
    throwaway subclass with its own ``model_config`` is version-proof.
    """
    if config_path is None:
        return Settings()

    class _FileSettings(Settings):
        model_config = SettingsConfigDict(
            **{**Settings.model_config, "yaml_file": str(config_path)}
        )

    return _FileSettings()


# --- local overrides (online API key + chosen mic) --------------------------
#
# A single git-ignored file holds per-machine settings that must not be committed
# to config.yaml: the online API key/model (``llm:``) and the chosen microphone
# (``audio:``). Deliberately kept out of load_settings() so tests that call
# load_settings() stay isolated from the developer's machine; main.py opts in by
# calling apply_local_secrets() at startup and the companion API opts in to writes.


def _read_local(path: Path = SECRETS_PATH) -> dict:
    """Whole local-overrides document (``{}`` if missing/corrupt). Never raises."""
    if not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # a corrupt file must never crash startup
        return {}


def _write_local(data: dict, path: Path = SECRETS_PATH) -> None:
    import yaml

    path.write_text(
        "# MEDO local overrides — git-ignored, do not commit.\n"
        + yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )


def load_llm_secrets(path: Path = SECRETS_PATH) -> dict:
    """Return the persisted llm secrets ``{field: value}`` (``{}`` if none)."""
    llm = _read_local(path).get("llm", {}) or {}
    return {k: llm[k] for k in _SECRET_LLM_FIELDS if k in llm}


def apply_local_secrets(settings: Settings, path: Path = SECRETS_PATH) -> Settings:
    """Overlay locally-saved llm secrets + chosen mic onto ``settings`` in place.

    Lets an online API key/model and the picked microphone survive a restart
    without ever being written to the git-tracked config.yaml. Returns the same
    settings for convenience.
    """
    data = _read_local(path)
    llm = data.get("llm", {}) or {}
    for field in _SECRET_LLM_FIELDS:
        value = llm.get(field)
        if value not in (None, ""):
            setattr(settings.llm, field, value)
    audio = data.get("audio", {}) or {}
    if "input_device" in audio:  # may be int, name, or None (= system default)
        settings.audio.input_device = audio["input_device"]
    remote = data.get("remote", {}) or {}
    if remote.get("token"):
        settings.remote.token = str(remote["token"])
    control = data.get("control", {}) or {}
    if "pc" in control:  # the HUD's PC CONTROL switch, remembered per machine
        settings.safety.pc_control_enabled = bool(control["pc"])
    return settings


def save_llm_secrets(path: Path = SECRETS_PATH, **fields: object) -> None:
    """Merge the given llm fields into the git-ignored overrides file.

    Merging (not overwriting) means switching provider to "ollama" keeps the
    stored key so the next switch back to online needs no re-entry. Only
    whitelisted fields are written; unknown kwargs are ignored. Best-effort.
    """
    payload = {k: v for k, v in fields.items() if k in _SECRET_LLM_FIELDS and v is not None}
    if not payload:
        return
    try:
        data = _read_local(path)
        data.setdefault("llm", {}).update(payload)
        _write_local(data, path)
    except Exception:
        logging.getLogger(__name__).exception("could not persist llm secrets")


def ensure_remote_token(settings: Settings, path: Path = SECRETS_PATH) -> str:
    """Return the companion-API token, generating + persisting one on first run.

    The token authenticates LAN clients (watch app etc.); localhost is exempt.
    It lives only in the git-ignored secrets file — never in config.yaml. If the
    secrets file can't be written the in-memory token still works for this run.
    """
    if settings.remote.token:
        return settings.remote.token
    stored = (_read_local(path).get("remote", {}) or {}).get("token")
    if stored:
        settings.remote.token = str(stored)
        return settings.remote.token
    import secrets as _secrets

    token = _secrets.token_urlsafe(24)
    settings.remote.token = token
    try:
        data = _read_local(path)
        data.setdefault("remote", {})["token"] = token
        _write_local(data, path)
    except Exception:
        logging.getLogger(__name__).exception("could not persist remote token")
    return token


def save_pc_control(enabled: bool, path: Path = SECRETS_PATH) -> None:
    """Persist the HUD's PC CONTROL switch (may MEDO act on this machine?)."""
    try:
        data = _read_local(path)
        data.setdefault("control", {})["pc"] = bool(enabled)
        _write_local(data, path)
    except Exception:
        logging.getLogger(__name__).exception("could not persist the PC-control switch")


def save_audio_input(device: int | str | None, path: Path = SECRETS_PATH) -> None:
    """Persist the chosen microphone (index/name/None) to the overrides file."""
    try:
        data = _read_local(path)
        data.setdefault("audio", {})["input_device"] = device
        _write_local(data, path)
    except Exception:
        logging.getLogger(__name__).exception("could not persist audio input device")
