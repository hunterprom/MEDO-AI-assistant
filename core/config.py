"""Typed configuration loaded from ``config.yaml``.

The whole app reads settings through :class:`Settings`. Values come from the YAML
file, but any field can be overridden by an environment variable (e.g.
``MEDO_LLM__DEFAULT_MODEL=qwen2.5:7b``) thanks to pydantic-settings — handy for
tests and CI without editing the file.
"""

from __future__ import annotations

import logging
import os
import re
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
    # Per-machine tool-brain override: a VRAM-limited box borrowing the local
    # brain for live-info/council (when the main provider is a CLI agent) may
    # need a small model that fits, without editing the tracked config.yaml.
    "tool_brain_model",
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
    # Tier-2 SEMANTIC router (M2.5): sits between the regex fast path and the
    # LLM. When no pattern matches, an utterance can still reach a query-style
    # skill by MEANING — each eligible skill's routing_phrases are embedded once
    # (the same local nomic-embed-text the facts stack uses) and the utterance is
    # cosine-matched against them. Only skills that opt in with routing_phrases
    # are eligible, and only when the best match clears `semantic_threshold` AND
    # beats the runner-up by `semantic_margin` (an ambiguous match falls through
    # to the LLM). `semantic_shadow` logs what it WOULD route without acting —
    # leave it on until the logs show it routing well, then flip it off to go
    # live. Embedder down => the tier is simply skipped.
    semantic_enabled: bool = True
    semantic_shadow: bool = True          # log-only until proven on real usage
    semantic_threshold: float = 0.6       # min cosine similarity to route
    semantic_margin: float = 0.04         # best must beat the runner-up by this
    # Adaptive route memory (M2.5e): LEARN from confirmed LLM resolutions. When
    # a query misses the fast path and the curated semantic tier, and the LLM
    # resolves it to exactly one non-destructive query skill, that (utterance ->
    # skill) is stored as an embedded exemplar; later, the same/near-same
    # phrasing shortcuts straight to the skill on the SEMANTIC path. CONSULTING
    # runs inside _semantic_route (so it inherits `semantic_enabled` and the
    # `semantic_shadow` gate); LEARNING is gated only by `route_memory_enabled`,
    # so a personal corpus can accumulate even while the curated tier is still in
    # shadow. Default OFF: the whole feature is a no-op until flipped on.
    route_memory_enabled: bool = False    # master gate; inert when False
    route_memory_threshold: float = 0.82  # HIGH cosine gate, no margin (exemplars
                                          # are near-exact single user phrasings)
    route_memory_max: int = 500           # row cap; least-used evicted past this

    # The semantic tier has THREE honest states, which the two flags above
    # encode. The HUD CONFIG tab and the companion API speak in this single
    # word so a user never has to reason about the flag pair:
    #   "off"    -> disabled (fast -> LLM, as before)
    #   "shadow" -> logs the would-be route but still uses the LLM (safe rollout)
    #   "live"   -> routes by meaning
    def semantic_mode(self) -> str:
        """The tier's state as one word: ``off`` | ``shadow`` | ``live``."""
        if not self.semantic_enabled:
            return "off"
        return "shadow" if self.semantic_shadow else "live"

    def apply_semantic_mode(self, mode: str) -> str:
        """Set the tier from one word; returns the resolved mode. Raises
        ``ValueError`` on anything but off/shadow/live so a bad API/HUD value is
        rejected rather than silently ignored."""
        mode = (mode or "").strip().lower()
        if mode == "off":
            self.semantic_enabled = False
        elif mode == "shadow":
            self.semantic_enabled, self.semantic_shadow = True, True
        elif mode == "live":
            self.semantic_enabled, self.semantic_shadow = True, False
        else:
            raise ValueError(
                f"semantic mode must be 'off', 'shadow', or 'live'; got {mode!r}")
        return self.semantic_mode()


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


class FillerConfig(BaseModel):
    """Spoken 'let me think' fillers that bridge dead air on a SLOW LLM answer.

    Only the LLM path arms them — the fast path answers in milliseconds. A big
    or complex question gets a quick acknowledgement; ANY LLM answer still
    silent after ``delay_s`` gets a filler; a very long wait (Claude Code can
    take 20-30 s) gets ONE follow-up. Butler-toned phrases live in
    ``lang/filler_phrases/<code>.yaml`` and follow the active languages (a
    missing bank falls back to the primary + English). See core/filler.py.
    """

    enabled: bool = True
    #: Speak a filler when the LLM has produced nothing after this long.
    delay_s: float = 8.0
    #: A big/complex question gets this quicker acknowledgement instead.
    big_delay_s: float = 2.0
    #: A question with at least this many words counts as "big".
    big_question_words: int = 12
    #: After the first filler, wait this long for the reply before ONE follow-up.
    followup_s: float = 12.0


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


class OverlayConfig(BaseModel):
    """The desktop presence sphere — a small always-on-top CORE sphere pinned to
    a screen corner (``ui/overlay.py``), so MEDO stays visible while you work in
    other apps.

    It adds no listening of its own: the wake word already runs with the mic
    open whatever is on screen. This is the *presence* — you can see it heard
    you, watch it think and answer, click it to talk without the wake word, and
    double-click to open the full HUD. Runs as its own process so a UI crash can
    never take the assistant down.
    """

    enabled: bool = True
    size: int = 132                     # sphere diameter in px
    corner: Literal["bottom-right", "bottom-left",
                    "top-right", "top-left"] = "bottom-right"
    margin: int = 18                    # gap from the screen edge
    #: Frames/s (clamped 4..30). The sphere is composed in Python, so this is
    #: the CPU dial: ~9 ms a frame here, i.e. roughly 10% of ONE core at 12.
    #: Drop it to 8 for a near-free widget; the rotation is slow either way.
    fps: int = 12


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
    #: The corner presence sphere (see OverlayConfig).
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)


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
        """Fold ``language_mode`` into ``language`` — the one field the
        transcriber reads.

        Semantics: a FORCED ``language_mode`` (a real code) wins and pins
        ``language``. ``language_mode: "auto"`` means "no override from here" —
        so it defers to whatever ``language`` says, which is ``null`` (real
        auto-detect) by default but is RESPECTED when the user explicitly forced
        it (e.g. ``language: "mk"``). The earlier version keyed off
        ``"language_mode" in model_fields_set``, but the shipped config.yaml
        always writes ``language_mode: "auto"`` — so it silently wiped an
        explicit ``language: "mk"`` back to auto-detect, the exact misdetection
        the user was fixing. An unsupported code falls back to auto with a
        warning rather than forcing a language MEDO has no voice for.
        """
        from core import languages

        mode = (self.language_mode or "auto").strip().lower()
        forced_legacy = self.language if (
            self.language and str(self.language).strip().lower() not in ("", "auto")
        ) else None

        if mode in ("", "auto"):
            # No override from language_mode: honour an explicit `language`,
            # else auto-detect.
            self.language_mode = "auto"
            self.language = forced_legacy
            return self
        if languages.get(mode) is None:
            logging.getLogger(__name__).warning(
                "stt.language_mode=%r is not a language MEDO speaks — using "
                "auto-detect. Supported: %s", mode, ", ".join(languages.codes()))
            self.language_mode, self.language = "auto", forced_legacy
            return self
        self.language_mode = mode
        self.language = mode
        return self


def _default_available_languages() -> list[str]:
    from core import languages
    return languages.codes()


class LanguagesConfig(BaseModel):
    """The two-language product rule: MEDO runs with AT MOST 2 ACTIVE languages
    at a time (default English + Macedonian), and per-utterance detection is
    constrained to that pair.

    Why the cap is an ACCURACY feature, not just a preference: open-ended
    language ID across ~90 languages mis-detects constantly (spoken Macedonian
    heard as Bulgarian/Serbian, Spanish as Portuguese). Restricting the decoder
    to two known candidates is the single biggest lever on bilingual STT
    reliability — detection never guesses across all languages.

    This is the ONE source of truth for "which languages are live". Every
    language-dependent consumer (STT clamp, confirm-word banks, persona/filler
    phrases, TTS voice) reads it through ``Settings.active_languages()`` /
    ``primary_language()`` / ``detection_mode()`` — no ``"en"``/``"mk"`` literal
    survives elsewhere. The legacy ``stt.allowed_languages`` / ``spoken_languages``
    / ``language_mode`` fields are back-compat inputs only (see
    ``Settings._reconcile_languages``); this block wins whenever it is present.
    """

    #: Everything MEDO CAN support — i.e. has both STT support and a voice.
    #: A subset of the registry in core/languages.py; codes not in the registry
    #: are dropped with a warning (MEDO has no way to hear or speak them).
    available: list[str] = Field(default_factory=_default_available_languages)
    #: The AT MOST 2 languages live right now. Default English + Macedonian.
    active: list[str] = Field(default_factory=lambda: ["en", "mk"])
    #: The fallback language — the voice used when another active language has
    #: none, and the language the system prompt is written in. ``None`` resolves
    #: to the first active language; when set it MUST be one of ``active``.
    primary: str | None = None
    #: ``auto_pair`` = per-utterance detect BETWEEN the two active languages.
    #: ``fixed`` = always transcribe as ``primary`` (fastest + most accurate
    #: when you only ever speak one of the two).
    detection: Literal["auto_pair", "fixed"] = "auto_pair"

    @model_validator(mode="after")
    def _validate(self) -> LanguagesConfig:
        from core import languages as langs

        log = logging.getLogger(__name__)

        def _norm(seq: list[str]) -> list[str]:
            out: list[str] = []
            for c in seq:
                c = str(c).strip().lower()
                if c and c not in out:
                    out.append(c)
            return out

        # available: drop anything the registry can't hear/speak.
        available: list[str] = []
        for c in (_norm(self.available) or langs.codes()):
            if langs.get(c) is None:
                log.warning("languages.available: %r is not a language MEDO "
                            "supports — ignoring it", c)
            else:
                available.append(c)
        available = available or langs.codes()

        # active: 1 or 2 entries, each supported. HARD errors on the rest.
        active = _norm(self.active)
        if not active:
            raise ValueError("languages.active must list at least one language "
                             "(default is [\"en\", \"mk\"])")
        if len(active) > 2:
            raise ValueError(
                f"MEDO runs at most 2 active languages at a time; got "
                f"{len(active)}: {active}. Trim languages.active to two.")
        missing = [c for c in active if c not in available]
        if missing:
            raise ValueError(
                f"languages.active {missing} not in languages.available "
                f"{available} — add them to available or remove them.")

        # primary: None -> first active; when set it must be active.
        primary = (str(self.primary).strip().lower() if self.primary else "") or None
        if primary is None:
            primary = active[0]
        elif primary not in active:
            raise ValueError(
                f"languages.primary {primary!r} must be one of the active "
                f"languages {active}.")

        # warn (never fail) when an active language has no voice — TTS will fall
        # back to the primary voice (wired in S3).
        for c in active:
            lang = langs.get(c)
            if lang is not None and not lang.voice:
                log.warning("languages.active: %r has no TTS voice; it will be "
                            "spoken with the %r (primary) voice", c, primary)

        self.available, self.active, self.primary = available, active, primary
        return self

    @classmethod
    def from_legacy(cls, stt: STTConfig) -> LanguagesConfig:
        """Synthesize the active pair from the old ``stt.*`` fields, used only
        when a config has no ``languages:`` block (older installs).

        A single forced ``stt.language_mode`` becomes ``fixed`` on that language;
        otherwise the active pair is the first two supported languages from
        ``spoken_languages`` (then ``allowed_languages``), defaulting to en+mk.
        """
        from core import languages as langs

        forced = str(getattr(stt, "language_mode", "auto") or "auto").strip().lower()
        if forced not in ("", "auto") and langs.get(forced) is not None:
            return cls(available=langs.codes(), active=[forced],
                       primary=forced, detection="fixed")
        pool = (list(getattr(stt, "spoken_languages", []) or [])
                or list(getattr(stt, "allowed_languages", []) or [])
                or ["en", "mk"])
        seen: set[str] = set()
        active: list[str] = []
        for c in pool:
            c = str(c).strip().lower()
            if c and c not in seen and langs.get(c) is not None:
                seen.add(c)
                active.append(c)
                if len(active) == 2:
                    break
        active = active or ["en", "mk"]
        return cls(available=langs.codes(), active=active,
                   primary=active[0], detection="auto_pair")


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
    #: Path to a custom Chromium-based browser executable (e.g. Opera). When set
    #: it WINS over ``channel``: Playwright launches that binary — but always on
    #: MEDO's OWN ``profile_dir``, never the browser's real profile, so it never
    #: touches (say) your Opera logins/history. Empty => use ``channel``.
    executable_path: str = ""
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
    #: Optional bearer token for an HTTP server — sent as "Authorization: Bearer
    #: <token>". A secret, so servers added from the HUD are persisted to the
    #: git-ignored secrets file, never config.yaml.
    auth_token: str = ""


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


#: Events a webhook can fire on — the label maps to an EventBus subscription.
WEBHOOK_EVENTS = ("wake", "transcript", "routed", "reply")


class WebhookConfig(BaseModel):
    """One outbound webhook: POST a JSON payload to a URL when an event fires.

    Lets MEDO drive external automations (n8n, Home Assistant, a logger) with no
    per-integration code. ``auth_token`` is an optional bearer secret, so an
    HUD-added webhook is persisted to the git-ignored overrides, never committed.
    """

    name: str = "webhook"
    event: str = "routed"               # one of WEBHOOK_EVENTS
    url: str = ""
    auth_token: str = ""                # optional Authorization: Bearer <token>
    enabled: bool = True


class BriefingConfig(BaseModel):
    """Morning briefing sections (M8, skills/briefing.py) — order matters."""

    sections: list[str] = Field(
        default_factory=lambda: ["weather", "news", "reminders", "upcoming"]
    )


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    routing_stats: bool = True


class SelfDevConfig(BaseModel):
    """MEDO changing its OWN code — the self-programming engine (core/self_dev.py).

    OFF by default: this drives a coding agent that edits files. The safety model
    is 'propose & wait' — every change is made in an ISOLATED git worktree on a
    branch, gated by the test + lint suite, and NEVER merged into the live branch
    (nor does MEDO restart itself) until a human approves it. Turning ``enabled``
    on, or flipping ``auto_apply``, widens what MEDO can do to itself unattended.
    """

    enabled: bool = False
    #: Which installed coding agent drives the edits ("claude-code" or "codex").
    engine: str = "claude-code"
    #: Proposal branches are named "<prefix>/<slug>-<hash>".
    branch_prefix: str = "medo/self-dev"
    #: The gate a proposal must pass, run as ``<python> <args>`` in the worktree.
    test_cmd: list[str] = Field(default_factory=lambda: ["-m", "pytest", "-q"])
    lint_cmd: list[str] = Field(
        default_factory=lambda: ["-m", "ruff", "check", "."])
    #: Hard bound on the coding agent (it can otherwise run a very long time).
    agent_timeout_s: float = 900.0
    #: Hard bound on the test+lint gate.
    check_timeout_s: float = 600.0
    #: Claude Code permission mode used INSIDE the worktree. "acceptEdits" lets it
    #: edit files without prompting; the worktree isolation + human approval before
    #: any merge is the real safety boundary. The engine runs the tests itself, so
    #: the agent never needs shell access.
    permission_mode: str = "acceptEdits"
    #: propose & wait: proposals are never auto-merged. Only flip this for the
    #: 'auto-apply on green' autonomy level.
    auto_apply: bool = False


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
    filler: FillerConfig = Field(default_factory=FillerConfig)
    remote: RemoteConfig = Field(default_factory=RemoteConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    council: CouncilConfig = Field(default_factory=CouncilConfig)
    hud: HudConfig = Field(default_factory=HudConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    languages: LanguagesConfig = Field(default_factory=LanguagesConfig)
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
    webhooks: list[WebhookConfig] = Field(default_factory=list)
    self_dev: SelfDevConfig = Field(default_factory=SelfDevConfig)
    briefing: BriefingConfig = Field(default_factory=BriefingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    # Raw per-platform app launch table; interpreted by skills/apps.py (M2).
    skills: dict = Field(default_factory=dict)

    # --- the one language accessor every consumer uses -----------------------
    #
    # Nothing outside config reads a hardcoded "en"/"mk" any more: STT, the
    # confirmation gate, persona/filler phrases and TTS all go through these.

    def active_languages(self) -> list[str]:
        """The <=2 languages live right now (e.g. ``["en", "mk"]``)."""
        return list(self.languages.active)

    def primary_language(self) -> str:
        """The fallback language code (always one of ``active_languages()``)."""
        return self.languages.primary or self.languages.active[0]

    def detection_mode(self) -> str:
        """``"auto_pair"`` (detect between the pair) or ``"fixed"`` (primary)."""
        return self.languages.detection

    @model_validator(mode="after")
    def _reconcile_languages(self) -> Settings:
        """Back-compat: when a config has no ``languages:`` block, synthesize one
        from the legacy ``stt.*`` fields so older installs behave unchanged. An
        explicit ``languages:`` block always wins."""
        if "languages" not in self.model_fields_set:
            self.languages = LanguagesConfig.from_legacy(self.stt)
        return self

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
    browser = data.get("browser", {}) or {}  # the HUD's controlled-browser picker
    if "channel" in browser:
        settings.browser.channel = str(browser["channel"])
    if "executable_path" in browser:
        settings.browser.executable_path = str(browser["executable_path"])
    for name, spec in (data.get("mcp_servers", {}) or {}).items():
        # MCP servers added from the HUD (their auth tokens are secret, so they
        # live here, never in config.yaml). Merged onto any config.yaml servers.
        if not isinstance(spec, dict):
            continue
        try:
            settings.mcp.servers[str(name)] = MCPServerConfig(**spec)
        except Exception:
            logging.getLogger(__name__).warning(
                "ignoring invalid saved MCP server %r", name)
    saved_routines = data.get("routines")  # the HUD's scheduled-routines editor
    if isinstance(saved_routines, list):
        try:
            settings.routines = [RoutineItem(**r) for r in saved_routines
                                 if isinstance(r, dict)]
        except Exception:
            logging.getLogger(__name__).warning("ignoring invalid saved routines")
    saved_hooks = data.get("webhooks")  # the HUD's event-webhooks editor
    if isinstance(saved_hooks, list):
        try:
            settings.webhooks = [WebhookConfig(**h) for h in saved_hooks
                                 if isinstance(h, dict)]
        except Exception:
            logging.getLogger(__name__).warning("ignoring invalid saved webhooks")
    router = data.get("router", {}) or {}  # the HUD's SEMANTIC TIER selector
    if "semantic_enabled" in router:
        settings.router.semantic_enabled = bool(router["semantic_enabled"])
    if "semantic_shadow" in router:
        settings.router.semantic_shadow = bool(router["semantic_shadow"])
    langs = data.get("languages", {}) or {}
    if langs.get("active"):  # the HUD language picker, remembered per machine
        try:
            settings.languages = LanguagesConfig(
                available=settings.languages.available,
                active=langs["active"],
                primary=langs.get("primary"),
                detection=langs.get("detection") or settings.languages.detection,
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "ignoring invalid saved languages %r", langs)
    # MCP server secrets (GitHub PAT, Google/Brave keys…): kept in the git-
    # ignored overrides file under `mcp_env:` and injected into the environment
    # so spawned MCP servers inherit them — never written to config.yaml.
    mcp_env = data.get("mcp_env", {}) or {}
    for key, value in mcp_env.items():
        if value not in (None, "") and key not in os.environ:
            os.environ[str(key)] = str(value)
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


def save_semantic_mode(mode: str, path: Path = SECRETS_PATH) -> str:
    """Validate + persist the HUD's SEMANTIC TIER choice (off/shadow/live).

    Validation goes through :meth:`RouterConfig.apply_semantic_mode` on a throwaway
    config, so a bad word RAISES before anything is written (the caller surfaces
    it). The resolved flag PAIR is stored under ``router:`` in the git-ignored
    overrides file and re-applied at startup by ``apply_local_secrets`` — never
    written to the committed config.yaml. Returns the resolved mode.
    """
    cfg = RouterConfig()
    resolved = cfg.apply_semantic_mode(mode)  # raises ValueError on a bad mode
    try:
        data = _read_local(path)
        data.setdefault("router", {}).update(
            {"semantic_enabled": cfg.semantic_enabled,
             "semantic_shadow": cfg.semantic_shadow})
        _write_local(data, path)
    except Exception:
        logging.getLogger(__name__).exception("could not persist the semantic mode")
    return resolved


def save_audio_input(device: int | str | None, path: Path = SECRETS_PATH) -> None:
    """Persist the chosen microphone (index/name/None) to the overrides file."""
    try:
        data = _read_local(path)
        data.setdefault("audio", {})["input_device"] = device
        _write_local(data, path)
    except Exception:
        logging.getLogger(__name__).exception("could not persist audio input device")


def find_browser_executable(name: str) -> str | None:
    """Locate a Chromium-based browser's executable by short name (e.g. 'opera').

    Windows-focused (where MEDO runs). Opera installs into a versioned folder,
    so those are globbed and the newest is preferred. Returns the path or None.
    MEDO drives it on its OWN profile_dir, so which install it is doesn't matter.
    """
    import glob

    key = (name or "").strip().lower()
    la = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
    pf = os.environ.get("PROGRAMFILES", r"C:\Program Files")
    pfx = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    if key in ("opera", "opera gx", "operagx", "opera-gx"):
        folder = "Opera GX" if "gx" in key else "Opera"
        bases = [f"{la}\\Programs\\{folder}", f"{pf}\\{folder}", f"{pfx}\\{folder}"]
        found: list[str] = []
        for base in bases:
            found.append(f"{base}\\opera.exe")           # the launcher stub
            found += sorted(glob.glob(f"{base}\\*\\opera.exe"), reverse=True)  # versioned
        for c in found:
            if Path(c).is_file():
                return c
    return None


#: The choices the HUD browser picker offers.
BROWSER_CHOICES = ("chrome", "msedge", "chromium", "opera")


def browser_choice(browser: "BrowserConfig") -> str:
    """The picker label for the CURRENT controlled-browser config."""
    if browser.executable_path:
        low = browser.executable_path.lower()
        if "opera gx" in low or "opera_gx" in low:
            return "opera gx"
        return "opera" if "opera" in low else "custom"
    return browser.channel or "chromium"


def resolve_browser(choice: str) -> dict:
    """Validate a controlled-browser choice -> ``{choice, channel,
    executable_path}``. chrome/msedge -> a Playwright channel; chromium ->
    bundled; opera -> the detected Opera executable (driven on MEDO's own
    profile, never Opera's). Raises ValueError on an unknown choice, or when the
    named browser isn't installed. No file is written."""
    c = (choice or "").strip().lower()
    if c in ("chrome", "msedge"):
        channel, exe = c, ""
    elif c in ("chromium", "default", ""):
        channel, exe = "", ""
    elif c in ("opera", "opera gx", "operagx"):
        exe = find_browser_executable(c)
        if not exe:
            raise ValueError(f"couldn't find {choice} installed on this machine")
        channel = ""
    else:
        raise ValueError(
            f"unknown browser {choice!r}; use chrome, msedge, chromium, or opera")
    return {"choice": c, "channel": channel, "executable_path": exe}


def save_browser(choice: str, path: Path = SECRETS_PATH) -> dict:
    """Validate (:func:`resolve_browser`) + persist the browser choice."""
    resolved = resolve_browser(choice)              # raises on bad/missing
    try:
        data = _read_local(path)
        data["browser"] = {"channel": resolved["channel"],
                           "executable_path": resolved["executable_path"]}
        _write_local(data, path)
    except Exception:
        logging.getLogger(__name__).exception("could not persist the browser choice")
    return resolved


def save_mcp_server(name: str, url: str = "", auth_token: str = "",
                    command: str = "", args: list | None = None,
                    path: Path = SECRETS_PATH) -> dict:
    """Add/update an MCP server in the git-ignored overrides (its auth token is a
    secret). Validates via :class:`MCPServerConfig`; needs a url OR a command.
    Returns a token-free summary. The server connects on the next startup."""
    name = (name or "").strip()
    if not name:
        raise ValueError("a server name is required")
    spec = MCPServerConfig(url=(url or "").strip(), auth_token=(auth_token or "").strip(),
                           command=(command or "").strip(), args=list(args or []))
    if not spec.url and not spec.command:
        raise ValueError("give a URL (HTTP server) or a command (local server)")
    stored: dict = {"enabled": True}
    for k in ("url", "auth_token", "command"):
        if getattr(spec, k):
            stored[k] = getattr(spec, k)
    if spec.args:
        stored["args"] = spec.args
    try:
        data = _read_local(path)
        data.setdefault("mcp_servers", {})[name] = stored
        _write_local(data, path)
    except Exception:
        logging.getLogger(__name__).exception("could not persist the MCP server")
    return {"name": name, "url": spec.url, "command": spec.command,
            "has_auth": bool(spec.auth_token)}


def remove_mcp_server(name: str, path: Path = SECRETS_PATH) -> bool:
    """Remove an HUD-added MCP server from the overrides. True if it existed."""
    try:
        data = _read_local(path)
        servers = data.get("mcp_servers", {}) or {}
        if name in servers:
            del servers[name]
            _write_local(data, path)
            return True
    except Exception:
        logging.getLogger(__name__).exception("could not remove the MCP server")
    return False


def resolve_routines(routines: list) -> list[RoutineItem]:
    """Validate a list of routine dicts -> [RoutineItem]. Raises ValueError on a
    bad HH:MM time or a routine with nothing to ask."""
    out: list[RoutineItem] = []
    for r in routines or []:
        if not isinstance(r, dict):
            continue
        at = str(r.get("at") or "").strip()
        if not re.match(r"^\d{1,2}:\d{2}$", at):
            raise ValueError(f"time must be HH:MM, got {at!r}")
        hh, mm = (int(x) for x in at.split(":"))
        if hh > 23 or mm > 59:
            raise ValueError(f"{at} is not a valid time")
        ask = [str(a).strip() for a in (r.get("ask") or []) if str(a).strip()]
        if not ask:
            raise ValueError(f"routine {r.get('name') or 'routine'!r} has nothing to ask")
        out.append(RoutineItem(
            name=str(r.get("name") or "routine").strip() or "routine",
            at=f"{hh:02d}:{mm:02d}",
            days=[str(d).strip().lower()[:3] for d in (r.get("days") or [])],
            ask=ask, enabled=bool(r.get("enabled", True))))
    return out


def save_routines(routines: list, path: Path = SECRETS_PATH) -> list[RoutineItem]:
    """Validate + persist the HUD's scheduled routines. Takes effect on restart."""
    items = resolve_routines(routines)              # raises on a bad routine
    try:
        data = _read_local(path)
        data["routines"] = [i.model_dump() for i in items]
        _write_local(data, path)
    except Exception:
        logging.getLogger(__name__).exception("could not persist routines")
    return items


def save_webhook(name: str, event: str, url: str, auth_token: str = "",
                 path: Path = SECRETS_PATH) -> dict:
    """Validate + persist one HUD-added event webhook (its auth token is a secret,
    so it lives in the overrides, never config.yaml). Raises ValueError on a
    missing name/url or an unknown event. Takes effect on restart."""
    name = (name or "").strip()
    url = (url or "").strip()
    event = (event or "").strip().lower()
    if not name:
        raise ValueError("a webhook needs a name")
    if not url:
        raise ValueError("a webhook needs a URL to POST to")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("the URL must start with http:// or https://")
    if event not in WEBHOOK_EVENTS:
        raise ValueError(f"event must be one of {', '.join(WEBHOOK_EVENTS)}")
    spec = WebhookConfig(name=name, event=event, url=url,
                         auth_token=(auth_token or "").strip())
    try:
        data = _read_local(path)
        hooks = [h for h in (data.get("webhooks") or [])
                 if isinstance(h, dict) and h.get("name") != name]
        hooks.append(spec.model_dump())
        data["webhooks"] = hooks
        _write_local(data, path)
    except Exception:
        logging.getLogger(__name__).exception("could not persist the webhook")
    return {"name": name, "event": event, "url": url,
            "has_auth": bool(spec.auth_token)}


def remove_webhook(name: str, path: Path = SECRETS_PATH) -> bool:
    """Remove an HUD-added webhook from the overrides. True if it existed."""
    try:
        data = _read_local(path)
        hooks = data.get("webhooks") or []
        kept = [h for h in hooks if not (isinstance(h, dict) and h.get("name") == name)]
        if len(kept) != len(hooks):
            data["webhooks"] = kept
            _write_local(data, path)
            return True
    except Exception:
        logging.getLogger(__name__).exception("could not remove the webhook")
    return False


def save_languages(active: list[str], primary: str | None = None,
                   detection: str | None = None,
                   path: Path = SECRETS_PATH) -> LanguagesConfig:
    """Validate + persist the chosen active languages to the overrides file.

    Validation happens through :class:`LanguagesConfig` (so >2 active, an
    unknown code, or a bad primary RAISE before anything is written) — the
    caller (companion API) surfaces the error to the picker. Returns the
    validated config. The change needs a restart to reload the STT model.
    """
    cfg = LanguagesConfig(active=active, primary=primary,
                          detection=(detection or "auto_pair"))
    try:
        data = _read_local(path)
        data["languages"] = {"active": list(cfg.active), "primary": cfg.primary,
                             "detection": cfg.detection}
        _write_local(data, path)
    except Exception:
        logging.getLogger(__name__).exception("could not persist languages")
    return cfg
