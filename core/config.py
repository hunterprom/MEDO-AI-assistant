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

from pydantic import BaseModel, Field
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
    temperature: float = 0.6
    num_ctx: int = 4096
    request_timeout_s: float = 120.0
    # Ollama keep-alive: how long the model stays loaded after a request.
    keep_alive: str = "30m"


class RouterConfig(BaseModel):
    fast_path_enabled: bool = True


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
    max_fps: int = 15                       # throttle inference on CPU
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    stability_frames: int = 6               # consecutive frames to confirm a gesture
    cooldown_s: float = 2.0                 # min gap before re-firing a gesture
    pointer: PointerConfig = Field(default_factory=PointerConfig)
    bench: BenchConfig = Field(default_factory=BenchConfig)
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


class HudConfig(BaseModel):
    """M.E.D.O. web HUD (arc-reactor front end) — see ui/hud.py."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8730
    max_dir_dots: int = 300             # how many folder dots to scatter on the orb


class STTConfig(BaseModel):
    engine: str = "faster-whisper"
    model: str = "small"                 # small is noticeably more accurate than base
    device: str = "auto"
    compute_type: str = "auto"
    # null/None = per-utterance auto-detect (enables Macedonian + English).
    language: str | None = "en"
    # With auto-detect on, clamp detection to these languages: Whisper often
    # mistakes spoken Macedonian for Bulgarian/Serbian, which garbles the
    # decode. Outside-the-set detections are re-transcribed forced to the
    # first non-English entry. Empty = no clamp.
    allowed_languages: list[str] = Field(default_factory=list)
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


class AudioConfig(BaseModel):
    # int index, a case-insensitive name substring (e.g. "FHD Webcam") that
    # survives device re-indexing across reboots, or a PRIORITY LIST of those —
    # the first currently-available entry wins and the voice loop hot-swaps
    # (~2 s) when a higher-priority device connects. null = system default.
    input_device: int | str | list[int | str] | None = None
    output_device: int | None = None
    sample_rate: int = 16000
    silence_threshold: float = 0.015
    silence_duration_s: float = 1.0


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


class WeatherConfig(BaseModel):
    default_city: str = "Skopje"
    latitude: float = 41.9973
    longitude: float = 21.4280


class NewsConfig(BaseModel):
    feeds: list[str] = Field(default_factory=list)


class VisionLLMConfig(BaseModel):
    """Local vision model for 'what do you see' / 'read my screen'."""

    model: str = "moondream"
    timeout_s: float = 60.0


class MemoryConfig(BaseModel):
    db_path: str = "jarvis.db"
    max_turns: int = 10
    max_facts: int = 20                 # facts injected into the system prompt
    # Local Ollama embedding model for semantic fact recall ("dentist" finds
    # "my dentist is Dr. ..."). Empty string disables -> newest-N as before.
    embed_model: str = "nomic-embed-text"


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
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)
    remote: RemoteConfig = Field(default_factory=RemoteConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    hud: HudConfig = Field(default_factory=HudConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    wakeword: WakeWordConfig = Field(default_factory=WakeWordConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)
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
