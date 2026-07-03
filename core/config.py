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
_SECRET_LLM_FIELDS = ("provider", "api_key", "openai_base_url", "default_model")


def expand_path(path: str | Path) -> Path:
    """Expand ``~`` and environment variables, returning an absolute path."""
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


class LLMConfig(BaseModel):
    provider: Literal["ollama", "openai"] = "ollama"
    host: str = "http://localhost:11434"
    api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
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


class RemoteConfig(BaseModel):
    """Companion API (watch app) — see remote/server.py."""

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8710


class PointerConfig(BaseModel):
    """Gesture mouse control (pointer mode) — see vision/engine.py.

    ``enabled`` gates the feature; the runtime toggle always starts OFF and is
    flipped by voice ("pointer on"), the HUD switch, or POST :8731/pointer.
    """

    enabled: bool = True
    sensitivity: float = 2.5            # hand range → screen range amplification
    ema_alpha: float = 0.4              # 0..1 smoothing (higher = snappier)
    click_debounce_ms: int = 600        # min gap between gesture clicks
    hold_frames: int = 3                # frames a click pose must hold to fire


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


class STTConfig(BaseModel):
    engine: str = "faster-whisper"
    model: str = "small"                 # small is noticeably more accurate than base
    device: str = "auto"
    compute_type: str = "auto"
    # null/None = per-utterance auto-detect (enables Macedonian + English).
    language: str | None = "en"
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


class WakeWordConfig(BaseModel):
    engine: str = "openwakeword"
    phrase: str = "hey_jarvis"
    threshold: float = 0.5


class AudioConfig(BaseModel):
    input_device: int | None = None
    output_device: int | None = None
    sample_rate: int = 16000
    silence_threshold: float = 0.015
    silence_duration_s: float = 1.0


class SafetyConfig(BaseModel):
    confirm_destructive: bool = True
    whitelist_dirs: list[str] = Field(default_factory=list)

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
    max_facts: int = 20                 # newest facts injected into the system prompt


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


# --- local secrets (online API key) -----------------------------------------
#
# Deliberately kept out of load_settings() so tests that call load_settings()
# stay fully isolated from the developer's machine. main.py opts in by calling
# apply_local_secrets() at startup, and the companion API opts in to writes.


def load_llm_secrets(path: Path = SECRETS_PATH) -> dict:
    """Return the persisted llm secrets ``{field: value}`` (``{}`` if none)."""
    if not path.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        llm = data.get("llm", {}) or {}
        return {k: llm[k] for k in _SECRET_LLM_FIELDS if k in llm}
    except Exception:  # a corrupt secrets file must never crash startup
        return {}


def apply_local_secrets(settings: Settings, path: Path = SECRETS_PATH) -> Settings:
    """Overlay any locally-saved llm secrets onto ``settings.llm`` in place.

    Lets an online API key + model survive a restart without ever being written
    to the git-tracked config.yaml. Returns the same settings for convenience.
    """
    for field, value in load_llm_secrets(path).items():
        if value not in (None, ""):
            setattr(settings.llm, field, value)
    return settings


def save_llm_secrets(path: Path = SECRETS_PATH, **fields: object) -> None:
    """Merge the given llm fields into the git-ignored secrets file.

    Merging (not overwriting) means switching provider to "ollama" keeps the
    stored key so the next switch back to online needs no re-entry. Only
    whitelisted fields are written; unknown kwargs are ignored. Best-effort.
    """
    payload = {k: v for k, v in fields.items() if k in _SECRET_LLM_FIELDS and v is not None}
    if not payload:
        return
    try:
        import yaml

        existing = load_llm_secrets(path)
        existing.update(payload)
        path.write_text(
            "# MEDO local secrets — git-ignored, do not commit.\n"
            + yaml.safe_dump({"llm": existing}, sort_keys=False),
            encoding="utf-8",
        )
    except Exception:
        logging.getLogger(__name__).exception("could not persist llm secrets")
