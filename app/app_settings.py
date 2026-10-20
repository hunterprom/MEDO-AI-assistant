"""The friendly Settings backend (S4) — what a non-technical user can change,
written to the per-user file (app.user_settings), never the dev config.yaml.

Exposes exactly the fields a normal person touches: language pair, voice on/off +
volume, wake-word sensitivity (a 0-100 SLIDER, not a raw threshold), start-on-
boot, and the hardware profile. Every setter VALIDATES, so the UI can't persist
an out-of-range or unknown value. The slider↔threshold and profile↔config
translation lives here so the HUD stays dumb.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from app import user_settings as _us
from app.profiles import BY_NAME

#: Languages MEDO has confirm-word banks / voices for (lang/confirm_words/*.yaml).
SUPPORTED_LANGS = {"en", "mk", "es", "de"}

_DEFAULTS: Dict[str, Any] = {
    "voice_enabled": True,
    "volume": 80,                 # 0..100
    "wake_sensitivity": 50,       # 0..100 slider
    "start_on_boot": False,
    "language_primary": "en",
    "language_secondary": "mk",
}


def get_all(path=None) -> Dict[str, Any]:
    """Every user-facing setting, defaults merged with what's stored, plus the
    current hardware profile (owned by app.user_settings)."""
    stored = _us.load(path).get("app", {})
    merged = dict(_DEFAULTS)
    for k, v in stored.items():
        if k in _DEFAULTS:
            merged[k] = v
    merged["profile"] = _us.get_profile(path)
    merged["profile_override"] = _us.is_override(path)
    return merged


def _validate(key: str, value: Any) -> Any:
    if key in ("volume", "wake_sensitivity"):
        v = int(value)
        if not 0 <= v <= 100:
            raise ValueError(f"{key} must be 0-100, got {value}")
        return v
    if key in ("voice_enabled", "start_on_boot"):
        return bool(value)
    if key in ("language_primary", "language_secondary"):
        if value not in SUPPORTED_LANGS:
            raise ValueError(f"unsupported language {value!r}")
        return value
    raise KeyError(f"unknown setting {key!r}")


def set_field(key: str, value: Any, path=None) -> Any:
    """Validate then persist one setting. Raises ValueError/KeyError on a bad
    value — the invalid state is never written."""
    validated = _validate(key, value)
    if key in ("language_primary", "language_secondary"):
        other_key = ("language_secondary" if key == "language_primary"
                     else "language_primary")
        if validated == get_all(path).get(other_key):
            raise ValueError("the two languages must be different")
    data = _us.load(path)
    data.setdefault("app", {})[key] = validated
    _us.save(data, path)
    return validated


def set_language_pair(primary: str, secondary: str, path=None) -> None:
    """The 2-language feature: both must be supported and DIFFERENT. Written
    atomically (both at once) so the pair is never transiently equal."""
    _validate("language_primary", primary)
    _validate("language_secondary", secondary)
    if primary == secondary:
        raise ValueError("the two languages must be different")
    data = _us.load(path)
    app = data.setdefault("app", {})
    app["language_primary"] = primary
    app["language_secondary"] = secondary
    _us.save(data, path)


def set_profile(name: str, path=None) -> str:
    """Manual profile override from Settings (marks it a deliberate choice)."""
    if name not in BY_NAME:
        raise ValueError(f"unknown profile {name!r}")
    _us.set_profile(name, override=True, path=path)
    return name


# -- slider <-> engine-threshold translation ----------------------------------

def sensitivity_to_threshold(sensitivity: int) -> float:
    """0-100 slider -> wakeword.threshold (0..1). Higher sensitivity = LOWER
    threshold = easier to trigger. Maps 0 -> 0.9 (strict) … 100 -> 0.3 (loose)."""
    s = max(0, min(100, int(sensitivity)))
    return round(0.9 - (s / 100.0) * 0.6, 3)


def threshold_to_sensitivity(threshold: float) -> int:
    """Inverse of :func:`sensitivity_to_threshold`, for showing the stored
    engine value back on the slider."""
    t = max(0.3, min(0.9, float(threshold)))
    return int(round((0.9 - t) / 0.6 * 100))
