"""Friendly Settings backend (S4): persistence, validation, slider mapping."""

from __future__ import annotations

import pytest

from app import app_settings as s


def test_defaults_when_nothing_stored(tmp_path):
    p = tmp_path / "settings.json"
    got = s.get_all(p)
    assert got["voice_enabled"] is True and got["volume"] == 80
    assert got["language_primary"] == "en" and got["language_secondary"] == "mk"
    assert got["profile"] is None


def test_set_and_persist(tmp_path):
    p = tmp_path / "settings.json"
    s.set_field("volume", 40, p)
    s.set_field("voice_enabled", False, p)
    got = s.get_all(p)
    assert got["volume"] == 40 and got["voice_enabled"] is False


@pytest.mark.parametrize("key,bad", [
    ("volume", 200), ("volume", -1), ("wake_sensitivity", 101),
])
def test_out_of_range_is_rejected(tmp_path, key, bad):
    p = tmp_path / "settings.json"
    with pytest.raises(ValueError):
        s.set_field(key, bad, p)
    assert key not in s._us.load(p).get("app", {})       # nothing written


def test_unknown_setting_and_language_rejected(tmp_path):
    p = tmp_path / "settings.json"
    with pytest.raises(KeyError):
        s.set_field("nonsense", 1, p)
    with pytest.raises(ValueError):
        s.set_field("language_primary", "jp", p)


def test_language_pair_must_differ(tmp_path):
    p = tmp_path / "settings.json"
    with pytest.raises(ValueError):
        s.set_language_pair("en", "en", p)
    s.set_language_pair("en", "de", p)
    got = s.get_all(p)
    assert got["language_primary"] == "en" and got["language_secondary"] == "de"


def test_set_field_also_enforces_the_pair_differs(tmp_path):
    # The invariant must hold on a single-field set too (default primary is "en").
    p = tmp_path / "settings.json"
    with pytest.raises(ValueError):
        s.set_field("language_secondary", "en", p)
    # a different language is fine, and set_language_pair still swaps atomically
    s.set_field("language_secondary", "de", p)
    s.set_language_pair("mk", "en", p)          # would transiently equal if not atomic
    got = s.get_all(p)
    assert got["language_primary"] == "mk" and got["language_secondary"] == "en"


def test_manual_profile_override_marks_it(tmp_path):
    p = tmp_path / "settings.json"
    s.set_profile("balanced", p)
    got = s.get_all(p)
    assert got["profile"] == "balanced" and got["profile_override"] is True
    with pytest.raises(ValueError):
        s.set_profile("supercomputer", p)


def test_sensitivity_slider_maps_to_threshold_and_back():
    assert s.sensitivity_to_threshold(0) == 0.9          # strict
    assert s.sensitivity_to_threshold(100) == 0.3        # loose
    assert s.sensitivity_to_threshold(50) == pytest.approx(0.6)
    # round-trips within slider granularity
    for slider in (0, 25, 50, 75, 100):
        assert abs(s.threshold_to_sensitivity(
            s.sensitivity_to_threshold(slider)) - slider) <= 1


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
