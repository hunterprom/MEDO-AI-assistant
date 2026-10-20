"""P1-4: a nonsensical config.yaml is caught at startup, not hours later.

Every fatal case here is a value pydantic accepts happily — the right type,
the wrong meaning. The point of the check is that MEDO says so while you are
still looking at the terminal, instead of going quiet in a way that looks like
a bug in the feature you configured.
"""

from __future__ import annotations

import pytest

from core.config import load_settings
from core.config_check import (
    ConfigError,
    check_settings,
    format_problems,
    validate_settings,
)


def _fatal_keys(settings) -> set[str]:
    return {p.key for p in check_settings(settings) if p.fatal}


def _warning_keys(settings) -> set[str]:
    return {p.key for p in check_settings(settings) if not p.fatal}


def test_the_shipped_config_is_valid():
    """If this ever fails, MEDO does not start for anyone."""
    assert check_settings(load_settings(validate=False)) == []


def test_load_settings_validates_by_default():
    load_settings()          # raises ConfigError if the shipped config is bad


# --- fatal: values MEDO cannot run with --------------------------------------


def test_a_council_of_nobody_is_refused():
    settings = load_settings(validate=False)
    settings.council.max_members = 0
    assert "council.max_members" in _fatal_keys(settings)


def test_a_similarity_given_as_a_percentage_is_refused():
    """The classic: 60 instead of 0.6. Nothing would ever match again."""
    settings = load_settings(validate=False)
    settings.router.semantic_threshold = 60
    assert "router.semantic_threshold" in _fatal_keys(settings)


def test_an_impossible_port_is_refused():
    settings = load_settings(validate=False)
    settings.remote.port = 70000
    assert "remote.port" in _fatal_keys(settings)


def test_zero_conversation_turns_is_refused():
    settings = load_settings(validate=False)
    settings.memory.max_turns = 0
    assert "memory.max_turns" in _fatal_keys(settings)


def test_a_certificate_without_its_key_is_refused():
    settings = load_settings(validate=False)
    settings.remote.tls_certfile = "/etc/medo/cert.pem"
    settings.remote.tls_keyfile = ""
    assert "remote.tls_certfile / remote.tls_keyfile" in _fatal_keys(settings)


def test_a_matched_pair_of_tls_paths_is_accepted(tmp_path):
    settings = load_settings(validate=False)
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    cert.write_text("x", encoding="utf-8")
    key.write_text("x", encoding="utf-8")
    settings.remote.tls_certfile = str(cert)
    settings.remote.tls_keyfile = str(key)
    assert _fatal_keys(settings) == set()


def test_an_open_api_on_a_network_interface_is_refused():
    """Auth off plus a non-loopback bind is the whole LAN driving MEDO."""
    settings = load_settings(validate=False)
    settings.remote.enabled = True
    settings.remote.host = "0.0.0.0"
    settings.remote.auth_enabled = False
    assert "remote.auth_enabled" in _fatal_keys(settings)


def test_an_open_api_on_loopback_is_fine():
    settings = load_settings(validate=False)
    settings.remote.enabled = True
    settings.remote.host = "127.0.0.1"
    settings.remote.auth_enabled = False
    assert "remote.auth_enabled" not in _fatal_keys(settings)


def test_a_zero_step_agent_is_refused():
    settings = load_settings(validate=False)
    settings.browser.max_steps = 0
    settings.vision.agent_max_steps = 0
    fatal = _fatal_keys(settings)
    assert {"browser.max_steps", "vision.agent_max_steps"} <= fatal


def test_self_dev_with_an_empty_denylist_is_refused():
    """An empty denylist lets MEDO rewrite its own safety code."""
    settings = load_settings(validate=False)
    settings.self_dev.enabled = True
    settings.self_dev.denylist = []
    assert "self_dev.denylist" in _fatal_keys(settings)


def test_a_negative_correction_window_is_refused():
    settings = load_settings(validate=False)
    settings.logging.correction_window_s = -1
    assert "logging.correction_window_s" in _fatal_keys(settings)


def test_negative_rate_limits_are_refused():
    settings = load_settings(validate=False)
    settings.remote.rate_limit_per_minute = -5
    assert "remote.rate_limit_per_minute" in _fatal_keys(settings)


def test_zero_rate_limit_is_allowed_as_the_off_switch():
    settings = load_settings(validate=False)
    settings.remote.rate_limit_per_minute = 0
    assert "remote.rate_limit_per_minute" not in _fatal_keys(settings)


# --- warnings: surprising, but MEDO can still run honestly -------------------


def test_developer_mode_warns_without_blocking_startup():
    settings = load_settings(validate=False)
    settings.security.yolo = True
    assert "security.yolo" in _warning_keys(settings)
    validate_settings(settings)          # warns, does not raise


def test_confirmations_off_warns():
    settings = load_settings(validate=False)
    settings.safety.confirm_destructive = False
    assert "safety.confirm_destructive" in _warning_keys(settings)


def test_auto_apply_warns():
    settings = load_settings(validate=False)
    settings.self_dev.enabled = True
    settings.self_dev.auto_apply = True
    assert "self_dev.auto_apply" in _warning_keys(settings)


def test_a_huge_council_warns_about_the_latency():
    settings = load_settings(validate=False)
    settings.council.max_members = 12
    assert "council.max_members" in _warning_keys(settings)


def test_disabling_a_specialist_that_does_not_exist_warns():
    settings = load_settings(validate=False)
    settings.council.disabled = ["astrologer"]
    assert "council.disabled" in _warning_keys(settings)


def test_disabling_a_real_specialist_does_not_warn():
    settings = load_settings(validate=False)
    settings.council.disabled = ["electrical"]
    assert "council.disabled" not in _warning_keys(settings)


def test_an_extra_specialist_without_a_prompt_warns():
    settings = load_settings(validate=False)
    settings.council.extra = {"astrologer": {"title": "the astrologer"}}
    assert "council.extra.astrologer" in _warning_keys(settings)


def test_too_much_conversation_history_warns():
    settings = load_settings(validate=False)
    settings.memory.max_turns = 500
    assert "memory.max_turns" in _warning_keys(settings)


# --- the message a person actually reads -------------------------------------


def test_every_problem_is_reported_at_once():
    """One error per restart is miserable when a restart reloads a 18 GB model."""
    settings = load_settings(validate=False)
    settings.council.max_members = 0
    settings.memory.max_turns = 0
    settings.remote.port = 99999
    with pytest.raises(ConfigError) as raised:
        validate_settings(settings)
    message = str(raised.value)
    assert "council.max_members" in message
    assert "memory.max_turns" in message
    assert "remote.port" in message


def test_the_message_names_the_key_the_value_and_the_fix():
    settings = load_settings(validate=False)
    settings.council.max_members = 0
    message = format_problems(check_settings(settings))
    assert "council.max_members" in message      # the key, as written in YAML
    assert "0" in message                        # what it is set to
    assert "at least 1" in message               # what would be acceptable


def test_warnings_alone_never_raise():
    settings = load_settings(validate=False)
    settings.security.yolo = True
    settings.safety.confirm_destructive = False
    assert validate_settings(settings)           # problems were found
    # ...and startup continued: no exception escaped.


def test_strict_false_reports_without_raising():
    settings = load_settings(validate=False)
    settings.council.max_members = 0
    problems = validate_settings(settings, strict=False)
    assert any(p.fatal for p in problems)
