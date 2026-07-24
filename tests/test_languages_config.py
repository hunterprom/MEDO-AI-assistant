"""Constrained two-language mode — S1: config model + validation.

MEDO runs with AT MOST 2 active languages (default en+mk); detection is later
constrained to that pair. These pin the config-layer contract: the cap, the
membership rules, the primary rule, and the back-compat synthesis from the old
stt.* fields.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config import LanguagesConfig, STTConfig, load_settings


# --- validation rules --------------------------------------------------------

def test_two_active_ok():
    c = LanguagesConfig(active=["en", "mk"])
    assert c.active == ["en", "mk"]
    assert c.primary == "en"          # first active
    assert c.detection == "auto_pair"


def test_single_active_ok():
    c = LanguagesConfig(active=["mk"])
    assert c.active == ["mk"]
    assert c.primary == "mk"


def test_three_active_rejected():
    with pytest.raises(ValidationError, match="at most 2"):
        LanguagesConfig(active=["en", "mk", "de"])


def test_empty_active_rejected():
    with pytest.raises(ValidationError, match="at least one"):
        LanguagesConfig(active=[])


def test_active_not_in_available_rejected():
    with pytest.raises(ValidationError, match="not in languages.available"):
        LanguagesConfig(active=["en", "de"], available=["en", "mk"])


def test_primary_not_in_active_rejected():
    with pytest.raises(ValidationError, match="must be one of the active"):
        LanguagesConfig(active=["en", "mk"], primary="de")


def test_omitted_primary_defaults_to_first_active():
    assert LanguagesConfig(active=["es", "de"]).primary == "es"


def test_unknown_available_code_dropped():
    c = LanguagesConfig(active=["en"], available=["en", "mk", "zz", "xx"])
    assert "zz" not in c.available and "xx" not in c.available
    assert "en" in c.available and "mk" in c.available


def test_detection_literal_rejects_bad_value():
    with pytest.raises(ValidationError):
        LanguagesConfig(active=["en"], detection="detect_everything")


def test_codes_are_normalized_and_deduped():
    c = LanguagesConfig(active=[" EN ", "en", "MK"])
    assert c.active == ["en", "mk"]   # lower-cased, de-duplicated, order kept


# --- back-compat: synthesize from legacy stt.* fields ------------------------

def test_from_legacy_forced_language_mode_is_fixed():
    stt = STTConfig(language_mode="de")
    c = LanguagesConfig.from_legacy(stt)
    assert c.detection == "fixed"
    assert c.active == ["de"] and c.primary == "de"


def test_from_legacy_derives_pair_from_spoken_languages():
    stt = STTConfig(spoken_languages=["en", "es", "de"])   # 3 -> first 2 supported
    c = LanguagesConfig.from_legacy(stt)
    assert c.active == ["en", "es"]
    assert c.detection == "auto_pair"


def test_from_legacy_falls_back_to_en_mk():
    c = LanguagesConfig.from_legacy(STTConfig())
    assert c.active == ["en", "mk"]


# --- Settings integration ----------------------------------------------------

def test_shipped_config_uses_explicit_block():
    s = load_settings()
    assert s.active_languages() == ["en", "mk"]
    assert s.primary_language() == "en"
    assert s.detection_mode() == "auto_pair"


def _write(tmp_path, text: str):
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_no_languages_block_synthesizes_from_stt(tmp_path):
    # An old config with no `languages:` block must behave unchanged.
    cfg = _write(tmp_path, "stt:\n  allowed_languages: [en, es]\n")
    s = load_settings(cfg)
    assert s.active_languages() == ["en", "es"]
    assert s.detection_mode() == "auto_pair"


def test_explicit_block_wins_over_legacy_stt(tmp_path):
    cfg = _write(
        tmp_path,
        "languages:\n  active: [en, de]\nstt:\n  allowed_languages: [en, es]\n",
    )
    s = load_settings(cfg)
    assert s.active_languages() == ["en", "de"]   # block wins, not the stt clamp


def test_more_than_two_in_config_block_fails_to_load(tmp_path):
    cfg = _write(tmp_path, "languages:\n  active: [en, mk, de]\n")
    with pytest.raises(ValidationError, match="at most 2"):
        load_settings(cfg)


# --- S4: persistence + companion-API endpoint -------------------------------

def test_save_and_apply_languages_roundtrip(tmp_path):
    from core.config import apply_local_secrets, save_languages

    path = tmp_path / "secrets.local.yaml"
    save_languages(["en", "es"], "es", "fixed", path=path)
    s = load_settings()
    apply_local_secrets(s, path=path)
    assert s.active_languages() == ["en", "es"]
    assert s.primary_language() == "es"
    assert s.detection_mode() == "fixed"


def test_save_languages_rejects_more_than_two(tmp_path):
    from core.config import save_languages
    with pytest.raises(ValidationError, match="at most 2"):
        save_languages(["en", "mk", "de"], path=tmp_path / "s.yaml")


@pytest.mark.asyncio
async def test_languages_endpoint_validates_and_reports_restart():
    from aiohttp.test_utils import TestClient, TestServer

    from core.events import EventBus, StateMachine
    from core.router import Router
    from llm.client import OllamaClient
    from remote.server import RemoteServer
    from skills.base import SkillRegistry

    settings = load_settings()
    router = Router(settings, SkillRegistry(), OllamaClient(settings.llm), EventBus())
    router.model = None
    server = RemoteServer(settings, router, StateMachine(EventBus()))  # persist off
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    try:
        r = await client.post("/control/languages",
                              json={"active": ["en", "es"], "primary": "es"})
        assert r.status == 200
        d = await r.json()
        assert d["active"] == ["en", "es"] and d["restart_required"] is True
        # more than two is rejected with a 400, not silently accepted.
        r2 = await client.post("/control/languages",
                               json={"active": ["en", "mk", "de"]})
        assert r2.status == 400
        # a non-list body is a 400 too.
        r3 = await client.post("/control/languages", json={"active": "en"})
        assert r3.status == 400
    finally:
        await client.close()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
