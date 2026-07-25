"""The SEMANTIC TIER as a first-class setting: off | shadow | live.

The tier's three honest states are encoded by the ``(semantic_enabled,
semantic_shadow)`` flag pair. A user should never have to reason about that pair,
so the HUD CONFIG tab and the companion API speak in ONE word. This module locks
in that mapping, that a bad word is rejected (never silently ignored), that the
choice persists per machine across a restart, and that ``POST /control/semantic``
flips the live router while ``/status`` reports the current mode.

The routing BEHAVIOUR of each state (declines, shadow-logs, dispatches) is
covered in test_semantic_router.py; here we only prove the settings surface.
"""

from __future__ import annotations

import pytest

from core.config import RouterConfig, load_settings


# --- the one-word mode <-> flag-pair mapping ---------------------------------

def test_semantic_mode_reads_the_flag_pair():
    cfg = RouterConfig()
    cfg.semantic_enabled, cfg.semantic_shadow = True, True
    assert cfg.semantic_mode() == "shadow"
    cfg.semantic_shadow = False
    assert cfg.semantic_mode() == "live"
    cfg.semantic_enabled = False
    assert cfg.semantic_mode() == "off"          # off wins regardless of shadow


def test_apply_semantic_mode_sets_the_flags():
    cfg = RouterConfig()
    assert cfg.apply_semantic_mode("live") == "live"
    assert cfg.semantic_enabled is True and cfg.semantic_shadow is False
    assert cfg.apply_semantic_mode("shadow") == "shadow"
    assert cfg.semantic_enabled is True and cfg.semantic_shadow is True
    assert cfg.apply_semantic_mode("  OFF ") == "off"    # case/space tolerant
    assert cfg.semantic_enabled is False


def test_apply_semantic_mode_rejects_garbage():
    cfg = RouterConfig()
    for bad in ("", "on", "enabled", "true", "semantic"):
        with pytest.raises(ValueError):
            cfg.apply_semantic_mode(bad)


# --- per-machine persistence (survives a restart, never in committed config) -

def test_semantic_mode_persists_in_local_secrets(tmp_path):
    from core.config import apply_local_secrets, save_semantic_mode

    path = tmp_path / "secrets.local.yaml"
    assert save_semantic_mode("live", path) == "live"
    settings = load_settings()                          # defaults: enabled+shadow
    settings.router.apply_semantic_mode("shadow")       # a known starting point
    apply_local_secrets(settings, path)
    assert settings.router.semantic_mode() == "live"    # the saved choice wins


def test_saved_off_survives_a_restart(tmp_path):
    from core.config import apply_local_secrets, save_semantic_mode

    path = tmp_path / "secrets.local.yaml"
    save_semantic_mode("off", path)
    settings = load_settings()
    apply_local_secrets(settings, path)
    assert settings.router.semantic_enabled is False
    assert settings.router.semantic_mode() == "off"


def test_save_semantic_mode_rejects_garbage_before_writing(tmp_path):
    from core.config import save_semantic_mode

    path = tmp_path / "secrets.local.yaml"
    with pytest.raises(ValueError):
        save_semantic_mode("sideways", path)
    assert not path.exists()                    # nothing written on a bad mode


# --- the companion API endpoint + /status ------------------------------------

@pytest.mark.asyncio
async def test_semantic_endpoint_flips_the_tier(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer

    from core.events import EventBus, StateMachine
    from core.router import Router
    from llm.client import OllamaClient
    from remote.server import RemoteServer
    from skills.base import SkillRegistry

    settings = load_settings()
    settings.router.apply_semantic_mode("shadow")       # known starting point
    router = Router(settings, SkillRegistry(),
                    OllamaClient(settings.llm), EventBus())
    router.model = None
    server = RemoteServer(settings, router, StateMachine(EventBus()))
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    try:
        # /status advertises the current mode.
        status = await (await client.get("/status")).json()
        assert status["semantic"] == "shadow"

        # Go LIVE — the live router's flags change on the spot.
        resp = await client.post("/control/semantic", json={"mode": "live"})
        assert resp.status == 200 and (await resp.json())["mode"] == "live"
        assert settings.router.semantic_enabled is True
        assert settings.router.semantic_shadow is False
        assert (await (await client.get("/status")).json())["semantic"] == "live"

        # Turn it OFF.
        resp = await client.post("/control/semantic", json={"mode": "off"})
        assert (await resp.json())["mode"] == "off"
        assert settings.router.semantic_enabled is False

        # A bad mode is a 400, and the setting is left exactly as it was.
        resp = await client.post("/control/semantic", json={"mode": "sideways"})
        assert resp.status == 400
        assert settings.router.semantic_mode() == "off"

        # A missing mode is also a 400 (not a silent no-op).
        resp = await client.post("/control/semantic", json={})
        assert resp.status == 400
    finally:
        await client.close()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
