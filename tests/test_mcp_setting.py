"""Add an MCP server from the HUD: name + URL + optional bearer auth.

The auth token is a secret, so an HUD-added server is persisted to the
git-ignored overrides (never config.yaml) and re-applied at startup, merged onto
any config.yaml servers. HTTP transport sends it as an Authorization header.
"""

from __future__ import annotations

import pytest

from core.config import (
    MCPServerConfig,
    apply_local_secrets,
    load_settings,
    remove_mcp_server,
    save_mcp_server,
)


# --- config field + persistence ---------------------------------------------

def test_mcp_server_has_auth_token():
    s = MCPServerConfig(url="http://x/mcp", auth_token="sek")
    assert s.auth_token == "sek"


def test_save_requires_name_and_endpoint(tmp_path):
    path = tmp_path / "secrets.local.yaml"
    with pytest.raises(ValueError):
        save_mcp_server("", url="http://x/mcp", path=path)      # no name
    with pytest.raises(ValueError):
        save_mcp_server("x", path=path)                         # no url or command


def test_save_and_apply_roundtrip(tmp_path):
    path = tmp_path / "secrets.local.yaml"
    summary = save_mcp_server("home-assistant", url="http://localhost:3000/mcp",
                              auth_token="tok-123", path=path)
    assert summary == {"name": "home-assistant", "url": "http://localhost:3000/mcp",
                       "command": "", "has_auth": True}
    s = apply_local_secrets(load_settings(), path)
    srv = s.mcp.servers["home-assistant"]
    assert srv.url == "http://localhost:3000/mcp"
    assert srv.auth_token == "tok-123"                          # secret, applied at startup
    assert srv.enabled is True


def test_remove(tmp_path):
    path = tmp_path / "secrets.local.yaml"
    save_mcp_server("x", url="http://x/mcp", path=path)
    assert remove_mcp_server("x", path=path) is True
    assert remove_mcp_server("x", path=path) is False          # already gone
    s = apply_local_secrets(load_settings(), path)
    assert "x" not in s.mcp.servers


# --- the companion endpoints -------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_add_and_remove_endpoints():
    from aiohttp.test_utils import TestClient, TestServer

    from core.events import EventBus, StateMachine
    from core.router import Router
    from llm.client import OllamaClient
    from remote.server import RemoteServer
    from skills.base import SkillRegistry

    settings = load_settings()
    router = Router(settings, SkillRegistry(), OllamaClient(settings.llm), EventBus())
    router.model = None
    server = RemoteServer(settings, router, StateMachine(EventBus()))   # persist off
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    try:
        resp = await client.post("/control/mcp", json={
            "name": "ha", "url": "http://localhost:3000/mcp", "auth": "tok"})
        assert resp.status == 200
        srv = settings.mcp.servers["ha"]
        assert srv.url == "http://localhost:3000/mcp" and srv.auth_token == "tok"

        # missing endpoint -> 400
        assert (await client.post("/control/mcp", json={"name": "bad"})).status == 400

        # remove it
        resp = await client.post("/control/mcp/remove", json={"name": "ha"})
        assert resp.status == 200
        assert "ha" not in settings.mcp.servers
    finally:
        await client.close()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
