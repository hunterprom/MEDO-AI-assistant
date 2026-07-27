"""The HUD is an installable PWA: manifest + service worker + icons.

Chosen over a Flutter rewrite (LLM-council decision): it delivers the "make it
an app" intent — installable, home-screen icon, standalone launch, offline
shell — with zero new toolchain and against no duplicated API contract. The
service worker only caches the same-origin shell; the cross-origin companion
API/MJPEG and the SSE feed are always network. Revert by deleting the files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import load_settings
from core.events import EventBus
from ui.hud import HudServer

_WEB = Path(__file__).resolve().parent.parent / "ui" / "web"


# --- the static assets exist and are well-formed -----------------------------

def test_manifest_file_is_valid_and_installable():
    data = json.loads((_WEB / "manifest.json").read_text(encoding="utf-8"))
    assert data["display"] == "standalone"
    assert data["start_url"] == "/"
    sizes = {i["sizes"] for i in data["icons"]}
    assert {"192x192", "512x512"} <= sizes            # Chrome installability floor
    assert any(i.get("purpose") == "maskable" for i in data["icons"])
    for icon in data["icons"]:
        assert (_WEB / icon["src"].lstrip("/")).is_file()


def test_service_worker_never_caches_live_data():
    sw = (_WEB / "sw.js").read_text(encoding="utf-8")
    # Same-origin gate + SSE bypass are what keep live data off the cache.
    assert "self.location.origin" in sw
    assert "/events" in sw


# --- the HUD server serves them ----------------------------------------------

@pytest.mark.asyncio
async def test_hud_serves_pwa_routes():
    from aiohttp.test_utils import TestClient, TestServer

    settings = load_settings()
    hud = HudServer(settings, EventBus())
    client = TestClient(TestServer(hud.build_app()))
    await client.start_server()
    try:
        m = await client.get("/manifest.json")
        assert m.status == 200
        body = await m.json()
        # Name tracks the configured personality so a rename flows through.
        assert body["name"] == settings.personality.name
        assert len(body["icons"]) == 3

        sw = await client.get("/sw.js")
        assert sw.status == 200
        assert sw.headers.get("Service-Worker-Allowed") == "/"
        assert "javascript" in sw.headers.get("Content-Type", "")

        icon = await client.get("/icons/medo-192.png")
        assert icon.status == 200
        assert icon.headers.get("Content-Type") == "image/png"

        html = await (await client.get("/")).text()
        assert 'rel="manifest"' in html
        assert "register('/sw.js')" in html
    finally:
        await client.close()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
