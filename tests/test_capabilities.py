"""M21 S1 — the /capabilities feed the orb maps onto nodes."""

from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient, TestServer

from core.agents import capabilities_feed
from core.events import EventBus
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult


class _Sk(Skill):
    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("ok")


def make(name, module="skills.fake", description="does a thing", controls_pc=False):
    cls = type(f"S_{name}", (_Sk,), {
        "name": name, "description": description, "controls_pc": controls_pc})
    cls.__module__ = module
    return cls()


def reg(*skills):
    r = SkillRegistry()
    for s in skills:
        r.register(s)
    return r


# -- shape ---------------------------------------------------------------------

def test_feed_groups_skills_under_their_agent():
    feed = capabilities_feed(
        reg(make("see_screen"), make("web_search"), make("notes")), council=[])
    agents = {a["id"]: a for a in feed["agents"]}
    assert "vision" in agents and "web" in agents and "knowledge" in agents
    vision_skills = {s["id"] for s in agents["vision"]["skills"]}
    assert "see_screen" in vision_skills


def test_each_skill_carries_id_label_status_description():
    feed = capabilities_feed(reg(make("see_screen", description="reads the screen")),
                             council=[])
    skill = feed["agents"][0]["skills"][0]
    assert set(skill) >= {"id", "label", "kind", "status", "controlsPc", "description"}
    assert skill["id"] == "see_screen"
    assert skill["status"] == "idle"          # sane default snapshot status
    assert skill["kind"] == "skill"
    assert skill["description"] == "reads the screen"


def test_controls_pc_is_exposed_for_the_safety_click_flow():
    feed = capabilities_feed(
        reg(make("power", controls_pc=True), make("datetime", controls_pc=False)),
        council=[])
    flat = {s["id"]: s for a in feed["agents"] for s in a["skills"]}
    assert flat["power"]["controlsPc"] is True
    assert flat["datetime"]["controlsPc"] is False


def test_council_specialists_are_agent_kind_leaf_nodes():
    class Spec:
        def __init__(self, key, title): self.key, self.title = key, title
    feed = capabilities_feed(
        reg(make("ask_specialist")),
        council=[Spec("electrical", "the electrical engineer")])
    experts = next(a for a in feed["agents"] if a["id"] == "experts")
    node = experts["skills"][0]
    assert node["kind"] == "agent" and node["label"] == "Electrical"
    assert node["id"] == "experts:electrical"     # stable, slugged


def test_skill_index_maps_every_skill_to_its_agent():
    feed = capabilities_feed(reg(make("see_screen"), make("power")), council=[])
    assert feed["skillIndex"]["see_screen"] == "vision"
    assert feed["skillIndex"]["power"] == "system"


def test_a_plugin_skill_appears_automatically():
    feed = capabilities_feed(
        reg(make("dice", module="medo_plugin_example_dice")), council=[])
    ids = {a["id"] for a in feed["agents"]}
    assert "plugins" in ids
    plugin_skills = {s["id"] for a in feed["agents"] if a["id"] == "plugins"
                     for s in a["skills"]}
    assert "dice" in plugin_skills


# -- the real registry ---------------------------------------------------------

def test_the_live_registry_produces_a_capabilities_feed():
    from core.config import load_settings
    from core.docindex import DocumentIndex
    from main import Announcer, build_registry

    reg_ = build_registry(load_settings(), Announcer(),
                          doc_index=DocumentIndex(":memory:", None, []))
    feed = capabilities_feed(reg_)
    assert feed["agents"] and feed["skillIndex"]
    names = {s.name for s in reg_.all()}
    # every routable skill id in the feed is a real registered skill
    for a in feed["agents"]:
        for s in a["skills"]:
            if s["kind"] == "skill":
                assert s["id"] in names


# -- the HudServer endpoint ----------------------------------------------------

@pytest.mark.asyncio
async def test_capabilities_endpoint_serves_the_feed():
    from core.config import load_settings
    from core.docindex import DocumentIndex
    from main import Announcer, build_registry
    from ui.hud import HudServer

    settings = load_settings()
    registry = build_registry(settings, Announcer(),
                              doc_index=DocumentIndex(":memory:", None, []))
    server = HudServer(settings, EventBus(), registry=registry)
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    try:
        resp = await client.get("/capabilities")
        assert resp.status == 200
        data = await resp.json()
        assert data["agents"] and "skillIndex" in data
        vision = next(a for a in data["agents"] if a["id"] == "vision")
        assert any(s["id"] == "see_screen" for s in vision["skills"])
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_capabilities_endpoint_survives_without_a_registry():
    from core.config import load_settings
    from ui.hud import HudServer

    server = HudServer(load_settings(), EventBus(), registry=None)
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    try:
        resp = await client.get("/capabilities")
        assert resp.status == 200
        assert await resp.json() == {"agents": [], "skillIndex": {}}
    finally:
        await client.close()
