"""M9 MEDO Link: manifest validation, hot-registration, dispatch, auth."""

from __future__ import annotations

import asyncio
import json

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from core.config import load_settings
from core.events import EventBus, StateMachine
from core.router import Router
from link.registry import DeviceCapabilitySkill, LinkRegistry, validate_manifest
from llm.client import OllamaClient
from remote.server import RemoteServer
from skills.base import SkillRegistry

LAMP = {
    "device_id": "lamp",
    "name": "the desk lamp",
    "capabilities": [
        {"name": "on", "description": "Turn the desk lamp on.",
         "fast_patterns": [r"\blamp on\b"]},
        {"name": "off", "description": "Turn the desk lamp off."},
    ],
}

ROBODOG = json.loads(
    (__import__("pathlib").Path(__file__).parent.parent
     / "link" / "examples" / "robodog-manifest.json").read_text(encoding="utf-8"))


# --- manifest validation -----------------------------------------------------


def test_valid_manifests_pass():
    assert validate_manifest(LAMP) == []
    assert validate_manifest(ROBODOG) == []  # the shipped example must be valid


def test_invalid_manifests_report_every_problem():
    errors = validate_manifest({
        "device_id": "Bad Id!", "name": "",
        "transport": "carrier_pigeon",
        "capabilities": [{"name": "DO IT", "description": "",
                          "fast_patterns": ["("], "requires_confirmation": "yes"}],
    })
    joined = " ".join(errors)
    for fragment in ("device_id", "name", "transport", "capabilities[0].name",
                     "description", "not a regex", "requires_confirmation"):
        assert fragment in joined, fragment
    assert validate_manifest("not a dict") == ["manifest must be a JSON object"]
    assert "capabilities" in " ".join(validate_manifest(
        {"device_id": "x1", "name": "x", "capabilities": []}))


def test_redos_alternation_and_optional_patterns_are_rejected():
    # Beyond (a+)+, the heuristic must also reject optional- and alternation-
    # based catastrophic backtracking; search()'d against every utterance these
    # would hang the single-threaded router (e.g. "aaaaaaaaaaaaaaaa!" vs
    # "(a|a|a)+$"). They compile fine, so re.compile alone won't stop them.
    for evil in ["(a|a|a)+$", "(a?)+", "(a?)*", "([a-z]|[a-z])+"]:
        cap = {"name": "go", "description": "do it", "fast_patterns": [evil]}
        errors = validate_manifest({**LAMP, "capabilities": [cap]})
        assert any("catastrophic-backtracking" in e for e in errors), evil
    # A quantified *safe* alternation (no overlap, followed by \b) still passes.
    ok = {"name": "go", "description": "do it",
          "fast_patterns": [r"\bdog,?\s+(stand|up)\b"]}
    assert validate_manifest({**LAMP, "capabilities": [ok]}) == []


# --- registration -> tools + fast patterns ----------------------------------


def _link(tmp_path, timeout_s: float = 0.5) -> tuple[LinkRegistry, SkillRegistry]:
    skills = SkillRegistry()
    return LinkRegistry(skills, tmp_path / "link.db",
                        command_timeout_s=timeout_s), skills


def test_registration_creates_tools_and_patterns(tmp_path):
    link, skills = _link(tmp_path)
    assert link.register(LAMP) == []
    tool_names = [s["function"]["name"] for s in skills.tool_schemas()]
    assert "lamp_on" in tool_names and "lamp_off" in tool_names
    hit = skills.find_match("turn the lamp on please")
    assert hit is not None and hit[0].name == "lamp_on"
    # The capability description reaches the model, prefixed with the device.
    schema = skills.get("lamp_on").tool_schema()
    assert "desk lamp" in schema["function"]["description"]


def test_reregistration_replaces_instead_of_duplicating(tmp_path):
    link, skills = _link(tmp_path)
    link.register(LAMP)
    smaller = {**LAMP, "capabilities": [LAMP["capabilities"][0]]}
    assert link.register(smaller) == []
    assert skills.get("lamp_on") is not None
    assert skills.get("lamp_off") is None  # dropped capability gone


def test_manifests_persist_across_restarts(tmp_path):
    link, _ = _link(tmp_path)
    link.register(LAMP)
    link2, skills2 = _link(tmp_path)
    assert link2.load_persisted() == 1
    assert skills2.get("lamp_on") is not None


def test_confirmation_flag_becomes_a_gated_skill(tmp_path):
    link, skills = _link(tmp_path)
    link.register(ROBODOG)
    walk = skills.get("robodog_walk_forward")
    assert isinstance(walk, DeviceCapabilitySkill)
    assert walk.requires_confirmation is True
    assert skills.get("robodog_sit").requires_confirmation is False


# --- dispatch ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_offline_device_says_so_without_dispatching(tmp_path):
    link, _ = _link(tmp_path)
    link.register(LAMP)  # registered but never polled -> offline
    result = await link.dispatch("lamp", "on", {})
    assert not result.success and "offline" in result.speech


@pytest.mark.asyncio
async def test_dispatch_roundtrip_over_polling(tmp_path):
    link, _ = _link(tmp_path, timeout_s=2.0)
    link.register(LAMP)
    link.touch("lamp")  # device just polled -> online

    async def device_side():
        for _ in range(50):
            commands = link.drain_commands("lamp")
            if commands:
                cmd = commands[0]
                assert cmd["capability"] == "on"
                link.resolve("lamp", cmd["id"], True, "Lamp is on.")
                return
            await asyncio.sleep(0.01)

    task = asyncio.create_task(device_side())
    result = await link.dispatch("lamp", "on", {})
    await task
    assert result.success and result.speech == "Lamp is on."


@pytest.mark.asyncio
async def test_dispatch_timeout_speaks_no_response(tmp_path):
    link, _ = _link(tmp_path, timeout_s=0.1)
    link.register(LAMP)
    link.touch("lamp")
    result = await link.dispatch("lamp", "on", {})  # nobody answers
    assert not result.success and "respond" in result.speech


@pytest.mark.asyncio
async def test_confirmed_capability_rides_the_safety_gate(tmp_path):
    settings = load_settings()
    registry = SkillRegistry()
    link = LinkRegistry(registry, tmp_path / "link.db", command_timeout_s=1.0)
    link.register(ROBODOG)
    link.touch("robodog")
    router = Router(settings, registry, OllamaClient(settings.llm), EventBus())
    router.model = None

    async def device_side():
        for _ in range(100):
            commands = link.drain_commands("robodog")
            if commands:
                link.resolve("robodog", commands[0]["id"], True, "Demo complete.")
                return
            await asyncio.sleep(0.01)

    r1 = await router.route("robodog demo")     # fast pattern from the manifest
    assert router.awaiting_confirmation is True
    assert "sure" in r1.speech.lower()          # literal safety prompt, no quip
    task = asyncio.create_task(device_side())
    r2 = await router.route("yes")
    await task
    assert r2.speech == "Demo complete."


# --- HTTP endpoints (auth + contract) ---------------------------------------


def _server(tmp_path, *, token: str = "link-test-token") -> RemoteServer:
    settings = load_settings()
    settings.remote.auth_enabled = True
    settings.remote.token = token
    registry = SkillRegistry()
    router = Router(settings, registry, OllamaClient(settings.llm), EventBus())
    router.model = None
    link = LinkRegistry(registry, tmp_path / "link.db", command_timeout_s=0.5)
    return RemoteServer(settings, router, StateMachine(EventBus()), link=link)


@pytest_asyncio.fixture
async def lan(tmp_path):
    server = _server(tmp_path)
    server._is_local = lambda request: False  # simulate a LAN device
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    yield client
    await client.close()


AUTH = {"Authorization": "Bearer link-test-token"}


@pytest.mark.asyncio
async def test_register_requires_the_token(lan: TestClient):
    resp = await lan.post("/link/register", json=LAMP)
    assert resp.status == 401                       # no token -> rejected
    resp = await lan.post("/link/register", json=LAMP, headers=AUTH)
    assert resp.status == 200
    body = await resp.json()
    assert body["device_id"] == "lamp" and body["capabilities"] == 2


@pytest.mark.asyncio
async def test_register_rejects_bad_manifests_with_errors(lan: TestClient):
    resp = await lan.post("/link/register", json={"device_id": "!"}, headers=AUTH)
    assert resp.status == 422
    assert (await resp.json())["errors"]


@pytest.mark.asyncio
async def test_poll_result_and_device_list_contract(lan: TestClient):
    await lan.post("/link/register", json=LAMP, headers=AUTH)
    resp = await lan.get("/link/commands/lamp", headers=AUTH)
    assert resp.status == 200 and (await resp.json())["commands"] == []
    resp = await lan.get("/link/commands/ghost", headers=AUTH)
    assert resp.status == 404
    resp = await lan.post("/link/result", headers=AUTH, json={
        "device_id": "lamp", "id": "nope", "ok": True, "message": "hi"})
    assert (await resp.json())["matched"] is False  # unknown command id
    resp = await lan.get("/link/devices", headers=AUTH)
    devices = (await resp.json())["devices"]
    assert devices[0]["name"] == "the desk lamp"
    assert devices[0]["online"] is True             # it polled a moment ago
    assert devices[0]["capabilities"] == 2
