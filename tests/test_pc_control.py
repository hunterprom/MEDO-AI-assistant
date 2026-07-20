"""The PC CONTROL master switch + the open_website skill."""

from __future__ import annotations

import re

import pytest

from core.config import load_settings
from core.events import EventBus, RoutePath
from core.router import PC_CONTROL_OFF_REPLY, Router
from llm.client import OllamaClient
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult
from skills.web_open import OpenWebsiteSkill, to_url


# --- open_website ------------------------------------------------------------


def test_to_url_sites_and_searches():
    assert to_url("tinkercad.com") == "https://tinkercad.com"
    assert to_url("https://docs.python.org/3/") == "https://docs.python.org/3/"
    assert to_url("arduino uno pinout") == \
        "https://duckduckgo.com/?q=arduino+uno+pinout"


def test_open_website_patterns():
    s = OpenWebsiteSkill(opener=lambda url: True)
    for phrase in ("open tinkercad.com", "go to docs.python.org/3",
                   "open the website tinkercad",
                   "search for arduino sensors in the browser"):
        assert s.match(phrase) is not None, phrase
    # App launches and plain questions are not stolen.
    assert s.match("open chrome") is None
    assert s.match("search for pasta recipes") is None  # server-side web search


@pytest.mark.asyncio
async def test_open_website_opens_and_reports():
    opened = []
    s = OpenWebsiteSkill(opener=lambda url: opened.append(url) or True)
    r = await s.execute(SkillRequest(text="open tinkercad.com",
                                     match=s.match("open tinkercad.com")))
    assert r.success and opened == ["https://tinkercad.com"]
    assert "tinkercad.com" in r.speech
    assert s.controls_pc is True  # gated by the PC CONTROL switch


# --- the master switch -------------------------------------------------------


class _ActSkill(Skill):
    name = "apps"          # a controls_pc-style skill
    description = "test actuation"
    controls_pc = True
    patterns = [re.compile(r"\bopen chrome\b", re.IGNORECASE)]

    def __init__(self):
        self.ran = False

    async def execute(self, request: SkillRequest) -> SkillResult:
        self.ran = True
        return SkillResult("Opening Chrome.")


class _AnswerSkill(Skill):
    name = "datetime"      # sensing/answering — never gated
    description = "test answering"
    patterns = [re.compile(r"\bwhat time\b", re.IGNORECASE)]

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("It's noon.")


def _router(pc_on: bool) -> tuple[Router, _ActSkill]:
    settings = load_settings()
    settings.safety.pc_control_enabled = pc_on
    act = _ActSkill()
    registry = SkillRegistry()
    registry.register(act)
    registry.register(_AnswerSkill())
    router = Router(settings, registry, OllamaClient(settings.llm), EventBus())
    router.model = None
    return router, act


@pytest.mark.asyncio
async def test_switch_off_blocks_actuation_without_running_it():
    router, act = _router(pc_on=False)
    r = await router.route("open chrome")
    assert r.speech == PC_CONTROL_OFF_REPLY
    assert act.ran is False
    assert r.path is RoutePath.FAST and r.skill_name == "apps"


@pytest.mark.asyncio
async def test_switch_off_leaves_answering_skills_alone():
    router, _ = _router(pc_on=False)
    r = await router.route("what time is it")
    assert r.speech == "It's noon."


@pytest.mark.asyncio
async def test_switch_on_lets_actuation_run():
    router, act = _router(pc_on=True)
    r = await router.route("open chrome")
    assert act.ran is True and r.speech.startswith("Opening Chrome.")


def test_switch_filters_llm_tools():
    router, _ = _router(pc_on=False)
    from llm.tools import build_tools

    gated = {s.name for s in router._registry.all() if s.controls_pc}
    offered = [t["function"]["name"] for t in build_tools(router._registry)
               if t["function"]["name"] not in gated]
    assert "apps" not in offered and "datetime" in offered


def test_real_actuation_skills_carry_the_flag():
    from skills.desktop import PressKeysSkill, TypeTextSkill
    from skills.system import PowerSkill, VolumeSkill

    for cls in (TypeTextSkill, PressKeysSkill, PowerSkill, VolumeSkill,
                OpenWebsiteSkill):
        assert cls.controls_pc is True, cls.__name__
    from skills.datetime_skill import DateTimeSkill
    from skills.weather import WeatherSkill

    for cls in (DateTimeSkill, WeatherSkill):
        assert cls.controls_pc is False, cls.__name__


# --- endpoint + persistence --------------------------------------------------


@pytest.mark.asyncio
async def test_pc_control_endpoint_flips_the_setting(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer

    from core.events import StateMachine
    from remote.server import RemoteServer

    settings = load_settings()
    settings.safety.pc_control_enabled = True
    registry = SkillRegistry()
    router = Router(settings, registry, OllamaClient(settings.llm), EventBus())
    router.model = None
    server = RemoteServer(settings, router, StateMachine(EventBus()))
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    try:
        resp = await client.post("/control/pc", json={"on": False})
        assert resp.status == 200 and (await resp.json())["on"] is False
        assert settings.safety.pc_control_enabled is False
        status = await (await client.get("/status")).json()
        assert status["pc_control"] is False
        resp = await client.post("/control/pc", json={"on": True})
        assert settings.safety.pc_control_enabled is True
    finally:
        await client.close()


def test_pc_control_persists_in_local_secrets(tmp_path):
    from core.config import apply_local_secrets, save_pc_control

    path = tmp_path / "secrets.local.yaml"
    save_pc_control(False, path)
    settings = load_settings()
    settings.safety.pc_control_enabled = True
    apply_local_secrets(settings, path)
    assert settings.safety.pc_control_enabled is False


@pytest.mark.asyncio
async def test_spoken_dot_com_opens_the_site():
    # Whisper gives "tinkercad dot com" — words — for spoken URLs.
    opened = []
    s = OpenWebsiteSkill(opener=lambda url: opened.append(url) or True)
    for phrase in ("open tinkercad dot com", "go to docs dot python dot org"):
        m = s.match(phrase)
        assert m is not None, phrase
        await s.execute(SkillRequest(text=phrase, match=m))
    assert opened == ["https://tinkercad.com", "https://docs.python.org"]


def test_prompt_forbids_fake_actions():
    from core.config import PersonalityConfig
    from llm.prompts import system_prompt

    prompt = system_prompt(PersonalityConfig())
    assert "NEVER claim you performed an action" in prompt
    assert "open_website" in prompt


@pytest.mark.asyncio
async def test_bare_site_shortcuts_open_deterministically():
    opened = []
    s = OpenWebsiteSkill(opener=lambda url: opened.append(url) or True)
    m = s.match("open tinkercad")
    assert m is not None
    await s.execute(SkillRequest(text="open tinkercad", match=m))
    assert opened == ["https://tinkercad.com"]
    # Curated list only: apps and folders are never stolen.
    for phrase in ("open chrome", "open notepad", "open downloads"):
        assert s.match(phrase) is None, phrase
