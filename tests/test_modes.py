"""Session modes: continuous conversation + the modes skill/endpoint."""

from __future__ import annotations

import pytest

from core.modes import SessionModes
from skills.base import SkillRequest
from skills.modes_skill import ModesSkill


# --- the modes skill ---------------------------------------------------------


def test_modes_skill_patterns():
    s = ModesSkill(SessionModes())
    for phrase in ("continuous mode on", "conversation mode off",
                   "keep listening", "stop listening", "interpreter mode",
                   "start interpreting", "stop translating", "be my interpreter"):
        assert s.match(phrase) is not None, phrase
    assert s.match("what time is it") is None


@pytest.mark.asyncio
async def test_continuous_toggle_by_voice():
    modes = SessionModes()
    s = ModesSkill(modes)
    r = await s.execute(SkillRequest(text="continuous mode on"))
    assert modes.continuous is True and "keep listening" in r.speech.lower()
    r = await s.execute(SkillRequest(text="stop listening"))
    assert modes.continuous is False


@pytest.mark.asyncio
async def test_interpreter_toggle_by_voice():
    modes = SessionModes()
    s = ModesSkill(modes)
    r = await s.execute(SkillRequest(text="start interpreting"))
    assert modes.interpreter is True and "translate" in r.speech.lower()
    r = await s.execute(SkillRequest(text="stop interpreting"))
    assert modes.interpreter is False


@pytest.mark.asyncio
async def test_interpreter_wins_over_continuous_when_both_words_present():
    modes = SessionModes()
    s = ModesSkill(modes)
    await s.execute(SkillRequest(text="turn on translation mode"))
    assert modes.interpreter is True and modes.continuous is False


@pytest.mark.asyncio
async def test_tool_path_uses_explicit_mode_and_state():
    modes = SessionModes()
    s = ModesSkill(modes)
    await s.execute(SkillRequest(text="", args={"mode": "continuous", "state": "on"}))
    assert modes.continuous is True
    await s.execute(SkillRequest(text="", args={"mode": "continuous", "state": "off"}))
    assert modes.continuous is False


def test_skill_is_not_pc_gated():
    # Changing MEDO's own listening behaviour must work even with PC control off.
    assert ModesSkill.controls_pc is False


# --- the endpoint + status ---------------------------------------------------


@pytest.mark.asyncio
async def test_modes_endpoint_and_status(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer

    from core.config import load_settings
    from core.events import EventBus, StateMachine
    from core.router import Router
    from llm.client import OllamaClient
    from remote.server import RemoteServer
    from skills.base import SkillRegistry

    settings = load_settings()
    modes = SessionModes()
    router = Router(settings, SkillRegistry(), OllamaClient(settings.llm), EventBus())
    router.model = None
    server = RemoteServer(settings, router, StateMachine(EventBus()), modes=modes)
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    try:
        resp = await client.post("/control/modes", json={"mode": "continuous", "on": True})
        assert resp.status == 200 and (await resp.json())["on"] is True
        assert modes.continuous is True
        status = await (await client.get("/status")).json()
        assert status["continuous"] is True and status["interpreter"] is False
        resp = await client.post("/control/modes", json={"mode": "nope", "on": True})
        assert resp.status == 400
    finally:
        await client.close()


# --- continuous mode in the voice loop (re-listen decision) ------------------


def test_voiceloop_defaults_modes_from_config():
    from core.config import load_settings
    from voice.loop import VoiceLoop

    settings = load_settings()
    settings.conversation.continuous = True
    loop = VoiceLoop(settings, router=object(), sm=object(), announcer=object(),
                     ui=object(), log=object())
    assert loop._modes.continuous is True
