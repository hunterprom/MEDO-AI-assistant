"""Regression tests for bugs found in the deep-debug loop (2026-07-24).

Each test pins a specific verified defect so it can't silently return.
"""

from __future__ import annotations

import pytest

from core.config import STTConfig
from llm.tools import coerce_args
from skills.base import Skill, SkillRequest, SkillResult


# --- S1: a non-string tool arg must not crash .strip() -----------------------

class _ArgSkill(Skill):
    name = "argtool"
    description = "takes a string path"

    def tool_schema(self):
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"},
                                          "amount": {"type": "integer"}},
                           "required": ["path"]}}}

    async def execute(self, request):
        return SkillResult("ok")


def test_coerce_stringifies_string_params_only():
    skill = _ArgSkill()
    out = coerce_args(skill, {"path": 1234, "amount": 5})
    assert out["path"] == "1234"        # string param coerced
    assert out["amount"] == 5           # integer param left alone


def test_coerce_leaves_real_strings_and_none():
    skill = _ArgSkill()
    out = coerce_args(skill, {"path": "notes.md", "amount": None})
    assert out == {"path": "notes.md", "amount": None}


@pytest.mark.parametrize("skill_name,arg", [
    ("import_file", {"path": 123}),
    ("explain_process", {"process": 4567}),
    ("locate_app", {"app": 9}),
    ("install_app", {"app": 42}),
])
def test_real_skills_survive_a_numeric_arg_after_coercion(skill_name, arg):
    # Route through coerce_args (the dispatch boundary) then execute — no crash.
    import asyncio

    from core.config import load_settings
    from core.docindex import DocumentIndex
    from main import Announcer, build_registry

    settings = load_settings()
    settings.mode.lion = True          # so security skills are surfaced/execute
    reg = build_registry(settings, Announcer(),
                         doc_index=DocumentIndex(":memory:", None, []))
    skill = reg.get(skill_name)
    assert skill is not None
    coerced = coerce_args(skill, arg)
    # must not raise AttributeError on .strip()
    result = asyncio.run(skill.execute(
        SkillRequest(text="x", args=coerced, context={"via": "tool"})))
    assert isinstance(result, SkillResult)


# --- S2/S3: offensive detector accuracy --------------------------------------

def test_local_audit_mentioning_a_domain_is_not_offensive():
    from skills.security import is_offensive
    assert is_offensive("scan my computer for connections to facebook.com") is False
    assert is_offensive("check my ports") is False


def test_scanning_another_host_is_still_offensive():
    from skills.security import is_offensive
    assert is_offensive("scan 192.168.1.0/24 for open ports") is True
    assert is_offensive("nmap example.com") is True


@pytest.mark.parametrize("text", [
    "install backdoors", "download payloads", "deploy rootkits",
    "join botnets", "give me reverse shells",
])
def test_plural_attack_words_are_caught(text):
    from skills.security import is_offensive
    assert is_offensive(text) is True


# --- C1: language_mode must not wipe an explicit language --------------------

def test_language_auto_respects_an_explicit_forced_language():
    # The shipped config always writes language_mode: auto; a user forcing
    # language: "mk" must be respected, not silently reset to auto-detect.
    assert STTConfig(language="mk", language_mode="auto").language == "mk"
    assert STTConfig(language=None, language_mode="auto").language is None


# --- R2: the command turn is remembered, the bare "yes" is not ---------------

@pytest.mark.asyncio
async def test_confirmation_records_command_not_the_yes():
    from core.events import EventBus
    from core.config import load_settings
    from llm.client import OllamaClient
    from skills.base import SkillRegistry

    settings = load_settings()

    class Boom(Skill):
        name = "wipe"
        controls_pc = True
        patterns = __import__("re").compile(r"wipe it"),

        async def execute(self, request):
            if request.context.get("confirmed"):
                return SkillResult("Done.")
            return SkillResult("Are you sure?", needs_confirmation=True)

    from core.router import Router
    reg = SkillRegistry()
    reg.register(Boom())
    router = Router(settings, reg, OllamaClient(settings.llm), EventBus())
    router.model = None

    await router.route("wipe it")             # triggers confirmation
    await router.route("yes")                 # answers it
    turns = router.conversation.recent_messages()
    joined = " ".join(m["content"] for m in turns)
    # the command must be in memory; the naked "yes" must NOT be a user turn
    assert "wipe it" in joined
    assert not any(m["role"] == "user" and m["content"].strip().lower() == "yes"
                   for m in turns)


# --- R1: web_search keeps the tool-brain follow-up chain alive ---------------

def test_web_search_is_a_live_info_skill():
    from core.router import LIVE_INFO_SKILLS
    assert "web_search" in LIVE_INFO_SKILLS


@pytest.mark.asyncio
async def test_live_info_flag_set_when_a_live_tool_runs_on_llm_path(monkeypatch):
    # Simulate the CLI-brain path calling web_search; _turn_used_live_info must
    # be set so the NEXT question is treated as a live follow-up.
    from core.events import EventBus
    from core.config import load_settings
    from core.router import Router
    from llm.client import OllamaClient
    from skills.base import SkillRegistry

    settings = load_settings()
    router = Router(settings, SkillRegistry(), OllamaClient(settings.llm), EventBus())
    router._turn_used_live_info = False
    # directly exercise the tracking branch via _run_tool_calls with a fake tool
    calls = [{"function": {"name": "web_search", "arguments": {"query": "x"}}}]

    async def fake_dispatch(reg, name, args, ctx):
        return SkillResult("raw results", success=True)

    monkeypatch.setattr("core.router.dispatch_tool", fake_dispatch)
    await router._run_tool_calls(calls, "who won", {}, [])
    assert router._turn_used_live_info is True


# --- V1: STT confirm keys off the LOADED model's phrase, not the config ------

def test_effective_phrase_falls_back_when_custom_model_missing(tmp_path):
    from voice.wakeword import FALLBACK_PHRASE, effective_phrase
    missing = tmp_path / "hey_medo.onnx"                 # never created
    assert effective_phrase(str(missing)) == FALLBACK_PHRASE
    present = tmp_path / "present.onnx"
    present.write_bytes(b"x")
    assert effective_phrase(str(present)) == str(present)
    assert effective_phrase("hey_jarvis") == "hey_jarvis"


def test_confirm_wake_checks_against_the_loaded_phrase():
    # If the model fell back to hey_jarvis, "hey jarvis" must confirm even
    # though the configured phrase is hey_medo (the fallback's whole point).
    import types

    from core.config import load_settings
    from core.metrics import LatencyLog
    from unittest.mock import MagicMock

    from voice.loop import VoiceLoop
    loop = VoiceLoop(load_settings(), router=MagicMock(), sm=MagicMock(),
                     announcer=MagicMock(), ui=MagicMock(), log=LatencyLog())
    loop._wakeword = types.SimpleNamespace(phrase="hey_jarvis")
    loop._stt = types.SimpleNamespace(transcribe=lambda a: "hey jarvis")
    import numpy as np
    assert loop._confirm_wake(np.zeros(1280, dtype=np.int16)) is True
    loop._stt = types.SimpleNamespace(transcribe=lambda a: "hey medo")
    assert loop._confirm_wake(np.zeros(1280, dtype=np.int16)) is False
