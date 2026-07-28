"""M4: conversation context, reminder persistence, and prompt injection."""

from __future__ import annotations

import pytest

from core.config import load_settings
from core.events import EventBus, RoutePath
from core.memory import ConversationMemory, ReminderStore
from core.router import Router
from skills.base import SkillRegistry
from skills.datetime_skill import DateTimeSkill


# --- ConversationMemory ------------------------------------------------------
def test_conversation_rolls_and_formats():
    mem = ConversationMemory(max_turns=2)
    mem.add_turn("hi", "hello")
    mem.add_turn("weather?", "31 and sunny")
    mem.add_turn("news?", "quiet day")           # evicts the first turn
    msgs = mem.recent_messages()
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "weather?"       # "hi" rolled off
    mem.add_turn("", "ignored")                    # empty turns are dropped
    assert len(mem.recent_messages()) == 4


# --- ReminderStore -----------------------------------------------------------
def test_reminder_store_roundtrip(tmp_path):
    store = ReminderStore(tmp_path / "r.db")
    r = store.add("2030-01-01T09:00:00+00:00", "call mom")
    assert [x.label for x in store.all()] == ["call mom"]
    assert store.delete(r.id) is True
    assert store.all() == []
    store.close()


# --- context reaches the LLM path -------------------------------------------
class FakeLLM:
    """Records the messages it's asked to complete; returns a fixed reply."""

    def __init__(self) -> None:
        self.last_messages = None

    def is_available(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return ["fake"]

    async def chat(self, model, messages, *, tools=None, temperature=None):
        self.last_messages = messages
        return {"content": "Noted.", "tool_calls": []}


class LeakyLLM:
    """First call (with tools) leaks a botched tool-call as text; retry is clean."""

    def is_available(self):
        return True

    def list_models(self):
        return ["fake"]

    async def chat(self, model, messages, *, tools=None, temperature=None):
        if tools:
            return {"content": '{"name": "introduce", "parameters": {"action": "say"}}',
                    "tool_calls": []}
        return {"content": "I'm MEDO, at your service.", "tool_calls": []}


@pytest.mark.asyncio
async def test_leaked_tool_json_is_recovered():
    settings = load_settings()
    registry = SkillRegistry()
    registry.register(DateTimeSkill())  # gives the model a tool schema
    router = Router(settings, registry, LeakyLLM(), EventBus())
    router.model = "fake"

    result = await router.route("tell me about yourself")
    assert result.path is RoutePath.LLM
    assert "{" not in result.speech
    assert "MEDO" in result.speech


@pytest.mark.asyncio
async def test_prior_turn_is_injected_into_llm_prompt():
    settings = load_settings()
    registry = SkillRegistry()
    registry.register(DateTimeSkill())
    llm = FakeLLM()
    router = Router(settings, registry, llm, EventBus())
    router.model = "fake"

    first = await router.route("what time is it")   # fast path, recorded
    assert first.path is RoutePath.FAST

    await router.route("and after that?")            # LLM path
    contents = [m["content"] for m in llm.last_messages]
    # The earlier exchange must be present as context before the new question.
    assert "what time is it" in contents
    assert any("It's" in c for c in contents)        # the datetime answer
    assert llm.last_messages[-1]["content"] == "and after that?"
    assert llm.last_messages[0]["role"] == "system"
