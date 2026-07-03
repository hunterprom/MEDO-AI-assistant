"""Long-term facts: store semantics + the remember/recall/forget skills."""

from __future__ import annotations

import pytest

from core.facts import FactsStore
from skills.base import SkillRequest
from skills.memory_skill import ForgetFactSkill, RecallFactsSkill, RememberFactSkill


@pytest.fixture
def store(tmp_path):
    return FactsStore(tmp_path / "facts.sqlite")


def test_roundtrip_and_nocase_dedupe(store):
    assert store.add("I like coffee") is True
    assert store.add("i like COFFEE") is False  # UNIQUE COLLATE NOCASE
    assert store.recent() == ["I like coffee"]


def test_normalization(store):
    assert store.add("  my   dog is called   Rex . ") is True
    assert store.recent() == ["my dog is called Rex"]
    assert store.add("") is False


def test_forget_substring_case_insensitive(store):
    store.add("My sister is Ana")
    store.add("My favourite drink is coffee")
    assert store.forget("SISTER") == 1
    assert store.count() == 1


def test_recent_returns_newest_capped_oldest_first(store):
    for i in range(25):
        store.add(f"fact number {i}")
    got = store.recent(20)
    assert len(got) == 20
    assert got[0] == "fact number 5"  # oldest of the newest 20 reads first
    assert got[-1] == "fact number 24"


def _req(skill, text):
    """Build a SkillRequest the way the router does: with the regex match."""
    return SkillRequest(text=text, match=skill.match(text))


async def test_remember_and_recall_skills(store):
    remember = RememberFactSkill(store)
    recall = RecallFactsSkill(store, limit=20)

    r = await remember.execute(_req(remember, "remember that my sister is Ana"))
    assert "remember" in r.speech.lower()
    assert store.recent() == ["my sister is Ana"]

    r = await remember.execute(_req(remember, "remember that my sister is Ana"))
    assert "already" in r.speech.lower()

    r = await recall.execute(_req(recall, "what do you remember about me"))
    assert "my sister is Ana" in r.speech


async def test_forget_skill_refuses_vague_pronouns(store):
    store.add("something precious")
    forget = ForgetFactSkill(store)
    r = await forget.execute(_req(forget, "forget it"))
    assert r.success is False
    assert store.count() == 1  # nothing deleted on a vague request

    r = await forget.execute(_req(forget, "forget about something precious"))
    assert r.success and store.count() == 0


async def test_remember_pattern_is_start_anchored():
    # "what do you remember" must not match RememberFactSkill's pattern.
    skill = RememberFactSkill(FactsStore(":memory:"))
    assert skill.match("what do you remember about me") is None
    assert skill.match("remember that I ski") is not None
