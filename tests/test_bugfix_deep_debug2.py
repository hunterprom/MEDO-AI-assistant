"""Regression tests for bugs found in deep-debug loop iteration 2 (2026-07-24)."""

from __future__ import annotations

import asyncio

import pytest

from skills.base import SkillRequest, SkillResult


# --- CORE1/CORE2: unescaped LIKE wildcards wiped/over-matched everything ------

def test_facts_forget_does_not_wipe_on_a_wildcard(tmp_path):
    from core.facts import FactsStore
    store = FactsStore(tmp_path / "f.db")
    store.add("my dentist is Dr Petrov")
    store.add("I like tea")
    store.add("call the plumber")
    # a wildcard needle matches LITERALLY now, not everything (was: wiped all 3)
    assert store.forget("_") == 0
    assert store.forget("%") == 0
    assert store.count() == 3
    # a real literal substring still works
    assert store.forget("tea") == 1
    assert store.count() == 2


def test_notes_search_wildcard_matches_literally(tmp_path):
    from core.memory import NoteStore
    store = NoteStore(str(tmp_path / "n.db"))
    store.add("buy milk")
    store.add("call bank about 50% APR")
    assert len(store.search("_")) == 0           # not "every note"
    assert len(store.search("50%")) == 1


# --- SK1: the four skills must honour the LLM tool args ----------------------

def test_timers_arms_from_llm_args():
    from unittest.mock import MagicMock
    from skills.timers import TimerSkill
    skill = TimerSkill(MagicMock(), None)
    r = asyncio.run(skill.execute(SkillRequest(
        text="timers set 300 stretch",
        args={"action": "set", "seconds": 300, "label": "stretch"})))
    assert r.success and r.data.get("seconds") == 300
    assert "Timer set" in r.speech
    for t in list(skill._timers.values()):
        t.task.cancel()


def test_media_uses_the_action_arg():
    from skills.media import MediaSkill
    r = asyncio.run(MediaSkill().execute(SkillRequest(
        text="media playpause", args={"action": "playpause"})))
    # action recognised (data carries it) rather than the "which action?" prompt
    assert r.data.get("action") == "playpause"
    assert "Play, pause" not in r.speech


def test_notes_add_and_delete_from_llm_args(tmp_path):
    from core.memory import NoteStore
    from skills.notes import NotesSkill
    store = NoteStore(str(tmp_path / "n.db"))
    skill = NotesSkill(store)
    added = asyncio.run(skill.execute(SkillRequest(
        text="notes add buy milk", args={"action": "add", "text": "buy milk"})))
    assert added.success and store.list()[0].text == "buy milk"
    nid = store.list()[0].id
    deleted = asyncio.run(skill.execute(SkillRequest(
        text="notes delete", args={"action": "delete", "id": nid})))
    assert deleted.success and store.list() == []


def test_files_searches_from_the_query_arg(tmp_path):
    from core.safety import PathWhitelist
    from skills.files import FilesSkill
    (tmp_path / "resume.pdf").write_bytes(b"x")
    skill = FilesSkill(PathWhitelist([str(tmp_path)]))
    r = asyncio.run(skill.execute(SkillRequest(
        text="files find resume", args={"action": "find", "query": "resume"})))
    # it actually searched (found the file) rather than asking "which file?"
    assert "resume" in r.speech.lower()
    assert r.speech not in ("Which file?", "Која датотека?")


# --- REM2: non-object JSON body -> {} (handlers 400, never 500) ---------------

@pytest.mark.asyncio
async def test_json_dict_coerces_non_objects():
    from unittest.mock import AsyncMock
    from remote.server import _json_dict

    class Req:
        def __init__(self, value): self._v = value
        async def json(self): return self._v

    assert await _json_dict(Req({"a": 1})) == {"a": 1}
    for bad in (5, [1, 2], "x", True, None):
        assert await _json_dict(Req(bad)) == {}

    class BadReq:
        async def json(self): raise ValueError("not json")
    assert await _json_dict(BadReq()) == {}


# --- VIS2: GestureStabilizer fires after a cooldown, not just on the exact frame

def test_gesture_stabilizer_fires_after_cooldown_expires():
    from vision.gestures import GestureStabilizer
    st = GestureStabilizer(stability_frames=2, cooldown_frames=3)
    # gesture A fires, latching a cooldown
    fired = [st.update("victory") for _ in range(2)]
    assert "victory" in fired
    # release, then hold gesture B; its stable frame lands during A's cooldown —
    # with the old `== need` test B would be lost forever
    st.update("unknown")
    out = [st.update("three") for _ in range(8)]
    assert "three" in out
