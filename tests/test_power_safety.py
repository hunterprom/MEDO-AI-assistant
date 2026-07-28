"""Power and media must never act on an unnamed request.

Written after PowerSkill suspended a live machine. ``_action`` fell through to
``"sleep"`` for any text it didn't recognise, and "sleep" was not in the set it
asked about — so a call carrying no words at all (an LLM tool call with empty
arguments, or a harness invoking ``execute()`` directly) put the computer to
sleep with no confirmation and no way to intervene.

Every test here asserts that NOTHING ran: the subprocess spawn is patched and
the assertion is on the empty call list, not on the wording of the reply.
"""

from __future__ import annotations

import pytest

from skills.base import SkillRequest
from skills.media import MediaSkill
from skills.system import PowerSkill


@pytest.fixture
def power(monkeypatch):
    """PowerSkill whose OS calls are recorded instead of performed."""
    skill = PowerSkill()
    skill.ran = []
    monkeypatch.setattr("skills.system.subprocess.Popen",
                        lambda cmd, *a, **k: skill.ran.append(cmd))
    return skill


async def _run(skill, text="", args=None, confirmed=False):
    return await skill.execute(SkillRequest(text=text, args=args or {},
                                            context={"confirmed": confirmed}))


# --- the bug ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "   ", "hello there", "what time is it"])
async def test_power_never_acts_on_words_it_does_not_recognise(power, text):
    result = await _run(power, text)
    assert power.ran == [], f"{text!r} triggered {power.ran}"
    assert result.success is False


@pytest.mark.asyncio
async def test_sleep_asks_before_suspending(power):
    """The specific regression: sleep used to be silent and immediate."""
    result = await _run(power, "go to sleep")
    assert power.ran == []
    assert result.needs_confirmation is True


@pytest.mark.asyncio
async def test_sleep_runs_once_confirmed(power):
    result = await _run(power, "go to sleep", confirmed=True)
    assert len(power.ran) == 1 and result.success


# --- the rest of the surface --------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["shut down", "restart the computer",
                                  "reboot", "go to sleep"])
async def test_every_interrupting_action_asks_first(power, text):
    result = await _run(power, text)
    assert power.ran == [], f"{text!r} acted before confirmation"
    assert result.needs_confirmation is True


@pytest.mark.asyncio
async def test_locking_does_not_ask(power):
    """Locking the screen loses nothing, so it is the one that just runs."""
    result = await _run(power, "lock the screen")
    assert len(power.ran) == 1 and result.success
    assert result.needs_confirmation is False


@pytest.mark.asyncio
async def test_an_invented_tool_action_is_refused(power):
    """The LLM path passes `action` straight through; it must be validated."""
    result = await _run(power, "", args={"action": "self_destruct"})
    assert power.ran == [] and result.success is False


@pytest.mark.asyncio
async def test_a_valid_tool_action_still_asks(power):
    result = await _run(power, "", args={"action": "shutdown"})
    assert power.ran == [] and result.needs_confirmation is True


# --- media: same shape of bug, smaller blast radius ---------------------------


@pytest.mark.asyncio
async def test_media_does_not_toggle_playback_from_nothing(monkeypatch):
    skill = MediaSkill()
    ran = []
    monkeypatch.setattr(MediaSkill, "_run", lambda self, action: ran.append(action) or "x")
    result = await skill.execute(SkillRequest(text="", args={}, context={}))
    assert ran == [] and result.success is False


@pytest.mark.asyncio
@pytest.mark.parametrize("text,expected", [
    ("play the music", "playpause"),
    ("pause my song", "playpause"),
    ("next track", "next"),
    ("previous song", "previous"),
    ("пушти музика", "playpause"),
])
async def test_media_still_acts_on_a_real_request(monkeypatch, text, expected):
    skill = MediaSkill()
    ran = []
    monkeypatch.setattr(MediaSkill, "_run", lambda self, action: ran.append(action) or "x")
    await skill.execute(SkillRequest(text=text, args={}, context={}))
    assert ran == [expected]
