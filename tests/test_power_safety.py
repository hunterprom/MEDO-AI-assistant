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


# -- honesty: never announce a power action the OS refused -------------------

def test_power_reports_a_refused_command_instead_of_claiming_it(monkeypatch):
    """`shutdown /s` without the privilege exits 1 immediately ("Access is
    denied"). Popen doesn't raise, so MEDO used to say "Shutting down now." and
    the machine stayed on — right after the user confirmed and walked away."""
    import io

    import skills.system as sysmod

    class _Refused:
        returncode = 1
        stderr = io.BytesIO(b"Access is denied.(5)")
        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(sysmod.subprocess, "Popen", lambda *a, **k: _Refused())
    speech = sysmod.PowerSkill()._run("shutdown")
    assert "couldn't" in speech.lower() and "shutting down now" not in speech.lower()


def test_power_still_announces_when_the_command_is_accepted(monkeypatch):
    import skills.system as sysmod

    class _Accepted:
        returncode = None
        stderr = None
        def wait(self, timeout=None):
            raise sysmod.subprocess.TimeoutExpired("cmd", timeout)

    monkeypatch.setattr(sysmod.subprocess, "Popen", lambda *a, **k: _Accepted())
    assert "shutting down" in sysmod.PowerSkill()._run("shutdown").lower()


def test_absolute_volume_never_guesses_a_direction(monkeypatch):
    """Without an audio endpoint the current level is unknowable, so the old
    `level >= (_current_volume() or 0)` made EVERY request turn the volume UP —
    including "set volume to 10"."""
    import skills.system as sysmod

    nudges = []
    monkeypatch.setattr(sysmod, "IS_MACOS", False)
    monkeypatch.setattr(sysmod, "IS_WINDOWS", True)
    monkeypatch.setattr(sysmod, "_win_set_volume", lambda level: False)
    monkeypatch.setattr(sysmod, "_nudge_media_key", lambda up: nudges.append(up))
    speech = sysmod.VolumeSkill()._set_absolute(10)
    assert nudges == []                       # nothing nudged the wrong way
    assert "exact" in speech.lower() or "can't" in speech.lower()
