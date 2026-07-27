"""Core/remote robustness: a bad input must degrade, not take a subsystem down.

Bug-audit findings:
* A routine loaded from config.yaml skips the HUD's HH:MM validation, so a typo
  like ``at: "8am"`` made ``seconds_until`` raise and — unguarded — killed the
  whole scheduler task, silently stopping ALL routines.
* ``/pair/confirm`` fed a non-ASCII code straight into ``hmac.compare_digest``,
  which raises ``TypeError`` on non-ASCII str — a 500 instead of a clean 401.
"""

from __future__ import annotations

import time

import pytest

from core.config import RoutineItem
from core.routines import RoutineScheduler


class _Stop(Exception):
    """Sentinel to break the scheduler's infinite loop after the first fire."""


@pytest.mark.asyncio
async def test_bad_routine_time_does_not_kill_the_scheduler(monkeypatch):
    good = RoutineItem(name="good", at="08:00", ask=["the weather"])
    bad = RoutineItem(name="bad", at="8am", ask=["the news"])   # invalid HH:MM
    sched = RoutineScheduler([good, bad], router=None, announcer=None)

    fired: list[str] = []

    async def fake_fire(routine):
        fired.append(routine.name)
        raise _Stop

    async def fast_sleep(_seconds):
        return None

    monkeypatch.setattr(sched, "fire", fake_fire)
    monkeypatch.setattr("core.routines.asyncio.sleep", fast_sleep)

    with pytest.raises(_Stop):
        await sched.run_forever()
    # The bad routine was skipped, not fatal; the good one still scheduled.
    assert fired == ["good"]


@pytest.mark.asyncio
async def test_pair_confirm_rejects_non_ascii_code_with_401():
    from aiohttp.test_utils import TestClient, TestServer

    from core.config import load_settings
    from core.events import EventBus, StateMachine
    from core.router import Router
    from llm.client import OllamaClient
    from remote.server import RemoteServer
    from skills.base import SkillRegistry

    settings = load_settings()
    router = Router(settings, SkillRegistry(), OllamaClient(settings.llm), EventBus())
    router.model = None
    server = RemoteServer(settings, router, StateMachine(EventBus()))
    server._pair = {"code": "123456", "expires": time.monotonic() + 60, "attempts": 0}
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    try:
        resp = await client.post("/pair/confirm", json={"code": "12345é"})
        assert resp.status == 401                 # a clean wrong-code, not a 500
    finally:
        await client.close()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
