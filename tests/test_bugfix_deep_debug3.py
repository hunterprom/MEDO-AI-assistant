"""Regression tests for bugs found in deep-debug loop iteration 3 (2026-07-24)."""

from __future__ import annotations

import asyncio


from skills.base import SkillRequest, SkillResult


# --- MAIN3: routine scheduler must not die when two routines share a fire time -

def test_routine_scheduler_survives_a_tie():
    from unittest.mock import MagicMock
    from core.config import RoutineItem
    from core.routines import RoutineScheduler
    # two enabled routines with the SAME time -> the min() used to compare
    # RoutineItem objects and raise TypeError, killing the scheduler task.
    items = [RoutineItem(name="weather", at="08:00", ask=["weather"], enabled=True),
             RoutineItem(name="news", at="08:00", ask=["news"], enabled=True)]
    sched = RoutineScheduler(items, MagicMock(), MagicMock())
    # one scheduling pass must not raise; run_forever loops, so drive the min()
    # the same way it does, via a tiny timeout.
    async def one_pass():
        task = asyncio.ensure_future(sched.run_forever())
        await asyncio.sleep(0.05)   # let it compute the first min() and sleep
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    asyncio.run(one_pass())   # no TypeError == pass


# --- LINK: manifest validation hardening -------------------------------------

def test_link_rejects_redos_and_bad_patterns():
    from link.registry import validate_manifest

    def manifest(patterns):
        return {"device_id": "dev1", "name": "Dev", "transport": "http_poll",
                "capabilities": [{"name": "go", "description": "do it",
                                  "fast_patterns": patterns}]}

    assert any("nested quantifier" in e for e in
               validate_manifest(manifest(["(a+)+$"])))          # ReDoS
    assert any("must be strings" in e for e in
               validate_manifest(manifest([123])))               # non-string (was TypeError->500)
    assert any("too long" in e for e in
               validate_manifest(manifest(["a" * 200])))         # unbounded length
    assert any("at most 8" in e for e in
               validate_manifest(manifest(["a"] * 9)))           # count cap
    assert validate_manifest(manifest(["^go$", "do it"])) == []  # good ones pass


def test_link_rejects_duplicate_capability_names():
    from link.registry import validate_manifest
    errs = validate_manifest({
        "device_id": "dev1", "name": "Dev", "transport": "http_poll",
        "capabilities": [{"name": "on", "description": "x"},
                         {"name": "on", "description": "y"}]})
    assert any("duplicate" in e for e in errs)


def test_link_reinstall_matches_owning_device_not_prefix():
    from skills.base import SkillRegistry
    from link.registry import LinkRegistry
    reg = SkillRegistry()
    link = LinkRegistry(reg, ":memory:")
    link._install({"device_id": "robo", "name": "Robo", "transport": "http_poll",
                   "capabilities": [{"name": "dog_sit", "description": "sit"}]})
    assert reg.get("robo_dog_sit") is not None
    # a second device whose id is a PREFIX must not clobber robo's skill
    link._install({"device_id": "robo_dog", "name": "RoboDog", "transport": "http_poll",
                   "capabilities": [{"name": "bark", "description": "bark"}]})
    assert reg.get("robo_dog_sit") is not None      # still there (was clobbered)
    assert reg.get("robo_dog_bark") is not None


# --- SK-BROWSER1: a blocked host must not leak to the system browser ----------

def test_browser_opener_does_not_fall_back_on_a_blocked_host(monkeypatch):
    import skills.browser as bmod
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)

    class FakeSession:
        async def goto(self, url):
            raise bmod.BrowserBlocked("bank is blocked")
    opener = bmod.browser_opener(FakeSession())
    ok = asyncio.run(opener("https://bank.com"))
    assert ok is False and opened == []      # NOT opened in the system browser

    class DeadSession:
        async def goto(self, url):
            raise RuntimeError("playwright missing")
    opener2 = bmod.browser_opener(DeadSession())
    asyncio.run(opener2("https://example.com"))
    assert opened == ["https://example.com"]  # a non-block failure DOES fall back


# --- SK-BENCH1 / SK-SCREEN1 ---------------------------------------------------

def test_bench_inventory_reachable_from_llm_query_arg():
    from core.config import load_settings
    from skills.bench import BenchSkill

    class Inv:
        def search(self, q): return [f"3x {q}"]
    skill = BenchSkill(load_settings(), inventory=Inv())
    r = asyncio.run(skill.execute(SkillRequest(
        text="see bench 10k resistors", args={"query": "10k resistors"})))
    assert "inventory" in r.speech.lower() and "10k resistors" in r.speech


def test_screen_agent_handles_a_non_object_model_reply():
    from core.config import load_settings
    from skills.screen_agent import ScreenAgentSkill

    async def ask(image, task, history):
        return '[{"action": "done"}]'      # a JSON ARRAY (common LLM habit)
    skill = ScreenAgentSkill(load_settings(),
                             capture=lambda: (b"img", (100, 100)),
                             ask=ask)
    r = asyncio.run(skill.execute(SkillRequest(
        text="do this for me: click ok", args={}, context={"confirmed": True})))
    assert isinstance(r, SkillResult) and not r.success   # graceful, no crash
