"""Software connectors: the abstraction, mechanisms, registration, security.

The OS layer is a FakeBackend and the policy engine is real, so every rung of the
ladder + the non-negotiables are pinned without a desktop: actions become gated
skills, the hotkey flow focuses→acts→restores, the CLI/API adapters obey the
policy, send/post always confirms, and an instruction from untrusted content is
refused.
"""

from __future__ import annotations

import asyncio

import pytest

from core.config import load_settings
from security.capabilities import ACTUATION_CAPS, Capability
from security.policy import Actor, PolicyEngine
from skills.base import SkillRequest
from software.connector_base import ActionResult
from software.connectors.browser import BrowserConnector
from software.connectors.media import MediaConnector
from software.connectors.template import TemplateConnector
from software.connectors.window import WindowConnector
from software.mechanisms import Mechanisms
from software.registry import (
    ConnectorActionSkill,
    build_software_skills,
    register_software,
)


class FakeWindow:
    def __init__(self, title):
        self.title = title


class FakeBackend:
    def __init__(self, *, running=(), installed=(), windows=(), active=None,
                 send_ok=True):
        self.running = set(running)
        self.installed = set(installed)
        self.windows = [FakeWindow(t) for t in windows]
        self.active = active
        self.focused = []
        self.keys_sent = []
        self.media = []
        self.launched = []
        self.send_ok = send_ok

    def find_window(self, sub):
        for w in self.windows:
            if sub and sub.lower() in w.title.lower():
                return w
        return None

    def active_window(self):
        return self.active

    def focus_window(self, h):
        self.focused.append(h)
        self.active = h
        return True

    def send_keys(self, keys):
        self.keys_sent.append(keys)
        return self.send_ok

    def send_media_key(self, key):
        self.media.append(key)
        return self.send_ok

    def running_processes(self):
        return list(self.running)

    def is_installed(self, exe):
        return exe.lower() in {e.lower() for e in self.installed}

    def launch(self, cmd):
        self.launched.append(cmd)
        return True

    def close_window(self, h):
        return True


def _policy(pc=True):
    s = load_settings()
    s.safety.pc_control_enabled = pc
    return PolicyEngine(s.security, s.safety, None)


def _mech(backend, *, pc=True, exe_paths=None):
    return Mechanisms(backend, _policy(pc), exe_paths=exe_paths or {},
                      actor=Actor.LOCAL_USER)


def _skill(connector, action_name):
    action = next(a for a in connector.actions() if a.name == action_name)
    return ConnectorActionSkill(connector, action)


def _run(skill, args=None, **ctx):
    return asyncio.run(skill.execute(
        SkillRequest(text="", args=args or {}, context=ctx)))


# -- S1: registration ---------------------------------------------------------

def test_actions_become_skills_with_patterns_and_tools():
    connector = MediaConnector(_mech(FakeBackend()))
    skill = _skill(connector, "play_pause")
    assert skill.name == "media_play_pause"
    assert skill.patterns and skill.match("pause the music") is not None
    schema = skill.tool_schema()
    assert schema["function"]["name"] == "media_play_pause"
    # control_input is actuation -> controls_pc derived True (security invariant)
    assert skill.capabilities == frozenset({Capability.CONTROL_INPUT})
    assert skill.controls_pc is bool(skill.capabilities & ACTUATION_CAPS) is True


def test_registry_respects_the_enabled_flag():
    s = load_settings()
    mech = _mech(FakeBackend())
    s.software.enabled = False
    assert build_software_skills(s, mech) == []
    s.software.enabled = True
    skills = build_software_skills(s, mech)
    names = {sk.name for sk in skills}
    assert "media_play_pause" in names and "window_close_window" in names
    assert "browser_new_tab" in names


def test_per_connector_opt_out():
    s = load_settings()
    s.software.enabled = True
    s.software.connectors = {"browser": False}
    names = {sk.name for sk in build_software_skills(s, _mech(FakeBackend()))}
    assert not any(n.startswith("browser_") for n in names)
    assert any(n.startswith("media_") for n in names)


# -- detect() states ----------------------------------------------------------

def test_detect_installed_and_running():
    b = FakeBackend(installed={"chrome.exe"}, running={"chrome.exe"})
    d = BrowserConnector(_mech(b)).detect()
    assert d.installed and d.running
    b2 = FakeBackend(installed=set(), running=set())
    d2 = BrowserConnector(_mech(b2)).detect()
    assert not d2.installed and not d2.running


# -- S2: the hotkey flow (focus target -> act -> restore) ---------------------

def test_hotkey_focuses_the_target_then_restores_focus():
    prev = FakeWindow("Some Editor")
    b = FakeBackend(running={"chrome.exe"}, installed={"chrome.exe"},
                    windows=["New Tab - Google Chrome"], active=prev)
    skill = _skill(BrowserConnector(_mech(b)), "new_tab")
    r = _run(skill)
    assert r.success
    assert b.keys_sent == ["ctrl+t"]
    # focused the browser, then restored the previous window
    assert [w.title for w in b.focused] == ["New Tab - Google Chrome", "Some Editor"]


def test_hotkey_reports_honestly_when_window_missing():
    b = FakeBackend(running={"chrome.exe"}, installed={"chrome.exe"}, windows=())
    skill = _skill(BrowserConnector(_mech(b)), "new_tab")
    r = _run(skill)
    assert r.success is False and "window" in r.speech.lower()


def test_media_key_is_sent_but_unverified():
    b = FakeBackend()
    r = _run(_skill(MediaConnector(_mech(b)), "next_track"))
    assert r.success and b.media == ["next"] and r.data["verified"] is False


def test_minimize_sends_a_global_hotkey_no_window_needed():
    # Regression: minimize named no window, so routing it through find-a-window
    # returned "couldn't find that app's window" and never sent win+down.
    b = FakeBackend(windows=())                 # no windows at all
    r = _run(_skill(WindowConnector(_mech(b)), "minimize"))
    assert r.success and b.keys_sent == ["win+down"]


# -- S2: CLI + API adapters go through the policy engine ----------------------

def test_cli_adapter_refuses_without_the_capability():
    m = _mech(FakeBackend())
    r = m.run_cli("x", ["tool", "--go"], declared=frozenset())
    assert r.success is False                         # undeclared -> denied


def test_cli_adapter_runs_with_the_capability():
    b = FakeBackend()
    m = _mech(b)
    r = m.run_cli("x", ["tool", "--go"], declared=frozenset({Capability.RUN_COMMAND}))
    assert r.success and b.launched == [["tool", "--go"]]


def test_cli_adapter_blocked_when_pc_control_off():
    b = FakeBackend()
    m = _mech(b, pc=False)
    r = m.run_cli("x", ["tool"], declared=frozenset({Capability.RUN_COMMAND}))
    assert r.success is False and b.launched == []     # actuation gate


def test_local_api_refuses_non_localhost():
    m = _mech(FakeBackend())
    r = m.call_local_api("x", "GET", "http://evil.example/api",
                         declared=frozenset({Capability.NETWORK}))
    assert r.success is False


def test_local_api_refuses_without_network_capability():
    m = _mech(FakeBackend())
    r = m.call_local_api("x", "GET", "http://127.0.0.1:9/x", declared=frozenset())
    assert r.success is False                          # undeclared network


# -- S3/S4: graceful states, trust boundary, confirm-before-send --------------

def test_not_installed_is_reported_cleanly():
    b = FakeBackend(installed=set())                   # myapp not installed
    skill = _skill(TemplateConnector(_mech(b)), "do_thing")
    r = _run(skill)
    assert r.success is False and r.data["reason"] == "not_installed"


def test_not_running_launches_the_app():
    b = FakeBackend(installed={"myapp.exe"}, running=set())
    m = _mech(b, exe_paths={"myapp": ["myapp.exe"]})
    skill = _skill(TemplateConnector(m), "do_thing")
    _run(skill)
    assert b.launched == [["myapp.exe"]]               # opened it because not running


def test_untrusted_instruction_is_refused():
    b = FakeBackend()
    skill = _skill(MediaConnector(_mech(b)), "play_pause")
    r = _run(skill, untrusted=True)
    assert r.success is False and r.data["refused"] == "untrusted"
    assert b.media == []                               # nothing happened
    # same via the provenance marker
    assert _run(skill, provenance="untrusted").data["refused"] == "untrusted"


def test_close_window_forces_confirmation():
    b = FakeBackend(windows=["Notepad"])
    skill = _skill(WindowConnector(_mech(b)), "close_window")
    r = _run(skill, args={"app": "Notepad"})
    assert r.needs_confirmation is True and b.keys_sent == []   # not yet
    r2 = _run(skill, args={"app": "Notepad"}, confirmed=True)
    assert r2.success and b.keys_sent == ["alt+F4"]


def test_send_action_always_confirms_first():
    b = FakeBackend(installed={"myapp.exe"}, running={"myapp.exe"})
    skill = _skill(TemplateConnector(_mech(b)), "send_message")
    r = _run(skill, text="hi")
    assert r.needs_confirmation is True                # never auto-sends


def test_template_connector_loads():
    conn = TemplateConnector(_mech(FakeBackend()))
    names = {a.name for a in conn.actions()}
    assert {"do_thing", "run_task", "send_message"} <= names
    # every send/post-style action requires confirmation
    for a in conn.actions():
        if a.name in ("send_message", "run_task"):
            assert a.requires_confirmation is True


def test_register_software_adds_to_a_registry():
    from skills.base import SkillRegistry
    s = load_settings()
    s.software.enabled = True
    reg = SkillRegistry()
    n = register_software(reg, s, _mech(FakeBackend()))
    assert n > 0 and reg.find_match("pause the music") is not None


# -- S5: the HUD SOFTWARE CONTROL panel (endpoints + persistence) -------------

def _serve(check):
    """Run ``check(client, settings)`` against a live RemoteServer (no persist)."""
    from aiohttp.test_utils import TestClient, TestServer
    from core.events import EventBus, StateMachine
    from remote.server import RemoteServer

    async def run():
        settings = load_settings()
        server = RemoteServer(settings, router=None, sm=StateMachine(EventBus()))
        client = TestClient(TestServer(server.build_app()))
        await client.start_server()
        try:
            return await check(client, settings)
        finally:
            await client.close()
    return asyncio.run(run())


def test_software_status_lists_connectors_and_actions():
    async def check(client, settings):
        d = await (await client.get("/software")).json()
        assert d["ok"] and d["enabled"] is False               # off by default
        ids = {c["app_id"] for c in d["connectors"]}
        assert {"media", "window", "browser"} <= ids           # all shipped ones
        media = next(c for c in d["connectors"] if c["app_id"] == "media")
        assert media["enabled"] is True and "play pause" in media["actions"]
    _serve(check)


def test_software_control_toggles_master_and_per_connector():
    async def check(client, settings):
        r = await (await client.post("/control/software", json={"on": True})).json()
        assert r["ok"] and r["enabled"] is True and r["restart"] is True
        assert settings.software.enabled is True               # applied live
        r2 = await (await client.post(
            "/control/software", json={"app_id": "browser", "app_on": False})).json()
        assert r2["connectors"]["browser"] is False
        assert settings.software.connectors["browser"] is False
        # a per-connector toggle without app_on, and an empty body, are 400s
        assert (await client.post("/control/software",
                                  json={"app_id": "media"})).status == 400
        assert (await client.post("/control/software", json={})).status == 400
    _serve(check)


def test_software_choice_persists_and_reloads(tmp_path):
    from core.config import apply_local_secrets, save_software
    p = tmp_path / "overrides.yaml"
    save_software(enabled=True, connectors={"browser": False}, path=p)
    settings = apply_local_secrets(load_settings(), p)
    assert settings.software.enabled is True
    assert settings.software.connectors.get("browser") is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
