"""Plugin SDK: discovery, setup() factories, broken-plugin isolation."""

from __future__ import annotations

import pytest

from core.plugins import load_plugins
from skills.base import SkillRegistry, SkillRequest


PLAIN = '''
import re
from skills.base import Skill, SkillRequest, SkillResult

class PingSkill(Skill):
    name = "plugin_ping"
    description = "test ping"
    patterns = [re.compile("plugin ping", re.IGNORECASE)]
    async def execute(self, request):
        return SkillResult("pong from plugin")
'''

FACTORY = '''
import re
from skills.base import Skill, SkillRequest, SkillResult

class EchoSkill(Skill):
    name = "plugin_echo"
    description = "test echo"
    patterns = [re.compile("plugin echo", re.IGNORECASE)]
    def __init__(self, tag):
        self._tag = tag
    async def execute(self, request):
        return SkillResult(f"echo {self._tag}")

def setup(services):
    return [EchoSkill(services["settings"] or "none")]
'''


@pytest.mark.asyncio
async def test_discovers_plain_and_factory_plugins(tmp_path):
    (tmp_path / "aping.py").write_text(PLAIN, encoding="utf-8")
    (tmp_path / "becho.py").write_text(FACTORY, encoding="utf-8")
    (tmp_path / "_ignored.py").write_text("raise RuntimeError('never imported')", encoding="utf-8")

    registry = SkillRegistry()
    loaded = load_plugins(registry, {"settings": "S"}, plugins_dir=tmp_path)
    assert loaded == ["aping:plugin_ping", "becho:plugin_echo"]

    skill, match = registry.find_match("plugin ping please")
    result = await skill.execute(SkillRequest(text="plugin ping please", match=match))
    assert result.speech == "pong from plugin"


def test_broken_plugin_is_skipped_not_fatal(tmp_path):
    (tmp_path / "broken.py").write_text("this is not valid python(", encoding="utf-8")
    (tmp_path / "good.py").write_text(PLAIN, encoding="utf-8")
    registry = SkillRegistry()
    loaded = load_plugins(registry, {}, plugins_dir=tmp_path)
    assert loaded == ["good:plugin_ping"]


def test_missing_dir_is_fine(tmp_path):
    assert load_plugins(SkillRegistry(), {}, plugins_dir=tmp_path / "nope") == []


# --- the Obsidian launcher ----------------------------------------------------
#
# It used to go straight to obsidian://, which needs a registered protocol
# handler. On a machine without one, Windows shows "Get an app to open this
# 'obsidian' link" while Popen returns successfully — so MEDO announced
# "Opening Obsidian" over a visible error dialog.


def _obsidian_open():
    import importlib.util

    spec = importlib.util.spec_from_file_location("obsidian_plugin",
                                                  "plugins/obsidian.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.ObsidianOpenSkill()


@pytest.mark.asyncio
async def test_obsidian_launches_the_installed_app_not_the_uri(monkeypatch):
    from pathlib import Path

    from skills.appfinder import FoundApp
    from skills.base import SkillRequest

    module, skill = _obsidian_open()
    opened, spawned = [], []
    monkeypatch.setattr("skills.appfinder.find_app",
                        lambda name, apps=None: FoundApp(
                            "Obsidian", Path("C:/Programs/Obsidian.exe"), "Start Menu"))
    monkeypatch.setattr("core.platform.open_path", lambda p: opened.append(p))
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: spawned.append(a))

    result = await skill.execute(SkillRequest(text="open obsidian",
                                              match=skill.match("open obsidian")))
    assert result.success and opened == [Path("C:/Programs/Obsidian.exe")]
    assert spawned == [], "the URI must not be used when the app was found"


@pytest.mark.asyncio
async def test_obsidian_falls_back_to_the_uri_when_undiscoverable(monkeypatch):
    from skills.base import SkillRequest

    module, skill = _obsidian_open()
    spawned = []
    monkeypatch.setattr("skills.appfinder.find_app", lambda name, apps=None: None)
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: spawned.append(a))

    result = await skill.execute(SkillRequest(text="open obsidian",
                                              match=skill.match("open obsidian")))
    assert result.success and len(spawned) == 1
    assert result.data.get("via") == "uri"


def test_obsidian_launcher_is_gated_by_the_pc_switch():
    _module, skill = _obsidian_open()
    assert skill.controls_pc is True
