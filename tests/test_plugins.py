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
