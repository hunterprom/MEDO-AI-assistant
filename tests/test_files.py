"""FilesSkill: 'open a file in <app>' opens the APP, not a file named '<app>'.

The live bug: "open file in Arduino IDE" searched for a file literally called
"in Arduino IDE" and reported it couldn't find one. These pin the fix — the
"in|with|using <app>" tail routes to opening that app (plus the named file, if
one was given and found) — without regressing an ordinary file search.
"""

from __future__ import annotations

import asyncio

import pytest

import skills.appfinder as appfinder
import skills.files as files
from skills.base import SkillRequest
from skills.files import _OPEN_WITH, FilesSkill


class FakeWL:
    """A whitelist with no roots (app-open needs none) unless given some."""

    def __init__(self, roots=()):
        self.roots = tuple(roots)

    def describe(self):
        return "Documents"

    def is_allowed(self, _p):
        return True


class FakeApp:
    def __init__(self, name):
        self.name = name
        self.path = f"C:/Start/{name}.lnk"


def _run(skill, text, **args):
    return asyncio.run(skill.execute(
        SkillRequest(text=text, match=skill.match(text), args=args)))


# -- parsing ------------------------------------------------------------------

def test_open_with_clause_splits_file_and_app():
    m = _OPEN_WITH.match("in Arduino IDE")
    assert m and m.group("app") == "Arduino IDE" and m.group("file").strip() == ""
    m2 = _OPEN_WITH.match("main.cpp in Arduino IDE")
    assert m2 and m2.group("file") == "main.cpp" and m2.group("app") == "Arduino IDE"
    assert _OPEN_WITH.match("report.pdf") is None       # no clause -> plain search


def test_patterns_tolerate_me_and_reach_the_skill():
    s = FilesSkill(FakeWL())
    for t in ["open file in Arduino IDE", "find me file in Arduino IDE",
              "open me the file report"]:
        assert s.match(t) is not None, t


# -- the fix: open the app, not a file named after it -------------------------

def test_open_file_in_app_opens_the_app(monkeypatch):
    opened = []
    monkeypatch.setattr(appfinder, "find_app",
                        lambda name: FakeApp("Arduino IDE")
                        if "arduino" in name.lower() else None)
    monkeypatch.setattr(files, "open_path", lambda p: opened.append(str(p)))
    r = _run(FilesSkill(FakeWL()), "open file in Arduino IDE")
    assert r.success and "Arduino IDE" in r.speech
    assert opened == ["C:/Start/Arduino IDE.lnk"]        # opened the app, no search


def test_named_file_in_app_launches_it_with_the_app(monkeypatch, tmp_path):
    (tmp_path / "sketch.ino").write_text("void setup(){}", encoding="utf-8")
    from core.safety import PathWhitelist
    wl = PathWhitelist([str(tmp_path)])
    launched = []
    monkeypatch.setattr(appfinder, "find_app", lambda name: FakeApp("Arduino IDE"))
    monkeypatch.setattr(files, "_launch_with",
                        lambda app, path: launched.append((app, str(path))) or True)
    r = _run(FilesSkill(wl), "open the file sketch.ino in Arduino IDE")
    assert r.success and "sketch.ino" in r.speech and "Arduino IDE" in r.speech
    assert launched and launched[0][0].endswith("Arduino IDE.lnk")
    assert launched[0][1].endswith("sketch.ino")


def test_me_tolerance_still_runs_a_plain_search(tmp_path):
    (tmp_path / "budget.xlsx").write_text("x", encoding="utf-8")
    from core.safety import PathWhitelist
    r = _run(FilesSkill(PathWhitelist([str(tmp_path)])), "find me the file budget")
    assert "budget.xlsx" in r.speech        # 'me' tolerance + ordinary search


def test_unknown_app_in_clause_opens_nothing(monkeypatch, tmp_path):
    # "in vim" doesn't resolve to an installed app -> fall through to a search,
    # never a wrong app-open.
    from core.safety import PathWhitelist
    opened = []
    monkeypatch.setattr(appfinder, "find_app", lambda name: None)
    monkeypatch.setattr(files, "open_path", lambda p: opened.append(str(p)))
    r = _run(FilesSkill(PathWhitelist([str(tmp_path)])),
             "open the file notes in vim")
    assert opened == [] and r.success is False   # searched, opened nothing


def test_app_open_works_without_a_whitelist(monkeypatch):
    # Opening an app needs no whitelisted folders (the old code errored here).
    monkeypatch.setattr(appfinder, "find_app", lambda name: FakeApp("Arduino IDE"))
    monkeypatch.setattr(files, "open_path", lambda p: None)
    r = _run(FilesSkill(FakeWL(roots=())), "open file in Arduino IDE")
    assert r.success and "whitelist" not in r.speech.lower()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
