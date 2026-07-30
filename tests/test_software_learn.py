"""Learning foundation: the app-UI map store, the pure scanner, and the learn /
locate / navigate skills. The OS is faked (a canned walker + a fake mechanisms),
so every rung is pinned without a desktop: a scan becomes a saved map, lookup
ranks the right control, navigate focuses + invokes it, and the trust boundary
refuses instructions that arrive from untrusted content.
"""

from __future__ import annotations

import asyncio

import pytest

from skills.base import SkillRequest
from software.connector_base import ActionResult
from software.ui_scan import RawElement


# -- fakes --------------------------------------------------------------------

class FakeWalker:
    def __init__(self, elements, menus=None):
        self._elements = list(elements)
        self._menus = dict(menus or {})

    def elements(self, window_hint, *, reveal_menus):
        return list(self._elements)

    def menus(self, window_hint):
        return dict(self._menus)


class FakeMech:
    def __init__(self, *, can_focus=True, invoke_ok=True):
        self.can_focus = can_focus
        self.invoke_ok = invoke_ok
        self.focused = []
        self.invoked = []

    def focus_app(self, hint):
        self.focused.append(hint)
        return self.can_focus

    def ui_automation(self, hint, name, action="invoke"):
        self.invoked.append((hint, name))
        return ActionResult(self.invoke_ok, "Done." if self.invoke_ok else "no",
                            verified=self.invoke_ok)


def _run(skill, args=None, ctx=None):
    return asyncio.run(skill.execute(
        SkillRequest(text="", args=args or {}, context=ctx or {})))


def _learn_capcut(tmp_path):
    from software.knowledge import AppMap, UIElement, save_map
    save_map(AppMap(app_id="capcut", display_name="CapCut", elements=[
        UIElement(name="Search effects", role="Edit", clickable=True,
                  path=("Effects",)),
        UIElement(name="Export", role="Button", clickable=True),
    ]), tmp_path)


# -- knowledge store ----------------------------------------------------------

def test_map_save_load_forget_and_list(tmp_path):
    from software.knowledge import (
        AppMap, UIElement, forget_map, list_maps, load_map, save_map,
    )
    m = AppMap(app_id="capcut", display_name="CapCut", scanned_at=123.0,
               elements=[UIElement(name="Effects", role="Button", clickable=True),
                         UIElement(name="Export", role="Button", clickable=True,
                                   path=("Toolbar",))],
               menus={"File": ["New", "Open"]})
    assert save_map(m, tmp_path).exists()
    got = load_map("capcut", tmp_path)
    assert got is not None and got.element_count() == 2
    assert got.elements[1].path == ("Toolbar",)      # tuples survive the round-trip
    lst = list_maps(tmp_path)
    assert lst and lst[0]["app_id"] == "capcut" and lst[0]["count"] == 2
    assert forget_map("capcut", tmp_path) is True
    assert load_map("capcut", tmp_path) is None


def test_search_ranks_by_relevance():
    from software.knowledge import AppMap, UIElement, search
    m = AppMap(app_id="capcut", elements=[
        UIElement(name="Effects", role="Button", clickable=True),
        UIElement(name="Search effects", role="Edit", clickable=True,
                  keywords=("effects panel",)),
        UIElement(name="Export", role="Button", clickable=True),
    ])
    hits = search(m, "effects search")
    assert hits and hits[0][0].name == "Search effects"
    assert search(m, "nothing like this") == []


# -- pure scanner -------------------------------------------------------------

def test_scan_app_dedups_and_folds_in_menus():
    from software.ui_scan import scan_app
    walker = FakeWalker(
        elements=[RawElement("Effects", "Button"),
                  RawElement("Effects", "Button"),     # duplicate -> collapsed
                  RawElement("", "Text"),              # nameless -> skipped
                  RawElement("Timeline", "Pane")],
        menus={"File": ["New", "Export"]})
    m = scan_app(walker, app_id="capcut", display_name="CapCut", now=lambda: 5.0)
    names = [e.name for e in m.elements]
    assert names.count("Effects") == 1                 # deduped
    assert "Timeline" in names and "New" in names and "Export" in names
    export = next(e for e in m.elements if e.name == "Export")
    assert export.source == "menu" and export.path == ("File",) and export.clickable
    assert next(e for e in m.elements if e.name == "Effects").clickable is True
    assert next(e for e in m.elements if e.name == "Timeline").clickable is False
    assert m.scanned_at == 5.0 and m.menus == {"File": ["New", "Export"]}
    assert "vision" in m.notes                         # thin tree -> vision hint


# -- learn skill --------------------------------------------------------------

def test_learn_skill_scans_focuses_and_saves(tmp_path):
    from skills.software_learn import LearnAppSkill
    from software.knowledge import load_map
    mech = FakeMech(can_focus=True)
    walker = FakeWalker([RawElement("Effects", "Button")], {"File": ["Export"]})
    r = _run(LearnAppSkill(mech, walker, base_dir=tmp_path), args={"app": "CapCut"})
    assert r.success and mech.focused == ["CapCut"]
    m = load_map("capcut", tmp_path)
    assert m is not None and any(e.name == "Effects" for e in m.elements)


def test_learn_skill_needs_the_app_open(tmp_path):
    from skills.software_learn import LearnAppSkill
    r = _run(LearnAppSkill(FakeMech(can_focus=False), FakeWalker([]),
                           base_dir=tmp_path), args={"app": "CapCut"})
    assert r.success is False and r.data.get("reason") == "not_open"


def test_learn_skill_refuses_untrusted(tmp_path):
    from skills.software_learn import LearnAppSkill
    r = _run(LearnAppSkill(FakeMech(), FakeWalker([]), base_dir=tmp_path),
             args={"app": "CapCut"}, ctx={"untrusted": True})
    assert r.success is False and r.data.get("refused") == "untrusted"


def test_learn_fast_path_matches_explicit_forms_only():
    from skills.software_learn import LearnAppSkill
    s = LearnAppSkill(FakeMech(), FakeWalker([]))
    for t in ["learn how to use CapCut", "learn the capcut app",
              "scan CapCut", "map davinci resolve", "explore Photoshop's UI"]:
        assert s.match(t) is not None, t
    # a bare topic-learn must NOT be hijacked into a UI scan (LLM decides those)
    assert s.match("learn python") is None
    assert s.match("what is CapCut") is None


# -- locate (read-only) + navigate (actuation) --------------------------------

def test_locate_describes_without_touching_anything(tmp_path):
    from skills.software_learn import AppLocateSkill
    _learn_capcut(tmp_path)
    mech = FakeMech()
    s = AppLocateSkill(mech, base_dir=tmp_path)
    assert s.controls_pc is False
    r = _run(s, args={"app": "CapCut", "target": "search effects"})
    assert r.success and "Search effects" in r.speech
    assert mech.focused == [] and mech.invoked == []     # read-only


def test_locate_needs_a_learned_app(tmp_path):
    from skills.software_learn import AppLocateSkill
    r = _run(AppLocateSkill(FakeMech(), base_dir=tmp_path),
             args={"app": "CapCut", "target": "export"})
    assert r.success is False and r.data.get("reason") == "not_learned"


def test_navigate_focuses_and_invokes_the_control(tmp_path):
    from skills.software_learn import AppNavigateSkill
    _learn_capcut(tmp_path)
    mech = FakeMech(invoke_ok=True)
    s = AppNavigateSkill(mech, base_dir=tmp_path)
    assert s.controls_pc is True
    r = _run(s, args={"app": "CapCut", "target": "export"})
    assert r.success and mech.focused == ["CapCut"]
    assert mech.invoked and mech.invoked[0][1] == "Export"


def test_navigate_no_match_is_reported(tmp_path):
    from skills.software_learn import AppNavigateSkill
    _learn_capcut(tmp_path)
    r = _run(AppNavigateSkill(FakeMech(), base_dir=tmp_path),
             args={"app": "CapCut", "target": "quantum flux capacitor"})
    assert r.success is False and r.data.get("reason") == "no_match"


def test_navigate_refuses_untrusted(tmp_path):
    from skills.software_learn import AppNavigateSkill
    _learn_capcut(tmp_path)
    mech = FakeMech()
    r = _run(AppNavigateSkill(mech, base_dir=tmp_path),
             args={"app": "CapCut", "target": "export"},
             ctx={"provenance": "untrusted"})
    assert r.success is False and r.data.get("refused") == "untrusted"
    assert mech.invoked == []                            # nothing happened


def test_locate_and_navigate_fast_paths():
    from skills.software_learn import AppLocateSkill, AppNavigateSkill
    loc, nav = AppLocateSkill(FakeMech()), AppNavigateSkill(FakeMech())
    assert loc.match("where is the export button in CapCut") is not None
    assert loc.match("find the search box in CapCut") is not None
    assert nav.match("open the effects panel in CapCut") is not None
    assert nav.match("in CapCut, click the export button") is not None
    # locate is read-only; a 'click' request should NOT match it
    assert loc.match("in CapCut, click the export button") is None


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
