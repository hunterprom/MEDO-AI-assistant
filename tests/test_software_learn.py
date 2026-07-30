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
    def __init__(self, *, can_focus=True, invoke_ok=True, rect=(100, 100, 900, 700),
                 title=""):
        self.can_focus = can_focus
        self.invoke_ok = invoke_ok
        self.rect = rect
        self.title = title
        self.focused = []
        self.invoked = []
        self.clicked = []

    def foreground_title(self):
        return self.title

    def focus_app(self, hint):
        self.focused.append(hint)
        return self.can_focus

    def ui_automation(self, hint, name, action="invoke"):
        self.invoked.append((hint, name))
        return ActionResult(self.invoke_ok, "Done." if self.invoke_ok else "no",
                            verified=self.invoke_ok)

    def foreground_rect(self):
        return self.rect

    def click_point(self, x, y):
        self.clicked.append((int(x), int(y)))
        return ActionResult(True, "Done.", verified=False)


def _run(skill, args=None, ctx=None):
    return asyncio.run(skill.execute(
        SkillRequest(text="", args=args or {}, context=ctx or {})))


def _run_text(skill, text):
    """Route text through match() (so groupdict targets are parsed) then execute."""
    return asyncio.run(skill.execute(
        SkillRequest(text=text, match=skill.match(text))))


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


# -- vision pass: parse, augment, translate, and use it ----------------------

def test_parse_vision_elements_from_json_and_prose():
    from software.vision_probe import parse_vision_elements
    raw = ('[{"label":"Export","area":"top","x":900,"y":50},'
           '{"label":"Effects","area":"left","x":100,"y":400}]')
    els = parse_vision_elements(raw)
    assert [e.name for e in els] == ["Export", "Effects"]
    assert els[0].source == "vision" and els[0].vision_xy == (900, 50)
    assert els[0].keywords == ("top",)
    # embedded in prose, a dupe, and junk items -> one clean element
    messy = 'Sure! [{"label":"Export"},{"label":"export"},{"nope":1},"x"] done'
    assert [e.name for e in parse_vision_elements(messy)] == ["Export"]
    assert parse_vision_elements("no json here") == []
    assert parse_vision_elements("") == []


def test_vision_augment_and_is_thin():
    from software.knowledge import AppMap, UIElement
    from software.ui_scan import is_thin, vision_augment
    m = AppMap(app_id="x", elements=[
        UIElement(name="Timeline", role="Pane", source="uia")])
    assert is_thin(m) is True
    m2 = vision_augment(m, [UIElement(name="Export", source="vision"),
                            UIElement(name="timeline", source="vision")])  # dupe
    assert [e.name for e in m2.elements] == ["Timeline", "Export"]
    assert "vision added 1" in m2.notes
    rich = AppMap(app_id="y", elements=[
        UIElement(name=f"B{i}", role="Button", source="uia") for i in range(8)])
    assert is_thin(rich) is False


def test_window_point_translates_norm_to_live_pixels():
    from software.ui_scan import window_point
    assert window_point((500, 500), (100, 100, 900, 700)) == (500, 400)
    assert window_point((0, 0), (100, 100, 900, 700)) == (100, 100)
    assert window_point((1000, 1000), (100, 100, 900, 700)) == (900, 700)


def test_vision_xy_round_trips_through_disk(tmp_path):
    from software.knowledge import AppMap, UIElement, load_map, save_map
    save_map(AppMap(app_id="x", elements=[
        UIElement(name="A", source="vision", vision_xy=(12, 34))]), tmp_path)
    assert load_map("x", tmp_path).elements[0].vision_xy == (12, 34)


def test_learn_uses_vision_only_when_uia_is_thin(tmp_path):
    from skills.software_learn import LearnAppSkill
    from software.knowledge import UIElement, load_map

    async def fake_vision(app):
        return [UIElement(name="Search effects", source="vision", clickable=True,
                          keywords=("center",), vision_xy=(500, 500))]

    # thin UIA (1 control) -> vision runs and augments the map
    r = _run(LearnAppSkill(FakeMech(), FakeWalker([RawElement("Timeline", "Pane")]),
                           base_dir=tmp_path, vision=fake_vision),
             args={"app": "CapCut"})
    assert r.success and r.data.get("vision") == 1
    m = load_map("capcut", tmp_path)
    assert any(e.source == "vision" and e.name == "Search effects"
               for e in m.elements)

    # rich UIA (>= threshold) -> vision is NOT called
    called = []

    async def spy_vision(app):
        called.append(app)
        return []

    rich = [RawElement(f"Button {i}", "Button") for i in range(10)]
    r2 = _run(LearnAppSkill(FakeMech(), FakeWalker(rich), base_dir=tmp_path,
                            vision=spy_vision), args={"app": "Notepad"})
    assert r2.success and called == []


def test_navigate_clicks_a_vision_control_when_uia_cannot(tmp_path):
    from skills.software_learn import AppNavigateSkill
    from software.knowledge import AppMap, UIElement, save_map
    save_map(AppMap(app_id="capcut", display_name="CapCut", elements=[
        UIElement(name="Search effects", source="vision", clickable=True,
                  keywords=("center",), vision_xy=(500, 500))]), tmp_path)
    mech = FakeMech(invoke_ok=False, rect=(100, 100, 900, 700))   # UIA fails
    r = _run(AppNavigateSkill(mech, base_dir=tmp_path),
             args={"app": "CapCut", "target": "search effects"})
    assert r.success and r.data.get("via") == "vision"
    assert mech.clicked == [(500, 400)]          # re-resolved against the window
    assert mech.focused == ["CapCut"]


def test_navigate_prefers_uia_over_the_vision_click(tmp_path):
    from skills.software_learn import AppNavigateSkill
    from software.knowledge import AppMap, UIElement, save_map
    save_map(AppMap(app_id="capcut", display_name="CapCut", elements=[
        UIElement(name="Export", source="vision", clickable=True,
                  vision_xy=(500, 500))]), tmp_path)
    mech = FakeMech(invoke_ok=True)              # UIA succeeds
    r = _run(AppNavigateSkill(mech, base_dir=tmp_path),
             args={"app": "CapCut", "target": "export"})
    assert r.success and mech.clicked == []      # no coordinate click needed


def test_vision_probe_captures_asks_and_parses():
    from software.vision_probe import VisionProbe

    async def fake_describe(image):
        return '[{"label":"Export","x":900,"y":50}]'

    probe = VisionProbe(settings=None, capture=lambda hint: "IMG",
                        describe=fake_describe)
    els = asyncio.run(probe("CapCut"))
    assert [e.name for e in els] == ["Export"] and els[0].vision_xy == (900, 50)


# -- app-context tracking: resolve the app in focus / just named --------------

def test_foreground_app_matches_longest_learned_name():
    from skills.software_learn import _foreground_app
    learned = [{"display_name": "IDE", "app_id": "ide"},
               {"display_name": "Arduino IDE", "app_id": "arduino ide"}]
    assert _foreground_app("sketch | Arduino IDE 2.3.2", learned) == "Arduino IDE"
    assert _foreground_app("Notepad", learned) == ""       # nothing learned matches


def test_session_context_remembers_last_app():
    from core.app_context import SessionAppContext
    ctx = SessionAppContext()
    ctx.note("CapCut")
    assert ctx.last_app == "CapCut"
    ctx.note("")                              # blank doesn't clobber
    assert ctx.last_app == "CapCut"


def test_context_less_pattern_needs_a_current_app():
    from core.app_context import SessionAppContext
    from skills.software_learn import AppNavigateSkill
    # no context -> the context-less pattern declines (falls to the LLM)
    assert AppNavigateSkill(FakeMech(), ctx=SessionAppContext()).match(
        "open the tools menu") is None
    # once an app is remembered -> it claims the fast path
    assert AppNavigateSkill(FakeMech(), ctx=SessionAppContext(last_app="CapCut")
                            ).match("open the tools menu") is not None
    # an explicit 'in <app>' always matches regardless of context
    assert AppNavigateSkill(FakeMech()).match(
        "open the export panel in CapCut") is not None


def test_navigate_resolves_the_focused_app(tmp_path):
    from core.app_context import SessionAppContext
    from skills.software_learn import AppNavigateSkill
    from software.knowledge import AppMap, UIElement, save_map
    save_map(AppMap(app_id="capcut", display_name="CapCut", elements=[
        UIElement(name="Tools", role="MenuItem", clickable=True)]), tmp_path)
    ctx = SessionAppContext(last_app="CapCut")           # so match() claims it
    mech = FakeMech(invoke_ok=True, title="Untitled - CapCut")
    r = _run_text(AppNavigateSkill(mech, base_dir=tmp_path, ctx=ctx),
                  "open the tools menu")
    assert r.success and "Tools" in r.speech             # no app named, focus won
    assert mech.focused == ["CapCut"] and mech.invoked[0][1] == "Tools"


def test_locate_falls_back_to_last_app(tmp_path):
    from core.app_context import SessionAppContext
    from skills.software_learn import AppLocateSkill
    from software.knowledge import AppMap, UIElement, save_map
    save_map(AppMap(app_id="capcut", display_name="CapCut", elements=[
        UIElement(name="Export", role="Button", clickable=True)]), tmp_path)
    ctx = SessionAppContext(last_app="CapCut")
    mech = FakeMech(title="Some Unrelated Window")        # foreground not learned
    r = _run_text(AppLocateSkill(mech, base_dir=tmp_path, ctx=ctx),
                  "where's the export button")
    assert r.success and "Export" in r.speech             # used the remembered app


def test_llm_path_no_app_uses_focused_app_and_remembers_it(tmp_path):
    from core.app_context import SessionAppContext
    from skills.software_learn import AppLocateSkill
    from software.knowledge import AppMap, UIElement, save_map
    save_map(AppMap(app_id="capcut", display_name="CapCut", elements=[
        UIElement(name="Tools", role="MenuItem", clickable=True)]), tmp_path)
    ctx = SessionAppContext()                             # empty
    mech = FakeMech(title="project - CapCut")
    skill = AppLocateSkill(mech, base_dir=tmp_path, ctx=ctx)
    r = asyncio.run(skill.execute(SkillRequest(text="", args={"target": "tools"})))
    assert r.success and "Tools" in r.speech
    assert ctx.last_app == "CapCut"                       # focused app remembered


def test_learn_notes_the_app_in_context(tmp_path):
    from core.app_context import SessionAppContext
    from skills.software_learn import LearnAppSkill
    ctx = SessionAppContext()
    _run(LearnAppSkill(FakeMech(can_focus=True),
                       FakeWalker([RawElement("X", "Button")]),
                       base_dir=tmp_path, ctx=ctx), args={"app": "CapCut"})
    assert ctx.last_app == "CapCut"


def test_context_resolution_asks_which_app_when_unknown(tmp_path):
    from core.app_context import SessionAppContext
    from skills.software_learn import AppLocateSkill
    ctx = SessionAppContext(last_app="CapCut")   # lets match() claim it...
    mech = FakeMech(title="Unrelated")           # ...but nothing learned/focused
    # no map for CapCut here -> not_learned (context resolved an app but it's new)
    r = _run_text(AppLocateSkill(mech, base_dir=tmp_path, ctx=ctx),
                  "where's the tools menu")
    assert r.success is False and r.data.get("reason") == "not_learned"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
