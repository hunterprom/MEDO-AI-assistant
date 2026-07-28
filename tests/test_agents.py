"""The agent cluster graph — the data behind the HUD's sphere view.

The one property that matters: the graph is derived from the registry, so a
skill that exists shows up, and a skill that doesn't, doesn't. These tests pin
that, plus the council-as-stars special case and the plugin fallback.
"""

from __future__ import annotations



from core.agents import (
    DOMAIN_ORDER,
    agent_graph,
    build_spheres,
    prettify,
    skill_domain_index,
)
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult


class _Sk(Skill):
    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("ok")


def make(name, module="skills.fake", controls_pc=False):
    cls = type(f"S_{name}", (_Sk,), {"name": name, "controls_pc": controls_pc})
    cls.__module__ = module
    return cls()


def reg(*skills):
    r = SkillRegistry()
    for s in skills:
        r.register(s)
    return r


class FakeSpecialist:
    def __init__(self, key, title):
        self.key, self.title = key, title


# -- grouping ------------------------------------------------------------------

def test_stars_come_from_registered_skills():
    r = reg(make("see_screen"), make("web_search"), make("notes"))
    spheres = {s.key: s for s in build_spheres(r, council=[])}
    assert "Screen Read" in [st.label for st in spheres["vision"].stars]
    assert "Web Search" in [st.label for st in spheres["web"].stars]
    assert "Notes" in [st.label for st in spheres["knowledge"].stars]


def test_empty_domains_are_dropped():
    spheres = build_spheres(reg(make("weather")), council=[])
    keys = [s.key for s in spheres]
    assert keys == ["environment"]           # only the one that has a star


def test_spheres_follow_the_declared_order():
    r = reg(make("notes"), make("weather"), make("see_screen"))  # knowledge, env, vision
    order = [s.key for s in build_spheres(r, council=[])]
    # vision precedes knowledge precedes environment, regardless of reg order
    assert order == ["vision", "knowledge", "environment"]
    declared = [k for k, _ in DOMAIN_ORDER]
    assert order == [k for k in declared if k in order]


def test_a_new_skill_appears_without_touching_the_map():
    # An unmapped skill in a normal module lands in 'other', never vanishes.
    r = reg(make("teleport", module="skills.teleport"))
    spheres = {s.key: s for s in build_spheres(r, council=[])}
    assert "other" in spheres
    assert spheres["other"].stars[0].label == "Teleport"


def test_a_plugin_skill_lands_in_plugins():
    r = reg(make("dice", module="medo_plugin_example_dice"))
    spheres = {s.key: s for s in build_spheres(r, council=[])}
    assert "plugins" in spheres and spheres["plugins"].stars[0].label == "Dice"


def test_controls_pc_is_carried_onto_the_star():
    r = reg(make("power", controls_pc=True), make("datetime", controls_pc=False))
    spheres = {s.key: s for s in build_spheres(r, council=[])}
    power = next(st for st in spheres["system"].stars if st.label == "Power")
    time_ = next(st for st in spheres["agenda"].stars if st.label == "Time")
    assert power.controls_pc is True and time_.controls_pc is False


# -- council -------------------------------------------------------------------

def test_council_stars_are_the_specialists_not_the_skills():
    r = reg(make("ask_specialist"), make("convene_council"), make("circuit_help"))
    council = [FakeSpecialist("electrical", "the electrical engineer"),
               FakeSpecialist("robotics", "the roboticist")]
    spheres = {s.key: s for s in build_spheres(r, council=council)}
    labels = [st.label for st in spheres["experts"].stars]
    # the specialists, cleanly named — not "ask_specialist"
    assert labels == ["Electrical", "Robotics"]
    assert all(st.skill is None for st in spheres["experts"].stars)


def test_disabling_a_specialist_removes_its_star():
    r = reg(make("ask_specialist"))
    council = [FakeSpecialist("law", "the lawyer")]        # economist switched off
    spheres = {s.key: s for s in build_spheres(r, council=council)}
    assert [st.label for st in spheres["experts"].stars] == ["Law"]


def test_council_skills_still_light_the_council_sphere():
    # The three council skills aren't stars, but firing them must light the
    # sphere — so the lighting index maps them to 'experts'.
    r = reg(make("ask_specialist"), make("convene_council"))
    idx = skill_domain_index(r)
    assert idx["ask_specialist"] == "experts"
    assert idx["convene_council"] == "experts"


# -- lighting index ------------------------------------------------------------

def test_skill_domain_index_covers_every_skill():
    r = reg(make("see_screen"), make("power"), make("teleport", module="skills.x"))
    idx = skill_domain_index(r)
    assert set(idx) == {"see_screen", "power", "teleport"}
    assert idx["see_screen"] == "vision"
    assert idx["power"] == "system"
    assert idx["teleport"] == "other"


# -- graph shape ---------------------------------------------------------------

def test_agent_graph_is_json_able_and_shaped():
    r = reg(make("see_screen"), make("ask_specialist"))
    g = agent_graph(r, council=[FakeSpecialist("law", "the lawyer")])
    assert set(g) == {"spheres", "skillDomain"}
    vision = next(s for s in g["spheres"] if s["key"] == "vision")
    star = vision["stars"][0]
    assert set(star) == {"label", "skill", "controlsPc"}
    assert star["skill"] == "see_screen"
    import json

    json.dumps(g)                             # must serialize


def test_prettify():
    assert prettify("open_in_editor") == "Open In Editor"
    assert prettify("news") == "News"


# -- the real registry ---------------------------------------------------------

def test_the_live_registry_produces_a_sane_graph():
    from core.config import load_settings
    from core.docindex import DocumentIndex
    from main import Announcer, build_registry

    r = build_registry(load_settings(), Announcer(),
                       doc_index=DocumentIndex(":memory:", None, []))
    g = agent_graph(r)
    keys = {s["key"] for s in g["spheres"]}
    # the headline spheres exist
    for expected in ("vision", "experts", "web", "knowledge", "system"):
        assert expected in keys, expected
    # every star with a skill points at a real registered skill
    names = {s.name for s in r.all()}
    for sphere in g["spheres"]:
        for star in sphere["stars"]:
            if star["skill"] is not None:
                assert star["skill"] in names
    # every registered skill is reachable in the lighting index
    assert names <= set(g["skillDomain"])
