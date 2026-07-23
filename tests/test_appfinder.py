"""Finding installed apps, launching them, and installing the missing ones.

The install path is the only one here that changes the machine, so its tests
assert on an empty subprocess call list rather than on wording: the question is
whether anything ran. The package manager is never invoked for real.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skills.appfinder import (
    FoundApp,
    InstallAppSkill,
    LocateAppSkill,
    OpenDiscoveredAppSkill,
    find_app,
    install_command,
    parse_winget_search,
    score_name,
)
from skills.base import SkillRequest

APPS = [
    FoundApp("Obsidian", Path("C:/Start/Obsidian.lnk"), "Start Menu"),
    FoundApp("Visual Studio Code", Path("C:/Start/Code.lnk"), "Start Menu"),
    FoundApp("Visual Studio", Path("C:/Start/VS.lnk"), "Start Menu"),
    FoundApp("Google Chrome", Path("C:/Start/Chrome.lnk"), "Start Menu"),
    FoundApp("Bambu Studio", Path("C:/Start/Bambu.lnk"), "Start Menu"),
    FoundApp("Autodesk Fusion", Path("C:/Start/Fusion.lnk"), "Start Menu"),
]


def test_distinctive_token_finds_a_renamed_app():
    # "open Fusion 360" must still find "Autodesk Fusion" (Autodesk dropped the
    # "360") — the failure the user hit.
    assert find_app("fusion 360", APPS).name == "Autodesk Fusion"
    assert find_app("autodesk fusion", APPS).name == "Autodesk Fusion"


def test_distinctive_token_does_not_match_filler():
    # "how much space do I have in my PC" once captured "in my PC" as an app —
    # short filler words must never resolve to anything.
    for junk in ("in my pc", "in my computer", "some space", "any room"):
        assert find_app(junk, APPS) is None


def test_locate_declines_a_disk_space_question():
    # It must not claim "how much space do I have in my PC" (a disk question).
    assert LocateAppSkill().match("how much space do I have in my PC") is None
    assert LocateAppSkill().match("how much room is left") is None
    # but a genuine "do I have X" is still claimed
    assert LocateAppSkill().match("do I have obsidian") is not None


def test_disk_space_question_routes_to_system_info():
    from core.config import load_settings
    from core.docindex import DocumentIndex
    from main import Announcer, build_registry

    reg = build_registry(load_settings(), Announcer(),
                         doc_index=DocumentIndex(":memory:", None, []))
    for q in ("how much space do I have in my PC",
              "how much disk space do I have",
              "how much free space"):
        hit = reg.find_match(q)
        assert hit is not None and hit[0].name == "system_info", q


# --- matching ------------------------------------------------------------------


@pytest.mark.parametrize("query,expected", [
    ("obsidian", "Obsidian"),
    ("Obsidian", "Obsidian"),
    ("chrome", "Google Chrome"),
    ("bambu", "Bambu Studio"),
    ("visual studio code", "Visual Studio Code"),
    ("vs code", "Visual Studio Code"),        # spoken abbreviation
])
def test_find_app_matches_how_people_say_it(query, expected):
    assert find_app(query, APPS).name == expected


def test_the_shortest_container_wins():
    """"visual studio" must not resolve to "Visual Studio Code"."""
    assert find_app("visual studio", APPS).name == "Visual Studio"


def test_an_unknown_app_is_not_guessed_at():
    for query in ("blender", "photoshop", "", "   "):
        assert find_app(query, APPS) is None


def test_score_name_ranks_exact_over_prefix_over_substring():
    assert score_name("Obsidian", "obsidian") == 1.0
    assert score_name("Obsidian Sandbox", "obsidian") > score_name("My Obsidian", "obsidian")
    assert score_name("Notepad", "obsidian") == 0.0


# --- locating ------------------------------------------------------------------


async def _say(skill, text, monkeypatch, apps=APPS):
    monkeypatch.setattr("skills.appfinder.installed_apps", lambda force=False: apps)
    match = skill.match(text)
    assert match is not None, f"no pattern matched: {text!r}"
    return await skill.execute(SkillRequest(text=text, match=match))


@pytest.mark.parametrize("phrase", [
    "do I have obsidian",
    "do I have obsidian installed",
    "is obsidian installed",
    "where is obsidian installed",
    "дали го имам obsidian",
])
@pytest.mark.asyncio
async def test_locate_finds_an_installed_app(monkeypatch, phrase):
    result = await _say(LocateAppSkill(), phrase, monkeypatch)
    assert result.success and result.data["installed"] is True
    assert "Obsidian" in result.speech


#: The offer to install, not the word "install" — "I don't see X installed"
#: contains that substring and would pass a looser check either way.
_OFFER = "and i'll fetch it"


@pytest.mark.asyncio
async def test_locate_reports_a_missing_app_and_offers_to_install(monkeypatch):
    monkeypatch.setattr("skills.appfinder.package_manager", lambda: ("winget", ["winget"]))
    result = await _say(LocateAppSkill(), "do I have blender", monkeypatch)
    assert result.success is False and result.data["installed"] is False
    assert _OFFER in result.speech.lower()


@pytest.mark.asyncio
async def test_locate_does_not_offer_an_install_without_a_package_manager(monkeypatch):
    monkeypatch.setattr("skills.appfinder.package_manager", lambda: None)
    result = await _say(LocateAppSkill(), "do I have blender", monkeypatch)
    assert _OFFER not in result.speech.lower()


@pytest.mark.asyncio
async def test_listing_summarises_rather_than_reciting(monkeypatch):
    """Reading 152 app names aloud is not an answer."""
    result = await _say(LocateAppSkill(), "what apps do I have", monkeypatch)
    assert result.data["count"] == len(APPS)


def test_locating_is_not_gated_because_it_only_looks():
    assert LocateAppSkill().controls_pc is False


# --- launching -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_opening_a_discovered_app_launches_its_shortcut(monkeypatch):
    opened = []
    monkeypatch.setattr("skills.appfinder.open_path", lambda p: opened.append(p))
    result = await _say(OpenDiscoveredAppSkill(), "open bambu studio", monkeypatch)
    assert result.success and opened == [Path("C:/Start/Bambu.lnk")]


@pytest.mark.asyncio
async def test_opening_something_not_installed_is_left_for_the_llm(monkeypatch):
    """The fast path declines rather than answering "I couldn't find blender".

    Matching is gated on discovery now, so an app nobody has never reaches this
    skill — the router carries on and the model can say something useful about
    it. Reached directly (an LLM tool call), it still fails cleanly.
    """
    opened = []
    monkeypatch.setattr("skills.appfinder.installed_apps", lambda force=False: APPS)
    monkeypatch.setattr("skills.appfinder.open_path", lambda p: opened.append(p))
    skill = OpenDiscoveredAppSkill()

    assert skill.match("open blender") is None
    result = await skill.execute(SkillRequest(text="", args={"app": "blender"}))
    assert result.success is False and opened == []


def test_launching_is_gated_by_the_pc_switch():
    assert OpenDiscoveredAppSkill().controls_pc is True


# --- installing ----------------------------------------------------------------


WINGET_OUTPUT = """
Name                 Id                        Version   Source
-------------------------------------------------------------------
Obsidian             Obsidian.Obsidian         1.5.3     winget
Obsidian Sandbox     Obsidian.Sandbox          0.1       winget
"""


def test_parse_winget_search_pulls_name_and_id():
    rows = parse_winget_search(WINGET_OUTPUT)
    assert rows[0] == ("Obsidian", "Obsidian.Obsidian")
    assert len(rows) == 2


def test_parse_winget_search_survives_junk():
    assert parse_winget_search("") == []
    assert parse_winget_search("no packages found") == []


def test_install_command_passes_the_id_as_one_argument(monkeypatch):
    """A spoken app name reaches this function; it must never become a shell
    string where a space or a semicolon could change the command."""
    monkeypatch.setattr("skills.appfinder.package_manager", lambda: ("winget", ["winget"]))
    argv = install_command("Obsidian.Obsidian")
    assert isinstance(argv, list)
    assert "Obsidian.Obsidian" in argv
    assert not any(";" in part or "&" in part for part in argv)


def test_install_command_is_none_without_a_manager(monkeypatch):
    monkeypatch.setattr("skills.appfinder.package_manager", lambda: None)
    assert install_command("anything") is None


@pytest.mark.asyncio
async def test_install_names_the_package_before_running_anything(monkeypatch):
    ran = []
    monkeypatch.setattr("skills.appfinder.installed_apps", lambda force=False: APPS)
    monkeypatch.setattr("skills.appfinder.package_manager", lambda: ("winget", ["winget"]))
    monkeypatch.setattr("skills.appfinder.search_packages",
                        lambda q, timeout_s=25.0: [("Blender", "BlenderFoundation.Blender")])
    monkeypatch.setattr("subprocess.run", lambda *a, **k: ran.append(a))

    skill = InstallAppSkill()
    phrase = "install blender"
    result = await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert result.needs_confirmation is True
    assert ran == [], "nothing may run before the user has seen the package"
    assert "BlenderFoundation.Blender" in result.speech


@pytest.mark.asyncio
async def test_install_runs_only_after_confirmation(monkeypatch):
    ran = []

    class _Done:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr("skills.appfinder.installed_apps", lambda force=False: APPS)
    monkeypatch.setattr("skills.appfinder.package_manager", lambda: ("winget", ["winget"]))
    monkeypatch.setattr("skills.appfinder.search_packages",
                        lambda q, timeout_s=25.0: [("Blender", "BlenderFoundation.Blender")])
    monkeypatch.setattr("subprocess.run",
                        lambda argv, **k: ran.append(argv) or _Done())

    skill = InstallAppSkill()
    phrase = "install blender"
    result = await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase),
                                              context={"confirmed": True}))
    assert result.success and len(ran) == 1
    assert "BlenderFoundation.Blender" in ran[0]


@pytest.mark.asyncio
async def test_installing_something_already_present_says_so(monkeypatch):
    ran = []
    monkeypatch.setattr("skills.appfinder.installed_apps", lambda force=False: APPS)
    monkeypatch.setattr("skills.appfinder.package_manager", lambda: ("winget", ["winget"]))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: ran.append(a))

    skill = InstallAppSkill()
    phrase = "install obsidian"
    result = await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert result.data.get("already") is True and ran == []


@pytest.mark.asyncio
async def test_a_package_nobody_has_is_reported_not_invented(monkeypatch):
    monkeypatch.setattr("skills.appfinder.installed_apps", lambda force=False: APPS)
    monkeypatch.setattr("skills.appfinder.package_manager", lambda: ("winget", ["winget"]))
    monkeypatch.setattr("skills.appfinder.search_packages", lambda q, timeout_s=25.0: [])

    skill = InstallAppSkill()
    phrase = "install a thing that does not exist"
    result = await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase)))
    assert result.success is False and result.needs_confirmation is False


@pytest.mark.asyncio
async def test_a_failed_install_reports_why(monkeypatch):
    class _Failed:
        returncode, stdout, stderr = 1, "", "No applicable installer found"

    monkeypatch.setattr("skills.appfinder.installed_apps", lambda force=False: APPS)
    monkeypatch.setattr("skills.appfinder.package_manager", lambda: ("winget", ["winget"]))
    monkeypatch.setattr("skills.appfinder.search_packages",
                        lambda q, timeout_s=25.0: [("Blender", "X.Blender")])
    monkeypatch.setattr("subprocess.run", lambda argv, **k: _Failed())

    skill = InstallAppSkill()
    phrase = "install blender"
    result = await skill.execute(SkillRequest(text=phrase, match=skill.match(phrase),
                                              context={"confirmed": True}))
    assert result.success is False and "installer" in result.speech


def test_installing_is_gated_by_the_pc_switch():
    assert InstallAppSkill().controls_pc is True


# --- the fallback must not swallow ordinary sentences -------------------------
#
# "open <anything>" has to be broad, so on its own it claimed any sentence
# containing run/start: "start over please" became the app "over please", and
# "…to run it to find me an interesting video" became an app name in full.


@pytest.mark.parametrize("phrase", [
    "start over please",
    "run the numbers for me",
    "use skills from the metal project to run it to find me an interesting video",
    "open the pod bay doors",
    "start a timer for five minutes",
])
def test_the_fallback_passes_on_sentences_that_name_no_app(monkeypatch, phrase):
    monkeypatch.setattr("skills.appfinder.installed_apps", lambda force=False: APPS)
    assert OpenDiscoveredAppSkill().match(phrase) is None, phrase


@pytest.mark.parametrize("phrase", [
    "open obsidian", "launch bambu studio", "start google chrome",
])
def test_the_fallback_still_claims_real_apps(monkeypatch, phrase):
    monkeypatch.setattr("skills.appfinder.installed_apps", lambda force=False: APPS)
    assert OpenDiscoveredAppSkill().match(phrase) is not None, phrase
