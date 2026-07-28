"""Round-2 audit fixes: the genuine defects and info-agenda fabrications.

Same convention as test_audit_fixes.py — instantiate the skill and pin the
behaviour at ``.match()`` / a pure helper, no network.
"""

from __future__ import annotations

import pytest

from core.config import load_settings
from skills.base import SkillRequest


def _settings():
    return load_settings()


# --- weather: "like" no longer drops the city; snow/bare-rain covered ---------

def test_weather_like_keeps_the_city():
    from skills.weather import WeatherSkill
    w = WeatherSkill(_settings().weather)
    m = w.match("what's the weather like in London")
    assert m is not None
    assert (m.groupdict().get("city") or "").strip().lower() == "london"


def test_weather_covers_snow_and_bare_precipitation():
    from skills.weather import WeatherSkill
    w = WeatherSkill(_settings().weather)
    for x in ("will it snow", "is it going to snow tomorrow", "is it raining",
              "is it snowing", "will it rain"):
        assert w.match(x) is not None, x


# --- datetime: year / month / day-of-week never reach the LLM ----------------

def test_datetime_year_month_dayofweek():
    from skills.datetime_skill import DateTimeSkill
    d = DateTimeSkill()
    for x in ("what year is it", "what month is it", "what day of the week is it",
              "what month are we in"):
        assert d.match(x) is not None, x
        assert d._wants_date(x) is True, x


@pytest.mark.asyncio
async def test_datetime_year_answer_includes_the_real_year():
    from datetime import datetime

    from skills.datetime_skill import DateTimeSkill
    d = DateTimeSkill()
    res = await d.execute(SkillRequest(text="what year is it"))
    assert str(datetime.now().year) in res.speech


# --- news: the Macedonian singular now matches -------------------------------

def test_news_mk_singular_and_plural():
    from skills.news import NewsSkill
    n = NewsSkill(_settings().news)
    for x in ("вест", "веста", "вести", "вестите", "новости", "новост"):
        assert n.match(x) is not None, x


# --- notes: a dictated note whose body has a list-verb is STORED --------------

@pytest.mark.asyncio
async def test_note_that_body_with_show_is_stored_not_listed(tmp_path):
    from core.memory import NoteStore
    from skills.notes import NotesSkill
    store = NoteStore(str(tmp_path / "notes.db"))
    n = NotesSkill(store)
    text = "note that Bob will show up at 5"
    res = await n.execute(SkillRequest(text=text, match=n.match(text)))
    assert "Noted" in res.speech
    assert any("Bob will show up" in note.text for note in store.list())


@pytest.mark.asyncio
async def test_read_my_notes_still_lists(tmp_path):
    from core.memory import NoteStore
    from skills.notes import NotesSkill
    store = NoteStore(str(tmp_path / "notes.db"))
    store.add("buy milk")
    n = NotesSkill(store)
    res = await n.execute(SkillRequest(text="read my notes",
                                       match=n.match("read my notes")))
    assert "buy milk" in res.speech


# --- importer: a path with spaces is captured whole --------------------------

def test_import_path_with_spaces():
    from skills.importer import ImportFileSkill
    from core.safety import PathWhitelist
    s = _settings()
    skill = ImportFileSkill(s, PathWhitelist(s.safety.whitelist_dirs))
    m = skill.match(r"import C:\Users\Me\My Documents\report.pdf")
    assert m is not None
    assert m.groupdict().get("path2") == r"C:\Users\Me\My Documents\report.pdf"
    m2 = skill.match("import ~/Downloads/holiday photos/beach.jpg")
    assert m2 is not None
    assert m2.groupdict().get("path2") == "~/Downloads/holiday photos/beach.jpg"


# --- files: "pull up the invoice on my computer" (needs the disk hint) -------

def test_files_pull_up_with_disk_hint():
    from core.safety import PathWhitelist
    from skills.files import FilesSkill
    f = FilesSkill(PathWhitelist(_settings().safety.whitelist_dirs))
    assert f.match("pull up the invoice on my computer") is not None
    assert f.match("bring up the budget on my laptop") is not None
    # a bare "pull up X" (no disk hint) is a website/app, not a file search
    assert f.match("pull up youtube") is None


# --- circuit: 2nd-person + wiring-diagram; networking still declines ----------

def test_circuit_second_person_and_diagram():
    from skills.circuit import CircuitSkill
    c = CircuitSkill(_settings(), None, None)
    assert c.match("how do you wire up an LED to an Arduino") is not None
    assert c.match("can you give me a wiring diagram for a line-follower robot") is not None
    assert c.match("circuit diagram for an ultrasonic sensor") is not None
    # the everyday networking sense still declines (both persons)
    assert c.match("how do I connect my phone to the wifi") is None
    assert c.match("how do you connect your laptop to the projector") is None


# --- council: rank_specialists matches on word boundaries --------------------

def test_rank_specialists_word_boundary():
    from core.council import rank_specialists
    # "amp" must NOT fire inside "example"; "led" not inside "pulled".
    picked = rank_specialists("what's the best example layout for this")
    assert all(m.key != "electrical" for m in picked)
    # a real electrical trigger still ranks the electrical engineer first
    picked2 = rank_specialists("what resistor and capacitor for this circuit")
    assert picked2 and picked2[0].key == "electrical"


# --- vision: monitor/display + deictic camera; read is a whole word -----------

def test_see_screen_monitor_and_display():
    from skills.vision_skill import SeeScreenSkill
    s = SeeScreenSkill(_settings())
    for x in ("what's on my monitor", "what is on the display",
              "read my monitor", "what's on my screen"):
        assert s.match(x) is not None, x


def test_see_camera_deictic():
    from skills.vision_skill import SeeCameraSkill
    cam = SeeCameraSkill(_settings())
    for x in ("what am I holding up", "what am I holding", "what's in front of me"):
        assert cam.match(x) is not None, x


def test_monitor_query_is_not_stolen_by_camera():
    from skills.vision_skill import SeeCameraSkill, SeeScreenSkill
    cam, scr = SeeCameraSkill(_settings()), SeeScreenSkill(_settings())
    # a "what do you see on my monitor" is a SCREEN query, not the camera
    assert cam.match("what do you see on my monitor") is None
    assert scr.match("what do you see on my monitor") is not None


# --- screen agent: the wrapper phrase isn't captured as the task -------------

def test_screen_agent_task_extraction():
    from skills.screen_agent import ScreenAgentSkill
    tf = ScreenAgentSkill._task_from
    assert tf(SkillRequest(text="open my email for me on the screen")) == "open my email"
    assert tf(SkillRequest(text="do this for me: open notepad")) == "open notepad"
    assert tf(SkillRequest(text="operate my screen and open notepad")) == "open notepad"
    assert tf(SkillRequest(text="anything", args={"task": "click submit"})) == "click submit"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
