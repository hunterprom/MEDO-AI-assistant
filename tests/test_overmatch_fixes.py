"""Regression tests for the OVER-MATCHES an adversarial-verification workflow
found in this session's widened patterns — each was reproduced by running the
skill's match() and seeing it claim an utterance meant for another skill.

These lock the fences in place: a widened pattern must keep catching its
intended phrasings (positive cases) without grabbing everyday speech (negatives).
"""

from __future__ import annotations

import pytest

from core.config import load_settings


def _settings():
    return load_settings()


# --- play / apps / media -----------------------------------------------------

def test_put_on_household_and_getting_dressed_are_not_plays():
    from skills.sites import PlaySkill
    p = PlaySkill()
    for x in ("put on some coffee", "put on some clothes", "put on some weight",
              "put on a bit of sunscreen", "throw on a bit of deodorant",
              "put on the kettle", "put on your coat"):
        assert p.match(x) is None, x


def test_play_music_on_the_drive_is_local_media_not_google_drive():
    from skills.media import MediaSkill
    from skills.sites import PlaySkill
    assert PlaySkill().match("play some music on the drive") is None
    assert MediaSkill().match("play some music on the drive") is not None


def test_open_the_first_one_for_now_is_not_a_search():
    from skills.sites import PlaySkill
    p = PlaySkill()
    for x in ("open the first one for now", "open the first video for the meeting"):
        assert p.match(x) is None, x
    # but a real named-query first-result play still routes
    assert p.match("play the first video for lofi hip hop") is not None


def test_named_site_play_still_works():
    from skills.sites import PlaySkill
    p = PlaySkill()
    assert p.match("put on some lofi on youtube") is not None
    assert p.match("open spotify and play some jazz") is not None


def test_apps_declines_open_and_navigate():
    from skills.apps import AppsSkill
    a = AppsSkill({"chrome": {"windows": "start chrome"}})
    assert a.match("open chrome and go to youtube") is None
    assert a.match("open chrome and navigate to reddit") is None
    assert a.match("open chrome") is not None                  # bare launch ok


# --- datetime / weather / news ----------------------------------------------

def test_datetime_day_of_week_only_the_clock_sense():
    from skills.datetime_skill import DateTimeSkill
    d = DateTimeSkill()
    assert d.match("what day of the week is it") is not None
    for x in ("what day of the week does the pharmacy close",
              "what day of the week is my flight",
              "what day of the week works best for you"):
        assert d.match(x) is None, x


def test_weather_city_tail_rejects_non_places():
    from skills.weather import _city_from_tail
    assert _city_from_tail("how hot is it in Dubai") == "Dubai"
    assert _city_from_tail("is it raining in Paris") == "Paris"
    for x in ("do I need a jacket in the office", "will it rain at the wedding",
              "do I need a coat for the meeting", "is it raining in this weather"):
        assert _city_from_tail(x) is None, x


def test_news_happening_and_world_of():
    from skills.news import NewsSkill
    n = NewsSkill(_settings().news)
    assert n.match("what's happening") is not None
    assert n.match("what's happening in the world") is not None
    assert n.match("tell me about the world") is not None
    for x in ("what's happening with my order", "what's happening at the party tonight",
              "tell me about the world of warcraft"):
        assert n.match(x) is None, x


# --- council / circuit -------------------------------------------------------

def test_circuit_declines_social_and_fitness_senses():
    from skills.circuit import CircuitSkill
    c = CircuitSkill(_settings(), None, None)
    for x in ("how do you connect with people", "how do you connect the dots on this",
              "connect me with the sales team", "connect me to a human",
              "what's a good circuit for my abs workout",
              "give me a circuit for legs at the gym"):
        assert c.match(x) is None, x
    # real electronics questions still route
    assert c.match("how do you wire up an LED to an Arduino") is not None
    assert c.match("circuit diagram for an ultrasonic sensor") is not None


def test_ask_specialist_run_this_by_legal_resolves():
    from skills.council import AskSpecialistSkill
    s = AskSpecialistSkill(_settings())
    assert s.match("run this by legal") is not None            # 'legal' -> lawyer


# --- vision ------------------------------------------------------------------

def test_see_camera_holding_idioms_declined():
    from skills.vision_skill import SeeCameraSkill
    cam = SeeCameraSkill(_settings())
    assert cam.match("what am I holding up") is not None
    assert cam.match("what am I holding up right now") is not None
    for x in ("what am I holding you to", "what am I holding out for"):
        assert cam.match(x) is None, x


def test_screen_question_not_stolen_by_camera_modal():
    from skills.vision_skill import SeeCameraSkill, SeeScreenSkill
    cam, scr = SeeCameraSkill(_settings()), SeeScreenSkill(_settings())
    for x in ("can you see anything on my screen", "do you see anything on the display",
              "will you see me tomorrow", "do you see anything wrong with that plan"):
        assert cam.match(x) is None, x
    # the two real screen questions still reach see_screen
    assert scr.match("can you see anything on my screen") is not None
    # and the bare camera request still works
    assert cam.match("can you see me") is not None


def test_see_screen_ignores_ability_complaints():
    from skills.vision_skill import SeeScreenSkill
    scr = SeeScreenSkill(_settings())
    assert scr.match("see my screen") is not None
    for x in ("I can't see the screen", "I can see the screen fine"):
        assert scr.match(x) is None, x


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
