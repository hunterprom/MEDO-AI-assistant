"""Regression tests for the 2026-07 hallucination-audit workflow.

The recurring bug class: a natural action phrasing misses the fast-path regex
AND the semantic tier (which ships in shadow), reaches the LLM, and the model
fabricates an answer about the topic instead of DOING the thing. The canonical
case is the Gran Turismo transcript — "open up YouTube and search up Gran
Turismo music and open the first video" produced an invented paragraph about
the soundtrack and never opened the browser.

Each test pins a phrasing at the skill's own ``.match()`` (and, where it
matters, the captured groups execute() reads), the same way test_audit_fixes.py
does for the previous round.
"""

from __future__ import annotations

import pytest

from core.config import load_settings
from skills.sites import PlaySkill, SiteSearchSkill, clean_query, resolve_site
from skills.web_open import OpenWebsiteSkill


def _settings():
    return load_settings()


def _play_target(skill: PlaySkill, text: str):
    """Reproduce what PlaySkill.execute() reads from a match: (site, query)."""
    m = skill.match(text)
    if m is None:
        return None
    gd = m.groupdict()
    spoken = (gd.get("site") or gd.get("sitem") or "")
    query = (gd.get("q") or gd.get("qv") or gd.get("qm") or gd.get("qmv")
             or gd.get("qadj") or gd.get("qmood") or "")
    return spoken, clean_query(query)


# --- HB: "open up X" is the phrasal form people actually say -------------------

def test_open_up_a_site_matches():
    o = OpenWebsiteSkill()
    for x in ("open up youtube", "open up YouTube", "fire up youtube",
              "open up reddit"):
        assert o.match(x) is not None, x
    assert o.match("open youtube") is not None            # bare form still works


def test_open_up_site_and_search_still_routes_to_site_search():
    s = SiteSearchSkill()
    m = s.match("open up youtube and search for lofi beats")
    assert m is not None
    gd = m.groupdict()
    assert resolve_site(gd.get("site4") or "").key == "youtube"
    assert "lofi" in (gd.get("q4") or "")


# --- HB: THE Gran Turismo three-step command ---------------------------------

def test_gran_turismo_multi_step_routes_to_play():
    p = PlaySkill()
    utterance = ("Could you open up YouTube and search up Gran Turismo music "
                 "and open the first video that pops up?")
    target = _play_target(p, utterance)
    assert target is not None, "the Gran Turismo utterance must reach play_media"
    site, query = target
    assert resolve_site(site).key == "youtube"
    assert query.lower() == "gran turismo music"


def test_multi_step_variants_route_to_play():
    p = PlaySkill()
    for utterance, want_site, want_q in [
        ("open youtube and search for lofi and play the first result",
         "youtube", "lofi"),
        ("go to youtube and find relaxing jazz then play the top video",
         "youtube", "relaxing jazz"),
    ]:
        target = _play_target(p, utterance)
        assert target is not None, utterance
        site, query = target
        assert resolve_site(site).key == want_site, utterance
        assert query.lower() == want_q, (utterance, query)


def test_open_on_a_named_browser_still_routes():
    """A browser name between the verb and the site ('open ON OPERA youtube')
    must NOT break routing and dump the whole request onto web_search (which
    then fabricates). MEDO opens in its own browser regardless."""
    p, s, o = PlaySkill(), SiteSearchSkill(), OpenWebsiteSkill()
    target = _play_target(p, "open on opera youtube and search up gran turismo "
                             "and play me the first video that shows up")
    assert target is not None
    site, query = target
    assert resolve_site(site).key == "youtube"
    assert query.lower() == "gran turismo"
    # search-only (no play tail) reaches site_search
    assert s.match("open on opera youtube and search up gran turismo") is not None
    assert o.match("open on opera youtube") is not None          # bare open
    assert p.match("open in chrome youtube and play lofi") is not None
    # "on <non-browser>" is not a browser slot -> no false site match
    assert o.match("open on the table the manual") is None


def test_plain_open_and_search_does_not_go_to_play():
    """No 'play the first' tail => this stays a search (results page), so
    PlaySkill must NOT claim it; SiteSearchSkill will."""
    p, s = PlaySkill(), SiteSearchSkill()
    assert p.match("open up youtube and search for lofi beats") is None
    assert s.match("open up youtube and search for lofi beats") is not None


# --- HB: "play the first video for <query>" (named, not deictic) --------------

def test_play_first_result_for_named_query():
    p = PlaySkill()
    for utterance, want_q in [
        ("play the first video for gran turismo music", "gran turismo music"),
        ("open the top result for lofi hip hop on youtube", "lofi hip hop"),
    ]:
        target = _play_target(p, utterance)
        assert target is not None, utterance
        _site, query = target
        assert query.lower() == want_q, (utterance, query)


def test_deictic_play_the_first_video_still_works():
    """The page-already-open deictic must not be turned into a search for the
    literal words 'first video'."""
    p = PlaySkill()
    m = p.match("play the first video")
    assert m is not None
    gd = m.groupdict()
    assert not (gd.get("q") or gd.get("qv") or gd.get("qm"))   # no query captured


# --- HB: "put on / throw on <topic>" is a play, not local media --------------

def test_put_on_a_topic_on_a_site():
    p = PlaySkill()
    target = _play_target(p, "put on some lofi hip hop on youtube")
    assert target is not None
    site, query = target
    assert resolve_site(site).key == "youtube"
    assert "lofi" in query.lower()


def test_put_on_topic_requires_a_named_site():
    """The bare no-site 'put on some X' was removed — it grabbed 'put on some
    coffee / clothes / weight'. A named site is now required for a 'put on' play."""
    p = PlaySkill()
    assert p.match("put on some relaxing jazz") is None        # no site -> not a play
    target = _play_target(p, "put on some relaxing jazz on youtube")
    assert target is not None
    site, query = target
    assert resolve_site(site).key == "youtube"
    assert "relaxing jazz" in query.lower()


def test_put_on_does_not_steal_household_phrases():
    """The no-site 'put on X' must not grab non-media 'put on the kettle' etc.
    — the media quantifier ('some'/'a bit of') is what signals a play."""
    p = PlaySkill()
    for x in ("put on the kettle", "put on your coat", "put on a show",
              "put on the brakes"):
        assert p.match(x) is None, x


def test_put_on_music_stays_with_local_media():
    """'put on some music' is local playback (MediaSkill), not a YouTube play."""
    from skills.media import MediaSkill
    p, media = PlaySkill(), MediaSkill()
    assert p.match("put on some music") is None
    assert media.match("put on some music") is not None
    # ...but a named topic is a YouTube play, not local media.
    assert p.match("put on some lofi on youtube") is not None


# --- HB: "open up <app>" and compound "open <app> and play <x>" --------------

def _apps():
    from skills.apps import AppsSkill
    return AppsSkill({"spotify": {"windows": "start spotify"},
                      "chrome": {"windows": "start chrome"}})


def test_open_up_an_app_matches():
    a = _apps()
    for x in ("open up spotify", "start up chrome", "open spotify"):
        assert a.match(x) is not None, x


def test_open_app_and_play_declines_to_play_skill():
    a, p = _apps(), PlaySkill()
    assert a.match("open spotify and play some jazz") is None      # declined
    target = _play_target(p, "open spotify and play some jazz")
    assert target is not None
    site, query = target
    assert resolve_site(site).key == "spotify"
    assert query.lower() == "jazz"


def test_open_app_and_search_declines():
    a = _apps()
    assert a.match("open chrome and search for cats") is None
    assert a.match("open chrome") is not None                      # bare launch ok


# --- HB: bare media transport verbs ------------------------------------------

def test_bare_transport_verbs():
    from skills.media import MediaSkill
    m = MediaSkill()
    for x, action in [("next", "next"), ("skip", "next"), ("skip this one", "next"),
                      ("pause it", "playpause"), ("resume", "playpause"),
                      ("previous", "previous")]:
        match = m.match(x)
        assert match is not None, x
        assert m._action(x.lower()) == action, x
    # bare forms stay anchored: a stray verb inside a sentence still defers
    assert m.match("skip to the good part of the tutorial") is None


# --- system prompt now names the command-vs-topic rule -----------------------

def test_system_prompt_states_commands_are_commands():
    from llm.prompts import system_prompt
    text = system_prompt(_settings().personality).lower()
    assert "command to carry out" in text
    assert "not a topic to explain" in text


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
