"""Regression tests for the 16 confirmed bugs from the skill-audit workflow.

Most are the recurring "a natural phrasing misses the fast-path regex and falls
to the LLM, which then denies a capability MEDO has" class; a few are real logic
bugs (volume args-path, greedy weather city, timer-label discard, a security
guard that refused a LOCAL defensive request).
"""

from __future__ import annotations

import pytest

from core.config import load_settings


def _settings():
    return load_settings()


# --- #1 brightness phrasings + direction -------------------------------------

def test_brightness_natural_phrasings():
    from skills.desktop import BrightnessSkill
    b = BrightnessSkill()

    def direction(x):
        m = b.match(x)
        if not m:
            return None
        gd = m.groupdict()
        d = gd.get("ud") or gd.get("updown") or gd.get("updown2")
        if d:
            return d
        if gd.get("up") or gd.get("brighter"):
            return "up"
        if gd.get("down") or gd.get("dimmer"):
            return "down"
        return gd.get("level")

    assert direction("dim the screen") == "down"
    assert direction("lower the brightness") == "down"
    assert direction("make the screen brighter") == "up"
    assert direction("turn down the brightness") == "down"
    assert direction("turn up the brightness") == "up"
    assert direction("set brightness to 40") == "40"


# --- #1/#15 volume: brightness collision + args-path -------------------------

def test_volume_does_not_steal_brightness():
    from skills.system import VolumeSkill
    v = VolumeSkill()
    assert v.match("turn down the brightness") is None
    assert v.match("turn up the screen brightness") is None
    assert v.match("turn it down") is not None       # still a volume command
    assert v.match("volume up") is not None


# --- #4 type/press accept a polite/modal wrapper -----------------------------

def test_type_and_press_accept_modal_prefix():
    from skills.desktop import PressKeysSkill, TypeTextSkill
    t, p = TypeTextSkill(), PressKeysSkill()
    for x in ("can you type hello", "could you type out my email", "please type hi"):
        assert t.match(x) is not None, x
    for x in ("can you press enter", "could you hit control s"):
        assert p.match(x) is not None, x
    assert t.match("type hello") is not None          # bare form still works


# --- #2/#3 trailing "file" noun ---------------------------------------------

def test_drop_file_noun():
    from skills.file_edit import _drop_file_noun
    assert _drop_file_noun("config file") == "config"
    assert _drop_file_noun("shopping list file") == "shopping list"
    assert _drop_file_noun("makefile") == "makefile"      # no space, untouched
    assert _drop_file_noun("notes.txt") == "notes.txt"
    assert _drop_file_noun("file") == "file"              # never empties


# --- #5 browser click/type accept a modal wrapper ----------------------------

def test_browser_click_type_accept_modal_prefix():
    from skills.browser import BrowserControlSkill
    s = _settings()
    s.browser.enabled = True

    class _Session:
        is_open = True

    b = BrowserControlSkill(s.browser, _Session())
    for x in ("click sign in", "can you click sign in", "please click sign in",
              "would you click the login button",
              "can you type medo into the search box"):
        assert b.match(x) is not None, x


# --- #6 webfetch "what is this page about" -----------------------------------

def test_webfetch_what_is_this_page_about():
    import inspect

    import skills.webfetch as wf
    cls = next(c for _, c in inspect.getmembers(wf, inspect.isclass)
               if getattr(c, "name", None) == "web_fetch")
    try:
        w = cls(_settings())
    except TypeError:
        w = cls()
    assert w.match("what is this page about") is not None
    assert w.match("what's this article about") is not None
    assert w.match("what does this page say") is not None


# --- #7 security: local defensive request is not "offensive" ------------------

def test_is_offensive_allows_local_defensive_refuses_production():
    from skills.security import is_offensive
    assert is_offensive("scan my computer for malware") is False
    assert is_offensive("check this machine for rootkits") is False
    assert is_offensive("write malware") is True
    assert is_offensive("install a keylogger on my computer") is True
    assert is_offensive("scan 8.8.8.8 for open ports") is True


# --- #8 bench doesn't steal app-install questions ----------------------------

def test_bench_declines_app_install_questions():
    from skills.bench import BenchInventory, BenchSkill
    b = BenchSkill(_settings(), BenchInventory(":memory:"))
    assert b.match("do i have any games installed") is None
    assert b.match("do i have any browsers") is None
    assert b.match("do i have any 10k resistors") is not None   # real bench query


# --- #9 see_camera modal expansion -------------------------------------------

def test_see_camera_modal_expansion():
    from skills.vision_skill import SeeCameraSkill
    cam = SeeCameraSkill(_settings())
    for x in ("could you see me", "would you see me", "do you see anything",
              "tell me what you see"):
        assert cam.match(x) is not None, x


# --- #10 weather strips a trailing time word ---------------------------------

def test_weather_strips_trailing_time_word():
    from skills.weather import _strip_time_words
    assert _strip_time_words("tomorrow") == ""
    assert _strip_time_words("London tomorrow") == "London"
    assert _strip_time_words("New York") == "New York"


# --- #11/#12/#13 circuit -----------------------------------------------------

def test_circuit_fixes():
    from skills.circuit import CircuitSkill
    c = CircuitSkill(_settings(), None, None)
    assert c.match("како да спојам ардуино со диода") is not None   # Cyrillic ј
    assert c.match("how can I hook up a motor to the board") is not None
    assert c.match("how should I wire up this led") is not None
    # everyday networking sense is NOT a circuit
    assert c.match("connect my phone to the wifi") is None
    assert c.match("connect my laptop to the projector") is None


# --- #14 timer label survives a trailing delay -------------------------------

def test_timer_label_survives_trailing_delay():
    from skills.timers import TimerSkill
    t = TimerSkill(lambda m: None, None)
    assert t._extract_label("remind me to call mom in 5 minutes") == "call mom"
    assert t._extract_label("remind me to call mom in five minutes") == "call mom"
    assert t._extract_label("remind me to fill in the form") == "fill in the form"
    assert t._extract_label("set a timer to 5 minutes") == ""   # pure duration


# --- #16 datetime doesn't hijack "what time does X close" --------------------

def test_datetime_does_not_hijack_time_questions():
    from skills.datetime_skill import DateTimeSkill
    d = DateTimeSkill()
    assert d.match("what time is it") is not None
    assert d.match("what's the time") is not None
    assert d.match("what time does the pharmacy close") is None
    assert d.match("how much time do I have") is None


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
