"""Browser control: clicking, typing and reading real web pages over the DOM.

No browser is launched here — :class:`FakeSession` stands in for
``BrowserSession``, which is why every skill takes one by injection. What's
worth pinning is the element *matching* (spoken text -> the right button) and
the safety gates, not Playwright's ability to click.
"""

from __future__ import annotations

import pytest

from core.config import BrowserConfig
from skills.base import SkillRequest
from skills.browser import (
    NOT_ENABLED,
    BrowserAgentSkill,
    BrowserBlocked,
    BrowserControlSkill,
    BrowserUnavailable,
    browser_opener,
    describe,
    find_element,
    is_blocked,
    score_element,
)
from skills.sites import open_with

ELEMENTS = [
    {"i": 0, "tag": "input", "label": "Search the site", "type": "text",
     "typeable": True, "onscreen": True},
    {"i": 1, "tag": "button", "label": "Sign in", "type": "", "typeable": False,
     "onscreen": True},
    {"i": 2, "tag": "button", "label": "Subscribe", "type": "", "typeable": False,
     "onscreen": True},
    {"i": 3, "tag": "a", "label": "Next page", "type": "", "typeable": False,
     "onscreen": False},
]


class FakeSession:
    """Records what a real BrowserSession would have done to the page."""

    def __init__(self, elements=None, text="Hello from the page.", raises=None):
        self._elements = list(ELEMENTS if elements is None else elements)
        self._text = text
        self._raises = raises
        self.actions: list[tuple] = []
        self.is_open = True
        self.closed = False

    def _maybe_raise(self):
        if self._raises is not None:
            raise self._raises

    async def elements(self):
        self._maybe_raise()
        return self._elements

    async def click(self, index):
        self._maybe_raise()
        self.actions.append(("click", index))

    async def fill(self, index, text, submit=False):
        self._maybe_raise()
        self.actions.append(("fill", index, text, submit))

    async def scroll(self, amount):
        self.actions.append(("scroll", amount))

    async def back(self):
        self.actions.append(("back",))

    async def text(self, limit=2000):
        self._maybe_raise()
        return self._text

    async def where(self):
        return "Fake page", "https://example.com/"

    async def goto(self, url):
        self._maybe_raise()
        self.actions.append(("goto", url))
        return "Fake page"

    async def close(self):
        self.closed = True


@pytest.fixture
def config():
    return BrowserConfig(enabled=True)


@pytest.fixture
def session():
    return FakeSession()


@pytest.fixture
def skill(config, session):
    return BrowserControlSkill(config, session)


async def _say(skill, text):
    return await skill.execute(SkillRequest(text=text, match=skill.match(text)))


# --- pure helpers -------------------------------------------------------------


def test_is_blocked_matches_the_host_not_the_query():
    assert is_blocked("https://mybank.com/login", ["bank"])
    assert is_blocked("https://WWW.PAYPAL.COM/", ["paypal"])
    assert not is_blocked("https://example.com", ["bank"])
    assert not is_blocked("https://example.com", [])
    # The word in a *query string* is not the host — this must not trip.
    assert not is_blocked("https://duckduckgo.com/?q=bank+opening+hours", ["bank"])


def test_score_element_ranks_exact_over_partial():
    exact = {"label": "Sign in", "onscreen": False}
    prefix = {"label": "Sign in with Google", "onscreen": False}
    unrelated = {"label": "Careers", "onscreen": False}
    assert score_element(exact, "sign in") > score_element(prefix, "sign in")
    assert score_element(unrelated, "sign in") == 0.0


def test_score_element_prefers_what_is_on_screen():
    visible = {"label": "Subscribe", "onscreen": True}
    offscreen = {"label": "Subscribe", "onscreen": False}
    assert score_element(visible, "subscribe") > score_element(offscreen, "subscribe")


def test_find_element_by_spoken_text():
    assert find_element(ELEMENTS, "sign in")["i"] == 1
    assert find_element(ELEMENTS, "subscribe")["i"] == 2
    assert find_element(ELEMENTS, "next")["i"] == 3
    assert find_element(ELEMENTS, "nothing like this") is None


def test_find_element_typeable_only_returns_fields():
    # "type X in search" must land on the input, never on a link saying search.
    assert find_element(ELEMENTS, "search", typeable=True)["i"] == 0
    assert find_element(ELEMENTS, "sign in", typeable=True) is None


def test_describe_numbers_elements_for_the_prompt():
    text = describe(ELEMENTS)
    assert "[1] button: Sign in" in text
    assert "(input)" in text


# --- patterns -----------------------------------------------------------------


@pytest.mark.parametrize("phrase", [
    "click sign in",
    "click on the first video",
    "click the sign in button",
    "type robot dog into the search box",
    "enter hello into the message field",
    "scroll down",
    "go back",
    "read the page",
    "what can i click",
    "close the tab",
    "кликни на sign in",
    "напиши робот во полето за пребарување",
    "скролај надолу",
    "врати се назад",
    "прочитај ја страната",
])
def test_matches_page_actions(skill, phrase):
    assert skill.match(phrase) is not None, phrase


@pytest.mark.parametrize("phrase", [
    "press control s",          # PressKeysSkill — never a page click
    "hit enter",
    "type hello world",         # no field -> TypeTextSkill (focused window)
    "close the browser",        # AppsSkill kills the browser app
    "what time is it",
    "open youtube",
])
def test_does_not_steal_other_skills(skill, phrase):
    assert skill.match(phrase) is None, phrase


# --- actions ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_click_by_visible_text(skill, session):
    r = await _say(skill, "click sign in")
    assert r.success and session.actions == [("click", 1)]
    assert "Sign in" in r.speech


@pytest.mark.asyncio
async def test_type_into_a_named_field_and_submit_a_search(skill, session):
    r = await _say(skill, "type robot dog into the search box")
    # Search fields are useless without the Enter that submits them.
    assert session.actions == [("fill", 0, "robot dog", True)]
    assert r.success and r.data["submitted"] is True


@pytest.mark.asyncio
async def test_type_falls_back_to_the_only_input(config):
    only = [{"i": 7, "tag": "input", "label": "", "type": "text",
             "typeable": True, "onscreen": True}]
    session = FakeSession(elements=only)
    skill = BrowserControlSkill(config, session)
    r = await _say(skill, "type hello into the box")
    assert r.success and session.actions == [("fill", 7, "hello", False)]


@pytest.mark.asyncio
async def test_scroll_back_read_and_list(skill, session):
    assert (await _say(skill, "scroll down")).success
    assert (await _say(skill, "go back")).success
    read = await _say(skill, "read the page")
    listed = await _say(skill, "what can i click")
    assert session.actions == [("scroll", 1), ("back",)]
    assert "Hello from the page." in read.speech
    assert "Sign in" in listed.speech


@pytest.mark.asyncio
async def test_scroll_up_goes_negative(skill, session):
    await _say(skill, "scroll up")
    assert session.actions == [("scroll", -1)]


@pytest.mark.asyncio
async def test_close_closes_the_session(skill, session):
    r = await _say(skill, "close the tab")
    assert r.success and session.closed is True


@pytest.mark.asyncio
async def test_missing_element_fails_without_clicking(skill, session):
    r = await _say(skill, "click the enormous purple button")
    assert r.success is False and session.actions == []


@pytest.mark.asyncio
async def test_macedonian_click_answers_in_macedonian(skill, session):
    r = await _say(skill, "кликни на sign in")
    assert r.success and session.actions == [("click", 1)]
    assert r.speech.startswith("Кликнав")


# --- safety -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_config_declines_politely(session):
    skill = BrowserControlSkill(BrowserConfig(enabled=False), session)
    r = await _say(skill, "click sign in")
    assert r.success is False and r.speech == NOT_ENABLED
    assert session.actions == []


@pytest.mark.asyncio
async def test_blocked_domain_is_reported_not_clicked(config):
    session = FakeSession(raises=BrowserBlocked("mybank.com is on my blocked list."))
    skill = BrowserControlSkill(config, session)
    r = await _say(skill, "click sign in")
    assert r.success is False and "blocked list" in r.speech


@pytest.mark.asyncio
async def test_missing_playwright_is_explained(config):
    session = FakeSession(raises=BrowserUnavailable("Playwright isn't installed."))
    skill = BrowserControlSkill(config, session)
    r = await _say(skill, "read the page")
    assert r.success is False and "Playwright" in r.speech


def test_both_skills_are_gated_by_the_pc_switch(skill, config, session):
    assert skill.controls_pc is True
    agent = BrowserAgentSkill(config, session, think=None)
    assert agent.controls_pc is True and agent.requires_confirmation is True


# --- the multi-step agent -----------------------------------------------------


def _thinker(*replies):
    """A fake brain that returns the given action JSONs in order."""
    queue = list(replies)

    async def think(prompt: str) -> str:
        return queue.pop(0) if queue else '{"action": "done", "say": "Finished."}'

    return think


@pytest.mark.asyncio
async def test_agent_asks_before_touching_anything(config, session):
    agent = BrowserAgentSkill(config, session, think=_thinker())
    r = await agent.execute(SkillRequest(text="on the site, subscribe",
                                         match=agent.match("on the site, subscribe")))
    assert r.needs_confirmation is True and session.actions == []


@pytest.mark.asyncio
async def test_agent_runs_actions_then_stops_on_done(config, session):
    agent = BrowserAgentSkill(config, session, think=_thinker(
        '{"action": "click", "index": 1}',
        'sure thing! {"action": "type", "index": 0, "text": "medo"}',   # prose tolerated
        '{"action": "done", "say": "Signed in and searched."}',
    ))
    r = await agent.execute(SkillRequest(
        text="on the site, sign in and search", context={"confirmed": True},
        match=agent.match("on the site, sign in and search")))
    assert r.success and r.speech == "Signed in and searched."
    assert session.actions == [("click", 1), ("fill", 0, "medo", True)]


@pytest.mark.asyncio
async def test_agent_stops_at_the_step_cap(session):
    config = BrowserConfig(enabled=True, max_steps=2)
    agent = BrowserAgentSkill(config, session,
                              think=_thinker('{"action": "scroll"}',
                                             '{"action": "scroll"}',
                                             '{"action": "scroll"}'))
    r = await agent.execute(SkillRequest(
        text="on the site, scroll forever", context={"confirmed": True},
        match=agent.match("on the site, scroll forever")))
    assert r.data["steps"] == 2 and len(session.actions) == 2


@pytest.mark.asyncio
async def test_agent_gives_up_on_unparseable_reply(config, session):
    agent = BrowserAgentSkill(config, session, think=_thinker("I have no idea"))
    r = await agent.execute(SkillRequest(
        text="on the site, do something", context={"confirmed": True},
        match=agent.match("on the site, do something")))
    assert r.success is False and session.actions == []


@pytest.mark.asyncio
async def test_agent_without_a_brain_declines(config, session):
    agent = BrowserAgentSkill(config, session, think=None)
    r = await agent.execute(SkillRequest(
        text="on the site, subscribe", context={"confirmed": True},
        match=agent.match("on the site, subscribe")))
    assert r.success is False and session.actions == []


# --- routing opens through the controlled browser -----------------------------


@pytest.mark.asyncio
async def test_browser_opener_navigates_the_session(session):
    opener = browser_opener(session)
    assert await opener("https://example.com") is True
    assert session.actions == [("goto", "https://example.com")]


@pytest.mark.asyncio
async def test_browser_opener_falls_back_to_the_system_browser(monkeypatch):
    # Losing browser *control* must not cost the user the page itself.
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
    session = FakeSession(raises=BrowserUnavailable("no playwright"))
    assert await browser_opener(session)("https://example.com") is True
    assert opened == ["https://example.com"]


@pytest.mark.asyncio
async def test_open_with_handles_sync_and_async_openers():
    assert await open_with(lambda url: f"sync {url}", "u") == "sync u"

    async def coro(url):
        return f"async {url}"

    assert await open_with(coro, "u") == "async u"
