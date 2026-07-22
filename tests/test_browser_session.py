"""BrowserSession against a fake page — the guards, not Playwright.

The session is where the blocklist is enforced and where every page action
funnels through ``_live_page``. That guard is re-checked on the CURRENT url,
not just at navigation, so a page that redirects itself onto a blocked host
stops working mid-session. None of that needs a browser to test; it needs a
page object that reports a url.
"""

from __future__ import annotations

import pytest

from core.config import BrowserConfig
from skills.browser import BrowserBlocked, BrowserSession, BrowserUnavailable


class _FakeLocator:
    def __init__(self, page, selector):
        self._page, self._selector = page, selector

    @property
    def first(self):
        return self

    async def wait_for(self, state=None, timeout=None):
        if self._selector in self._page.missing:
            raise TimeoutError(f"{self._selector} never appeared")

    async def get_attribute(self, name):
        return self._page.attrs.get(name)

    async def inner_text(self):
        return self._page.text_of.get(self._selector, "clicked thing")

    async def click(self):
        self._page.actions.append(("click_selector", self._selector))


class _FakePage:
    """The slice of a Playwright page BrowserSession actually touches."""

    def __init__(self, url="https://example.com/", title="Example"):
        self.url = url
        self._title = title
        self.actions: list[tuple] = []
        self.evaluated: list[str] = []
        self.missing: set[str] = set()
        self.attrs: dict[str, str] = {}
        self.text_of: dict[str, str] = {}
        self.elements_result: list[dict] | None = None
        self.evaluate_raises = False

    async def title(self):
        return self._title

    async def goto(self, url, wait_until=None):
        self.url = url
        self.actions.append(("goto", url))

    async def evaluate(self, script):
        if self.evaluate_raises:
            raise RuntimeError("page went away mid-evaluate")
        self.evaluated.append(script)
        if "data-medo-i" in script:
            return self.elements_result if self.elements_result is not None else []
        return "page text here"

    async def click(self, selector):
        self.actions.append(("click", selector))

    async def fill(self, selector, text):
        self.actions.append(("fill", selector, text))

    async def press(self, selector, key):
        self.actions.append(("press", selector, key))

    async def go_back(self, wait_until=None):
        self.actions.append(("back",))

    async def wait_for_timeout(self, ms):
        pass

    def locator(self, selector):
        return _FakeLocator(self, selector)

    @property
    def keyboard(self):
        page = self

        class _Keyboard:
            async def press(self, keys):
                page.actions.append(("key", keys))

        return _Keyboard()

    @property
    def mouse(self):
        page = self

        class _Mouse:
            async def wheel(self, dx, dy):
                page.actions.append(("wheel", dy))

        return _Mouse()


def _session(page, **config):
    session = BrowserSession(BrowserConfig(enabled=True, **config))
    session._page = page                 # skip the real Playwright launch
    return session


@pytest.fixture
def page():
    return _FakePage()


@pytest.fixture
def session(page):
    return _session(page)


# --- the actions ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_goto_returns_the_title(session, page):
    assert await session.goto("https://example.com/x") == "Example"
    assert ("goto", "https://example.com/x") in page.actions


@pytest.mark.asyncio
async def test_click_fill_press_scroll_and_back(session, page):
    await session.click(3)
    await session.fill(1, "hello")
    await session.press("Enter")
    await session.scroll(2)
    await session.back()
    assert page.actions == [
        ("click", '[data-medo-i="3"]'),
        ("fill", '[data-medo-i="1"]', "hello"),
        ("key", "Enter"),
        ("wheel", 800),
        ("back",),
    ]


@pytest.mark.asyncio
async def test_fill_submits_only_when_asked(session, page):
    await session.fill(0, "query", submit=True)
    assert ("press", '[data-medo-i="0"]', "Enter") in page.actions


@pytest.mark.asyncio
async def test_elements_returns_the_snapshot(session, page):
    page.elements_result = [{"i": 0, "label": "Sign in"}]
    assert await session.elements() == [{"i": 0, "label": "Sign in"}]


@pytest.mark.asyncio
async def test_a_broken_snapshot_is_empty_not_an_exception(session, page):
    """A page that navigates mid-evaluate must not crash the whole turn."""
    page.evaluate_raises = True
    assert await session.elements() == []


@pytest.mark.asyncio
async def test_text_is_truncated_to_the_limit(session):
    assert await session.text(limit=4) == "page"


@pytest.mark.asyncio
async def test_where_reports_title_and_url(session):
    assert await session.where() == ("Example", "https://example.com/")


@pytest.mark.asyncio
async def test_click_selector_prefers_the_title_attribute(session, page):
    page.attrs["title"] = "The Real Title"
    assert await session.click_selector("a.result") == "The Real Title"
    assert ("click_selector", "a.result") in page.actions


@pytest.mark.asyncio
async def test_click_selector_falls_back_to_inner_text(session, page):
    page.text_of["a.result"] = "Inner text\nsecond line"
    # First line only: the rest is layout, not a name.
    assert await session.click_selector("a.result") == "Inner text"


@pytest.mark.asyncio
async def test_click_selector_raises_when_nothing_appears(session, page):
    page.missing.add("a.never")
    with pytest.raises(TimeoutError):
        await session.click_selector("a.never", timeout_s=0.01)


# --- the blocklist -------------------------------------------------------------


@pytest.mark.asyncio
async def test_navigating_to_a_blocked_host_is_refused(page):
    session = _session(page, blocked_domains=["bank"])
    with pytest.raises(BrowserBlocked):
        await session.goto("https://mybank.com/login")
    assert page.actions == [], "a blocked navigation must not reach the page"


@pytest.mark.asyncio
async def test_a_redirect_onto_a_blocked_host_stops_further_actions(page):
    """The guard re-runs on the CURRENT url, so arriving there is not enough."""
    session = _session(page, blocked_domains=["bank"])
    page.url = "https://mybank.com/account"        # as if redirected after landing
    for action in (session.click(1), session.elements(), session.text(),
                   session.scroll(1), session.back()):
        with pytest.raises(BrowserBlocked):
            await action


@pytest.mark.asyncio
async def test_an_unblocked_host_passes_the_guard(page):
    session = _session(page, blocked_domains=["bank"])
    await session.click(1)
    assert ("click", '[data-medo-i="1"]') in page.actions


# --- lifecycle -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_actions_before_anything_is_open_are_refused():
    session = BrowserSession(BrowserConfig(enabled=True))   # no page injected
    assert session.is_open is False
    with pytest.raises(BrowserUnavailable):
        await session.click(0)


@pytest.mark.asyncio
async def test_close_clears_the_session(session):
    assert session.is_open is True
    await session.close()
    assert session.is_open is False


@pytest.mark.asyncio
async def test_close_is_safe_to_call_twice(session):
    await session.close()
    await session.close()          # must not raise on an already-dead session
    assert session.is_open is False
