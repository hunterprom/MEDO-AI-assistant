"""Interact with websites — click buttons, fill fields, read pages.

The gap this closes: ``web_open``/``sites`` could *open* a page and the screen
agent could *guess pixels* at it, but neither could reliably press "Sign in".
This drives a real browser through Playwright and works on the **DOM**, not on
pixels — MEDO asks the page which buttons and inputs it has, matches the one
you named, and clicks that element. No coordinate guessing, so it doesn't care
about window size, theme, zoom or how dense the page is.

Two entry points, same session:

* :class:`BrowserControlSkill` — one deterministic action per utterance
  ("click sign in", "type medo in the search box", "scroll down", "go back",
  "read the page"), in English and Macedonian.
* :class:`BrowserAgentSkill` — a bounded multi-step loop for "on the site, log
  in and open my inbox", where an LLM picks the next action from the element
  list. Text-only prompts, so the main brain drives it, not the vision model.

Safety by construction:

* ``controls_pc`` on both — the HUD PC-CONTROL switch blocks them entirely.
* ``blocked_domains`` is checked on navigation *and* before every action, so a
  redirect can't carry MEDO onto your bank.
* The agent loop needs a spoken yes and is capped at ``browser.max_steps``.
* Playwright is an optional dependency: absent, every skill says so plainly
  and nothing else breaks.

The browser runs on a persistent profile (``browser.profile_dir``), so you log
into a site once and MEDO is still logged in tomorrow.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

from core import mk
from core.config import PROJECT_ROOT, BrowserConfig
from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

NOT_INSTALLED = (
    "My browser control isn't installed. Run 'pip install playwright' in the "
    "MEDO virtualenv and turn on browser.enabled in config."
)
NOT_ENABLED = "Browser control is switched off in my config."
NO_PAGE = "I don't have a page open yet — tell me a site to open first."

#: What counts as an interactive element. Kept deliberately broad: modern pages
#: build buttons out of divs, so role/tabindex/onclick matter as much as <button>.
_INTERACTIVE_JS = """
(() => {
  const sel = 'a,button,input,textarea,select,summary,[role=button],[role=link],'
            + '[role=tab],[role=menuitem],[role=checkbox],[role=textbox],'
            + '[role=combobox],[role=searchbox],[onclick],[tabindex]';
  const out = [];
  let i = 0;
  for (const el of document.querySelectorAll(sel)) {
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) continue;              // zero-size/hidden
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none' || st.opacity === '0')
      continue;
    if (el.disabled) continue;
    const tag = el.tagName.toLowerCase();
    const label = (
      el.getAttribute('aria-label') || el.innerText || el.value ||
      el.getAttribute('placeholder') || el.getAttribute('title') ||
      el.getAttribute('name') || el.getAttribute('alt') || ''
    ).replace(/\\s+/g, ' ').trim().slice(0, 90);
    const typeable = ['input', 'textarea', 'select'].includes(tag);
    if (!label && !typeable) continue;                       // nothing to name it by
    el.setAttribute('data-medo-i', String(i));
    out.push({
      i, tag, label,
      type: el.getAttribute('type') || '',
      typeable,
      onscreen: r.top >= 0 && r.top < innerHeight,
    });
    i++;
    if (i >= 120) break;                                     // keep the prompt sane
  }
  return out;
})()
"""

#: Page text for "read the page" — main content if the page marks one.
_TEXT_JS = """
(() => {
  const el = document.querySelector('main,[role=main],article') || document.body;
  return (el.innerText || '').replace(/\\n{3,}/g, '\\n\\n').trim();
})()
"""


def is_blocked(url: str, blocked: list[str]) -> bool:
    """True when ``url``'s host matches any blocked substring. Pure.

    Substring on the hostname only — a blocked "bank" must not be triggered by
    a search query that happens to contain the word.
    """
    if not blocked:
        return False
    host = (urlparse(url).hostname or url).lower()
    return any(b.strip().lower() in host for b in blocked if b.strip())


def score_element(element: dict, phrase: str) -> float:
    """How well an element's label answers the spoken phrase. Pure, 0 = no match.

    Ranked so the obvious reading wins: an exact label beats a prefix, which
    beats a substring, which beats "most words in common". Ties break toward
    what's actually on screen — with duplicate "Subscribe" buttons down a long
    page, the one you can see is the one you meant.
    """
    label = (element.get("label") or "").strip().lower()
    want = phrase.strip().lower()
    if not label or not want:
        return 0.0
    if label == want:
        score = 1.0
    elif label.startswith(want) or want.startswith(label):
        score = 0.85
    elif want in label:
        score = 0.7
    elif label in want:
        score = 0.6
    else:
        want_words = {w for w in re.findall(r"\w+", want) if len(w) > 1}
        label_words = {w for w in re.findall(r"\w+", label) if len(w) > 1}
        if not want_words or not label_words:
            return 0.0
        overlap = len(want_words & label_words) / len(want_words)
        if overlap < 0.5:
            return 0.0
        score = 0.3 + 0.2 * overlap
    return score + (0.05 if element.get("onscreen") else 0.0)


def find_element(elements: list[dict], phrase: str, typeable: bool = False) -> dict | None:
    """Best-matching element for a spoken phrase, or None.

    ``typeable`` restricts to fields you can put text into — "type medo in
    search" should never land on a link that happens to say "search".
    """
    pool = [e for e in elements if e.get("typeable")] if typeable else elements
    best, best_score = None, 0.0
    for element in pool:
        score = score_element(element, phrase)
        if score > best_score:
            best, best_score = element, score
    return best


def describe(elements: list[dict], limit: int = 40) -> str:
    """Numbered element list for the LLM prompt."""
    lines = []
    for element in elements[:limit]:
        kind = element.get("type") or element.get("tag")
        mark = " (input)" if element.get("typeable") else ""
        lines.append(f"[{element['i']}] {kind}{mark}: {element.get('label', '')}")
    return "\n".join(lines)


class BrowserSession:
    """One Playwright browser + page, launched on first use and kept warm.

    Everything the skills need funnels through here, which is also why they're
    trivially testable: swap this object for a fake and no browser is involved.
    """

    def __init__(self, config: BrowserConfig) -> None:
        self._config = config
        self._pw = None
        self._context = None
        self._page = None
        self._lock = asyncio.Lock()

    @property
    def is_open(self) -> bool:
        return self._page is not None

    async def _ensure_page(self):
        """Launch the browser on first use; reuse it afterwards."""
        async with self._lock:
            if self._page is not None:
                return self._page
            try:
                from playwright.async_api import async_playwright
            except ImportError as exc:
                raise BrowserUnavailable(NOT_INSTALLED) from exc

            profile = PROJECT_ROOT / self._config.profile_dir
            profile.mkdir(parents=True, exist_ok=True)
            self._pw = await async_playwright().start()
            launch: dict[str, Any] = {
                "user_data_dir": str(profile),
                "headless": self._config.headless,
                # A real window size: sites serve mobile layouts to tiny ones,
                # and the element list would then not match what the user sees.
                "viewport": {"width": 1366, "height": 900},
            }
            # A custom executable (e.g. Opera) WINS over the named channel —
            # Playwright drives that Chromium-based binary, but always on MEDO's
            # OWN profile dir, so it never touches the browser's real profile.
            picked = self._config.executable_path or self._config.channel
            if self._config.executable_path:
                launch["executable_path"] = self._config.executable_path
            elif self._config.channel:
                launch["channel"] = self._config.channel
            try:
                self._context = await self._pw.chromium.launch_persistent_context(**launch)
            except Exception as exc:
                # Nothing custom set -> no fallback; otherwise retry on bundled Chromium.
                if not picked:
                    await self._shutdown()
                    raise BrowserUnavailable(
                        f"I couldn't start the browser: {exc}") from exc
                logger.warning("browser %r failed (%s) — falling back to chromium",
                               picked, exc)
                launch.pop("channel", None)
                launch.pop("executable_path", None)
                try:
                    self._context = await self._pw.chromium.launch_persistent_context(**launch)
                except Exception as exc2:
                    await self._shutdown()
                    raise BrowserUnavailable(
                        "I couldn't start a browser. Run 'playwright install "
                        "chromium' in the MEDO virtualenv.") from exc2
            self._context.set_default_timeout(self._config.timeout_s * 1000)
            pages = self._context.pages
            self._page = pages[0] if pages else await self._context.new_page()
            return self._page

    def _guard(self, url: str) -> None:
        if is_blocked(url, self._config.blocked_domains):
            raise BrowserBlocked(f"{urlparse(url).hostname or url} is on my blocked list.")

    async def goto(self, url: str) -> str:
        self._guard(url)
        page = await self._ensure_page()
        await page.goto(url, wait_until="domcontentloaded")
        return await page.title()

    async def _live_page(self):
        """The current page, re-guarded against the *current* URL.

        Checked here rather than only at navigation because a page can redirect
        itself onto a blocked host after we arrived.
        """
        if self._page is None:
            raise BrowserUnavailable(NO_PAGE)
        self._guard(self._page.url)
        return self._page

    async def elements(self) -> list[dict]:
        page = await self._live_page()
        try:
            return await page.evaluate(_INTERACTIVE_JS) or []
        except Exception:
            logger.exception("element snapshot failed")
            return []

    async def click(self, index: int) -> None:
        page = await self._live_page()
        await page.click(f'[data-medo-i="{index}"]')
        await page.wait_for_timeout(600)      # let navigation/JS settle

    async def fill(self, index: int, text: str, submit: bool = False) -> None:
        page = await self._live_page()
        selector = f'[data-medo-i="{index}"]'
        await page.fill(selector, text)
        if submit:
            await page.press(selector, "Enter")
            await page.wait_for_timeout(800)

    async def click_selector(self, selector: str, timeout_s: float = 10.0) -> str:
        """Click the first element matching a CSS selector; returns its text.

        Used for "play the top result", where the target is known structurally
        (YouTube's first /watch link) rather than by the words on it. Waits for
        the selector because search results arrive after the page load event.
        """
        page = await self._live_page()
        element = page.locator(selector).first
        await element.wait_for(state="visible", timeout=timeout_s * 1000)
        label = ((await element.get_attribute("title"))
                 or (await element.inner_text()) or "").strip()
        await element.click()
        await page.wait_for_timeout(1200)   # let playback actually start
        return label.split("\n")[0][:90]

    async def press(self, keys: str) -> None:
        page = await self._live_page()
        await page.keyboard.press(keys)
        await page.wait_for_timeout(300)

    async def scroll(self, amount: int) -> None:
        page = await self._live_page()
        await page.mouse.wheel(0, amount * 400)
        await page.wait_for_timeout(300)

    async def back(self) -> None:
        page = await self._live_page()
        await page.go_back(wait_until="domcontentloaded")

    async def text(self, limit: int = 2000) -> str:
        page = await self._live_page()
        content = await page.evaluate(_TEXT_JS) or ""
        return content[:limit]

    async def where(self) -> tuple[str, str]:
        """(title, url) of the current page."""
        page = await self._live_page()
        return await page.title(), page.url

    async def close(self) -> None:
        await self._shutdown()

    async def _shutdown(self) -> None:
        for closer in (getattr(self._context, "close", None),
                       getattr(self._pw, "stop", None)):
            if closer is not None:
                try:
                    await closer()
                except Exception:
                    logger.debug("browser shutdown step failed", exc_info=True)
        self._pw = self._context = self._page = None


def browser_opener(session: BrowserSession):
    """An opener for ``web_open``/``sites`` that navigates the controlled browser.

    Same one-argument shape as ``webbrowser.open``, so it drops straight into
    the existing injection point — but async, so pages land in the window MEDO
    can then click on. If the controlled browser can't start (Playwright
    missing, Chrome busy), it falls back to the system browser: the user asked
    to see a page, and losing browser *control* shouldn't cost them the page.
    """
    async def _open(url: str) -> bool:
        try:
            await session.goto(url)
            return True
        except BrowserBlocked:
            # A blocked host must NOT leak to the system browser, which has no
            # block check — enforcing the blocklist is the whole point. Report
            # failure instead of falling back (that fallback is only for a
            # browser that can't START).
            logger.info("refused a blocked host; not falling back to the system browser")
            return False
        except Exception as exc:
            logger.warning("controlled browser failed (%s) — using system browser", exc)
            import webbrowser

            return await asyncio.to_thread(webbrowser.open, url)

    return _open


class BrowserUnavailable(RuntimeError):
    """Playwright missing, browser won't start, or nothing is open yet."""


class BrowserBlocked(RuntimeError):
    """The current or requested host is on ``browser.blocked_domains``."""


# --- deterministic single actions --------------------------------------------


class BrowserControlSkill(Skill):
    """One spoken action on the open page: click / type / scroll / back / read."""

    name = "browser_control"
    controls_pc = True
    description = (
        "Act on the web page MEDO has open: click a button or link by its "
        "visible text, type into a field, scroll, go back, or read the page "
        "out. Use after a site has been opened."
    )

    # "click" only — deliberately NOT "press"/"hit", which belong to
    # PressKeysSkill ("press control s"), nor "select", which is a key chord.
    _CLICK = r"click(?:\s+on)?"
    _TYPE = r"type|enter|write|fill\s+in"
    _CLICK_MK = r"кликни(?:\s+на)?"        # not "притисни" — that's a key press
    _TYPE_MK = r"напиши|внеси|искуцај"
    #: An optional polite/modal wrapper — "can you click…", "please type…" —
    #: which the model narrates ("I can't click for you") instead of doing.
    _POLITE = r"(?:(?:can|could|would|will)\s+you\s+|please\s+)?"

    #: Order matters inside a skill too — ``match()`` returns the first hit, so
    #: the specific forms are listed before the catch-all click.
    patterns = [
        # "type medo into the search box" — needs a field, otherwise this is a
        # plain type-into-the-focused-window and TypeTextSkill should have it.
        re.compile(rf"^\s*{_POLITE}(?:{_TYPE})\s+(?P<text>.+?)\s+(?:in|into)\s+"
                   r"(?:the\s+)?(?P<field>.+?)(?:\s+(?:box|field|bar))?\s*$",
                   re.IGNORECASE),
        re.compile(r"\bscroll\s+(?P<dir>up|down)(?:\s+(?P<times>\d+))?\b", re.IGNORECASE),
        re.compile(r"(?P<back>\bgo\s+back\b|\bback\s+to\s+the\s+previous\s+page\b)",
                   re.IGNORECASE),
        re.compile(r"(?P<read>\bread\s+(?:me\s+)?(?:the|this)\s+page\b"
                   r"|\bwhat(?:'s| is)\s+on\s+(?:the|this)\s+page\b)", re.IGNORECASE),
        re.compile(r"(?P<list>\bwhat\s+can\s+i\s+click\b"
                   r"|\blist\s+the\s+(?:buttons|links)\b)", re.IGNORECASE),
        # "close the browser" is NOT here: that phrase already means "kill the
        # browser app" via AppsSkill. This closes MEDO's own window.
        re.compile(r"(?P<close>\bclose\s+(?:the\s+)?(?:tab|page)\b)", re.IGNORECASE),
        re.compile(rf"^\s*{_POLITE}(?:{_CLICK})\s+(?:the\s+|on\s+the\s+)?(?P<click>.+?)"
                   r"(?:\s+(?:button|link|tab))?\s*$", re.IGNORECASE),
        # --- Macedonian ---
        re.compile(rf"^\s*(?:{_TYPE_MK})\s+(?P<text>.+?)\s+(?:во|на)\s+"
                   r"(?:полето\s+за\s+|полето\s+)?(?P<field>.+?)\s*$", re.IGNORECASE),
        re.compile(r"\bскролај\s+(?P<dir_mk>надолу|нагоре)\b", re.IGNORECASE),
        re.compile(r"(?P<back>\bврати\s+се\s+назад\b|\bназад\s+на\s+претходната\b)",
                   re.IGNORECASE),
        re.compile(r"(?P<read>\bпрочитај\s+(?:ја\s+)?(?:страната|страницата)\b)",
                   re.IGNORECASE),
        re.compile(r"(?P<list>\bшто\s+можам\s+да\s+кликнам\b)", re.IGNORECASE),
        re.compile(r"(?P<close>\bзатвори\s+(?:го\s+)?(?:табот|страната)\b)",
                   re.IGNORECASE),
        re.compile(rf"^\s*(?:{_CLICK_MK})\s+(?:го\s+|ја\s+)?(?P<click>.+?)"
                   r"(?:\s+(?:копче|копчето|линк|линкот))?\s*$", re.IGNORECASE),
    ]

    def __init__(self, config: BrowserConfig, session: BrowserSession | None = None) -> None:
        self._config = config
        self._session = session or BrowserSession(config)

    @property
    def session(self) -> BrowserSession:
        return self._session

    def match(self, text: str):
        """Only claim an utterance when there's actually a page to act on.

        These verbs are not the browser's alone: "write the prompt in notepad"
        matched the type-into-a-field pattern and was answered by the browser
        skill, on a page that did not exist. A page action with no page open is
        never the right reading, so the utterance is left for whoever else
        wants it (the app, file, or typing skills).
        """
        if not self._config.enabled or not self._session.is_open:
            return None
        return super().match(text)

    async def execute(self, request: SkillRequest) -> SkillResult:
        speak_mk = mk.is_cyrillic(request.text)
        if not self._config.enabled:
            return SkillResult(
                "Контролата на прелистувачот е исклучена во конфигурацијата."
                if speak_mk else NOT_ENABLED, success=False)

        gd = request.match.groupdict() if request.match else {}
        args = request.args
        # Fast path: whichever named group fired names the action. LLM path:
        # the model states it outright. No sniffing the raw utterance.
        action = (args.get("action") or "").lower()
        if not action:
            for name in ("close", "read", "list", "back"):
                if gd.get(name):
                    action = name
                    break
            else:
                if gd.get("dir") or gd.get("dir_mk"):
                    action = "scroll"
                elif gd.get("text") and gd.get("field"):
                    action = "type"
                elif gd.get("click"):
                    action = "click"

        try:
            if action == "close":
                await self._session.close()
                return SkillResult("Го затворив прелистувачот." if speak_mk
                                   else "I closed my browser.")

            if action == "read":
                content = await self._session.text()
                if not content.strip():
                    return SkillResult("Страната е празна." if speak_mk
                                       else "That page has no readable text.",
                                       success=False)
                return SkillResult(content[:1200], data={"chars": len(content)})

            if action == "list":
                elements = await self._session.elements()
                names = [e["label"] for e in elements if e.get("label")][:12]
                if not names:
                    return SkillResult("Не гледам ништо за кликање." if speak_mk
                                       else "I don't see anything to click.",
                                       success=False)
                joined = ", ".join(names)
                return SkillResult(
                    f"Можеш да кликнеш: {joined}." if speak_mk
                    else f"You can click: {joined}.", data={"count": len(elements)})

            if action == "back":
                await self._session.back()
                title, _url = await self._session.where()
                return SkillResult(f"Назад на {title}." if speak_mk
                                   else f"Back on {title}.")

            if action == "scroll":
                direction = str(gd.get("dir") or gd.get("dir_mk")
                                or args.get("direction") or "down").lower()
                times = int(gd.get("times") or args.get("amount") or 1)
                down = direction in ("down", "надолу")
                await self._session.scroll(times if down else -times)
                return SkillResult("Скролав." if speak_mk else "Scrolled.")

            if action == "type":
                value = args.get("text") or gd.get("text") or ""
                field = args.get("field") or gd.get("field") or ""
                return await self._type(value, field, speak_mk)

            if action == "click":
                return await self._click(args.get("target") or gd.get("click") or "",
                                         speak_mk)

            return SkillResult("Не разбрав што да направам на страната." if speak_mk
                               else "I didn't catch what to do on the page.",
                               success=False)
        except BrowserBlocked as exc:
            return SkillResult(str(exc), success=False)
        except BrowserUnavailable as exc:
            return SkillResult(str(exc), success=False)
        except Exception:
            logger.exception("browser action failed")
            return SkillResult("Тоа не успеа на страната." if speak_mk
                               else "That didn't work on the page.", success=False)

    async def _click(self, target: str, speak_mk: bool) -> SkillResult:
        target = target.strip(" .?!,")
        elements = await self._session.elements()
        if not elements:
            return SkillResult(NO_PAGE if not self._session.is_open else
                               ("Не гледам ништо за кликање." if speak_mk
                                else "I don't see anything to click."), success=False)
        hit = find_element(elements, target)
        if hit is None:
            return SkillResult(
                f"Не најдов '{target}' на страната." if speak_mk
                else f"I couldn't find '{target}' on the page.", success=False)
        await self._session.click(int(hit["i"]))
        label = hit.get("label") or target
        return SkillResult(f"Кликнав на {label}." if speak_mk
                           else f"Clicked {label}.", data={"label": label})

    async def _type(self, value: str, field: str, speak_mk: bool) -> SkillResult:
        elements = await self._session.elements()
        hit = find_element(elements, (field or "").strip(" .?!,"), typeable=True)
        if hit is None:
            # No field named that — fall back to the only input on the page.
            inputs = [e for e in elements if e.get("typeable")]
            if len(inputs) != 1:
                return SkillResult(
                    f"Не најдов поле '{field}'." if speak_mk
                    else f"I couldn't find a '{field}' field.", success=False)
            hit = inputs[0]
        # Search boxes are useless without the Enter that submits them.
        submit = bool(re.search(r"search|барај|пребарув", f"{field} {hit.get('label', '')}",
                                re.IGNORECASE))
        await self._session.fill(int(hit["i"]), value, submit=submit)
        where = hit.get("label") or field
        return SkillResult(f"Внесов '{value}' во {where}." if speak_mk
                           else f"Typed '{value}' into {where}.",
                           data={"field": where, "submitted": submit})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string",
                                   "enum": ["click", "type", "scroll", "back",
                                            "read", "list", "close"]},
                        "target": {"type": "string",
                                   "description": "Visible text of the button "
                                                  "or link to click."},
                        "text": {"type": "string", "description": "Text to type."},
                        "field": {"type": "string",
                                  "description": "Which field to type into."},
                        "direction": {"type": "string", "enum": ["up", "down"]},
                        "amount": {"type": "integer"},
                    },
                    "required": ["action"],
                },
            },
        }


# --- multi-step agent ---------------------------------------------------------


AGENT_PROMPT = (
    "You are operating a web browser to finish the user's task. You are given "
    "the page title, URL and a numbered list of the interactive elements on "
    "it. Choose the SINGLE next action and reply with ONLY a JSON object — no "
    "prose, no markdown. Schema:\n"
    '{"action": "click|type|scroll|back|goto|done", "index": int, '
    '"text": str, "url": str, "success": bool, "say": str}\n'
    "index is the [n] of the element to act on. For type, index must be an "
    "(input) element and text is what to type. For goto, url is an absolute "
    "URL. When you have actually COMPLETED the task, use \"done\" with success "
    "true and a short spoken result in say. If you cannot do it, use \"done\" "
    "with success FALSE and say why — never claim success for something you did "
    "not finish. Never guess a login or payment."
)


class BrowserAgentSkill(Skill):
    """Bounded multi-step web task: look at the elements, act, repeat."""

    name = "browser_task"
    controls_pc = True
    requires_confirmation = True
    description = (
        "Carry out a multi-step task on a website (e.g. 'on YouTube, open the "
        "first video and like it'). Use only when the user asks MEDO to DO "
        "something on a site, not to describe it."
    )

    patterns = [
        re.compile(r"\bon\s+the\s+(?:site|page|website)[,:]?\s+(?P<task>.+)", re.IGNORECASE),
        re.compile(r"\bin\s+the\s+browser[,:]?\s+(?:please\s+)?(?P<task>.+)", re.IGNORECASE),
        re.compile(r"\bна\s+(?:страната|сајтот)[,:]?\s+(?P<task>.+)", re.IGNORECASE),
        re.compile(r"\bво\s+прелистувачот[,:]?\s+(?P<task>.+)", re.IGNORECASE),
    ]

    def __init__(self, config: BrowserConfig, session: BrowserSession | None = None,
                 think=None) -> None:
        self._config = config
        self._session = session or BrowserSession(config)
        self._think = think          # async (prompt: str) -> str

    async def execute(self, request: SkillRequest) -> SkillResult:
        speak_mk = mk.is_cyrillic(request.text)
        if not self._config.enabled:
            return SkillResult(
                "Контролата на прелистувачот е исклучена." if speak_mk
                else NOT_ENABLED, success=False)
        gd = request.match.groupdict() if request.match else {}
        task = (request.args.get("task") or gd.get("task") or "").strip(" .?!")
        if not task:
            return SkillResult("Што да направам на страната?" if speak_mk
                               else "What should I do on the page?", success=False)
        if self._think is None:
            return SkillResult("My brain is offline, so I can't drive the page.",
                               success=False)
        if not request.context.get("confirmed"):
            return SkillResult(
                f"Ќе преземам контрола над страната за: {task}. Да продолжам?"
                if speak_mk else
                f"I'll take control of the page to: {task}. Shall I go ahead?",
                needs_confirmation=True)

        history: list[str] = []
        try:
            for step in range(self._config.max_steps):
                elements = await self._session.elements()
                title, url = await self._session.where()
                prompt = (f"{AGENT_PROMPT}\n\nTask: {task}\nPage: {title}\nURL: {url}\n"
                          f"Elements:\n{describe(elements)}")
                if history:
                    prompt += "\nDone so far: " + "; ".join(history)
                raw = await self._think(prompt)
                action = _parse_action(raw)
                if action is None:
                    return SkillResult(
                        "Не можев да го одредам следниот чекор, па застанав."
                        if speak_mk
                        else "I couldn't work out the next step, so I stopped.",
                        success=False)
                kind = str(action.get("action", "")).lower()
                if kind == "done":
                    # 'done' means finished OR impossible — honour success so a
                    # gave-up run isn't announced as a completed one.
                    ok = bool(action.get("success", True))
                    say = action.get("say") or (
                        ("Готово." if speak_mk else "Done.") if ok else
                        ("Не можев да го завршам тоа." if speak_mk
                         else "I couldn't finish that on the page."))
                    return SkillResult(say, success=ok, data={"steps": step})
                history.append(await self._apply(action, kind, elements))
            return SkillResult(
                f"Направив {self._config.max_steps} чекори и застанав на лимитот."
                if speak_mk else
                f"I ran {self._config.max_steps} steps and stopped at the limit.",
                data={"steps": self._config.max_steps})
        except (BrowserBlocked, BrowserUnavailable) as exc:
            return SkillResult(str(exc), success=False)
        except Exception:
            logger.exception("browser task failed")
            return SkillResult("Тој чекор не успеа, па застанав." if speak_mk
                               else "That step failed, so I stopped.", success=False)

    async def _apply(self, action: dict, kind: str, elements: list[dict]) -> str:
        """Run one model-chosen action; returns a line for the history."""
        index = action.get("index")
        label = next((e.get("label") for e in elements
                      if str(e.get("i")) == str(index)), str(index))
        if kind == "click":
            await self._session.click(int(index))
            return f"clicked {label!r}"
        if kind == "type":
            await self._session.fill(int(index), str(action.get("text", "")),
                                     submit=True)
            return f"typed {action.get('text', '')!r} into {label!r}"
        if kind == "scroll":
            await self._session.scroll(1)
            return "scrolled down"
        if kind == "back":
            await self._session.back()
            return "went back"
        if kind == "goto":
            title = await self._session.goto(str(action.get("url", "")))
            return f"opened {title!r}"
        return f"ignored unknown action {kind!r}"

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string",
                                 "description": "What to accomplish on the "
                                                "current website."},
                    },
                    "required": ["task"],
                },
            },
        }


def _parse_action(raw: str) -> dict | None:
    """Extract the action JSON from a model reply (tolerates surrounding prose)."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except (ValueError, TypeError):
            return None
    return None
