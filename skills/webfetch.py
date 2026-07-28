"""Read a web page aloud — fetch a URL, extract its text, summarize or answer.

The gap this closes: MEDO could *search* the web (``web_search``) and *open* a
page in the browser (``open_website``), but it could never **read** one. "what
does this link say" ended at a browser window the user still had to read
themselves — the wrong answer for an assistant you talk to with your hands
full. This skill turns a URL into a spoken paragraph.

Two routes, one skill (the house pattern):

* **Fast path** ("read https://…", "прочитај ја страницата"): fetch, extract,
  then summarize with the injected ``summarize`` coroutine so the reply is
  prose, not a wall of scraped text.
* **LLM tool path**: the model calls ``web_fetch`` with a URL — and is held to
  a tighter trust rule than the fast path (below).

Extraction is a **pure function** (:func:`extract_text`), so the parsing rules
are unit-testable without a network and without a new HTML dependency.

SAFETY — this skill is a prompt-injection surface, and is deliberately narrow.
A fetched page (or an imported document) is UNTRUSTED text: a URL sitting
inside one is a third party's instruction, not the user's. So only URLs the
user themselves supplied are fetched silently; anything that reaches the LLM
tool path without ``url_source == "user"`` asks for a spoken yes first, naming
the host. Non-http(s) schemes (``file:``, ``javascript:``, ``data:``) are
refused outright and are never offered as a confirmation — see
``docs/Decisions.md``.
"""

from __future__ import annotations

import html as html_lib
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

import httpx

from core import mk
from core.config import WebFetchConfig
from skills.base import Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)

Summarize = Callable[[str, str], Awaitable[str]]

#: How much extracted text the summarizer is handed. The local models run with
#: num_ctx 4096, so more than this is silently truncated by Ollama anyway —
#: better to cut at a place we chose.
MAX_SUMMARY_CHARS = 6000
#: How much raw text is spoken when there is no summarizer (degraded mode).
#: A voice reply nobody can interrupt should not be four minutes long.
MAX_SPOKEN_CHARS = 700

#: Content types that can contain readable prose. Everything else (PDF, images,
#: video, JSON APIs) is refused BEFORE the body is read — see fetch_page.
ALLOWED_CONTENT_TYPES = (
    "text/html", "application/xhtml+xml", "text/plain", "text/xml",
    "application/xml",
)

#: Spoken failures, EN then MK. One sentence per reason, so no failure path can
#: reach the user as a traceback.
_FAILURES: dict[str, tuple[str, str]] = {
    "timeout": ("{host} took too long to answer, so I stopped waiting.",
                "{host} не одговори навреме, па прекинав."),
    "unreachable": ("I couldn't reach {host}.",
                    "Не можев да дојдам до {host}."),
    "http_error": ("{host} refused to give me that page.",
                   "{host} го одби моето барање за таа страница."),
    "redirects": ("{host} kept redirecting me somewhere else, so I gave up.",
                  "{host} ме врти во круг со пренасочувања, па се откажав."),
    "not_html": ("{host} isn't a readable web page.",
                 "{host} не е читлива веб страница."),
    "too_big": ("That page is far too large to read, so I stopped downloading it.",
                "Таа страница е преголема за читање, па прекинав со преземањето."),
    "empty": ("I fetched {host} but found no readable text on it.",
              "Ја земав {host}, но не најдов читлив текст на неа."),
}


class FetchError(Exception):
    """A fetch that failed in a way we can say out loud.

    ``reason`` keys :data:`_FAILURES`; carrying a code rather than a message
    keeps the bilingual wording in one table instead of scattered at each
    raise site.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# --- extraction (pure, no network) -------------------------------------------
#
# A focused regex pass instead of a parser dependency: MEDO needs "the words on
# this page", not a DOM. The order below is the whole trick — comments and
# script/style bodies are removed BEFORE any text is taken out, so a page whose
# JavaScript contains prose (or whose comment contains a <div>) can't leak into
# what gets read aloud.

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_DROP_RE = re.compile(
    r"<(script|style|noscript|template|svg)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
#: An UNCLOSED script/style — what the size cap leaves behind when it cuts a
#: page mid-tag. Without this, the truncated tail of a minified bundle would be
#: spoken as if it were the article.
_DROP_UNCLOSED_RE = re.compile(
    r"<(?:script|style|noscript)\b[^>]*>.*", re.IGNORECASE | re.DOTALL
)
_MAIN_RE = re.compile(r"<main\b[^>]*>(.*?)</main\s*>", re.IGNORECASE | re.DOTALL)
_ARTICLE_RE = re.compile(r"<article\b[^>]*>(.*?)</article\s*>",
                         re.IGNORECASE | re.DOTALL)
_TITLE_RE = re.compile(r"<title\b[^>]*>(.*?)</title\s*>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]*>")
_WS_RE = re.compile(r"\s+")


def _flatten(fragment: str) -> str:
    """Tags out, entities decoded, whitespace collapsed to single spaces."""
    text = _TAG_RE.sub(" ", fragment)
    return _WS_RE.sub(" ", html_lib.unescape(text)).strip()


def extract_text(html: str) -> str:
    """The readable text of an HTML document.

    Pure: same input, same output, no I/O — which is what makes the parsing
    rules testable without pretending to be the internet.
    """
    if not html:
        return ""
    cleaned = _COMMENT_RE.sub(" ", html)
    cleaned = _DROP_RE.sub(" ", cleaned)
    cleaned = _DROP_UNCLOSED_RE.sub(" ", cleaned)
    # Prefer the article body when the page marks one. Navigation, cookie
    # banners and footers are most of a modern page's text, and summarizing
    # them instead of the article is the classic scraper failure.
    body = _MAIN_RE.search(cleaned) or _ARTICLE_RE.search(cleaned)
    if body is not None:
        cleaned = body.group(1)
    return _flatten(cleaned)


def extract_title(html: str) -> str:
    """The page's <title>, flattened for speech ("" when it has none)."""
    found = _TITLE_RE.search(_COMMENT_RE.sub(" ", html or ""))
    return _flatten(found.group(1)) if found is not None else ""


def trim_for_speech(text: str, limit: int = MAX_SPOKEN_CHARS) -> str:
    """Cut ``text`` to ``limit`` at a word boundary — nothing mid-syllable."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit // 2 else cut).rstrip(" ,;:") + "..."


# --- URLs ---------------------------------------------------------------------

#: Suffixes that must NOT read as a top-level domain, so "read notes.md" stays
#: a file for the file skills and never becomes an http request.
_NOT_A_TLD = (
    r"(?:md|markdown|txt|rtf|pdf|docx?|xlsx?|pptx?|odt|csv|tsv|json|ya?ml|toml"
    r"|ini|cfg|conf|log|py|js|ts|jsx|tsx|css|scss|xml|sql|bat|ps1|exe|dll|png"
    r"|jpe?g|gif|bmp|svg|webp|ico|zip|rar|7z|tar|gz|mp3|mp4|wav|flac|mkv|mov"
    r"|avi|bak|tmp)"
)

#: What counts as a URL in an utterance. The first branch matches ANY scheme on
#: purpose — including the ones we refuse — so "read file:///etc/passwd" is
#: answered with a spoken refusal instead of falling through to the LLM, which
#: would cheerfully discuss it.
_URL = (
    r"(?:[a-zA-Z][\w+.\-]*:(?://)?[^\s<>\"']+"
    r"|www\.[^\s<>\"']+"
    rf"|[\w-]+(?:\.[\w-]+)*\.(?!{_NOT_A_TLD}\b)[a-zA-Z]{{2,}}(?:/[^\s<>\"']*)?)"
)

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")

#: read / summarize, Macedonian. Kept here rather than in :mod:`core.mk`:
#: "прочитај" is this skill's verb alone, and the shared vocabulary is for
#: verbs several skills need. Longest forms first, as everywhere else.
_MK_READ = r"прочитајте|прочитај|прочитаj|читај|резимирај|сумирај"

#: The spoken noun for "the page" — what "прочитај ја страницата" points at.
_MK_PAGE = (r"веб\s+страна(?:та)?|страницата|страната|страница|страна|"
            r"статијата|статија|сајтот|линкот|линк")


def normalize_url(raw: str) -> str:
    """A spoken/typed URL -> one we can hand to httpx ("" when there is none).

    A scheme-less host gets https (not http): the fallback should be the secure
    one. Anything that already declares a scheme keeps it — including the bad
    ones, because they have to be *recognized* to be refused.
    """
    url = (raw or "").strip().strip("<>\"'“”").rstrip(".,;:!?)»")
    if not url:
        return ""
    if not _SCHEME_RE.match(url):
        url = f"https://{url}"
    return url


def is_fetchable(url: str) -> bool:
    """True only for http(s). file:/javascript:/data: are not web pages."""
    return urlparse(url).scheme in ("http", "https")


def url_is_trusted(context: dict[str, Any]) -> bool:
    """May this URL be fetched without asking the user first?

    The trust boundary: ``url_source`` is set by whatever *put* the URL in
    front of MEDO. "user" is the only provenance that needs no confirmation.
    A missing source is trusted only on the fast path — a regex match means
    the user said the URL themselves, out loud. On the LLM tool path there is
    no such guarantee: the model may have read the link out of a document or a
    previous page, which is untrusted third-party text.
    """
    source = str(context.get("url_source") or "").strip().lower()
    if source == "user":
        return True
    if source:                      # "document", "tool", or anything unknown
        return False
    return context.get("via") != "tool"


# --- fetching -----------------------------------------------------------------


async def fetch_page(url: str, config: WebFetchConfig) -> tuple[str, str]:
    """Fetch ``url``; return ``(html, content_type)``. Raises :class:`FetchError`.

    Streamed with a hard byte cap rather than downloaded-then-measured: the
    point of a cap is to never pull a 400 MB "page" onto this machine, and
    ``len(response.content)`` has already lost that fight by the time it can be
    read. Leaving the loop closes the response, so the rest never arrives.
    """
    # httpx timeouts are per-operation, so a server dripping one byte per
    # second forever would satisfy every one of them. This deadline bounds the
    # whole read.
    deadline = time.monotonic() + config.timeout_s
    body = b""
    encoding = "utf-8"
    content_type = ""
    try:
        async with httpx.AsyncClient(
            timeout=config.timeout_s,
            follow_redirects=True,
            max_redirects=config.max_redirects,
            headers={"User-Agent": config.user_agent},
        ) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                content_type = (response.headers.get("content-type") or "")
                content_type = content_type.split(";")[0].strip().lower()
                if content_type and content_type not in ALLOWED_CONTENT_TYPES:
                    # Decided on the headers alone: finding out a URL was a
                    # video by downloading the video is the bug, not the check.
                    raise FetchError("not_html", content_type)
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > config.max_bytes:
                        # Name the numbers, not just "too big": the log should
                        # say WHAT was read vs the limit it blew (e.g. a page
                        # 1000x the cap stops here, having pulled ~one chunk over).
                        logger.info(
                            "web_fetch: %s over the byte cap — read %d B, cap %d B; "
                            "aborting the stream", url, size, config.max_bytes)
                        raise FetchError(
                            "too_big",
                            f"read {size:,} B, over the {config.max_bytes:,} B cap")
                    if time.monotonic() > deadline:
                        raise FetchError(
                            "timeout",
                            f"body stalled after {size:,} B / {config.timeout_s:.0f}s")
                    chunks.append(chunk)
                encoding = getattr(response, "charset_encoding", None) or "utf-8"
                body = b"".join(chunks)
    except httpx.TimeoutException as exc:
        raise FetchError("timeout", str(exc)) from exc
    except httpx.TooManyRedirects as exc:
        raise FetchError("redirects", str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise FetchError("http_error", str(exc.response.status_code)) from exc
    except httpx.HTTPError as exc:      # connect/DNS/protocol — i.e. unreachable
        raise FetchError("unreachable", str(exc)) from exc
    try:
        return body.decode(encoding, errors="replace"), content_type
    except LookupError:                 # a charset label Python doesn't know
        return body.decode("utf-8", errors="replace"), content_type


class WebFetchSkill(Skill):
    """Fetch a page the user pointed at and say what it says."""

    name = "web_fetch"
    description = (
        "Fetch a web page by URL and summarize it, or answer a question about "
        "its contents. Use when the user gives a link and wants to know what "
        "is ON it — not when they want the page opened in a browser."
    )
    # Not controls_pc: this reads over the network, it doesn't touch the
    # machine. The gate that matters here is the URL-provenance one below.

    patterns = [
        # "fetch https://x.com", "read the page at example.com/post",
        # "summarize https://x.com/y"
        re.compile(
            rf"\b(?:fetch|read|summari[sz]e|scrape|pull)\s+(?:me\s+)?"
            rf"(?:the\s+|this\s+|that\s+)?"
            rf"(?:web\s*)?(?:page\s+(?:at\s+)?|url\s+(?:at\s+)?|link\s+(?:at\s+)?|"
            rf"site\s+(?:at\s+)?|article\s+(?:at\s+)?)?"
            rf"(?P<url>{_URL})",
            re.IGNORECASE),
        # "what does https://x.com say" / "… say about batteries"
        re.compile(rf"\bwhat\s+does\s+(?P<url2>{_URL})\s+say\b"
                   rf"(?:\s+about\s+(?P<q>.+))?", re.IGNORECASE),
        # Deictic: "read this page". No URL in the words, so it comes from the
        # caller's context (the HUD/browser knows what is open) — and when
        # there is none we ask rather than guess.
        re.compile(r"\b(?:read|summari[sz]e)\s+(?:me\s+)?(?:this|that|the)\s+"
                   r"(?:web\s*)?(?:page|article|link|site)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+does\s+(?:this|that|the)\s+(?:web\s*)?"
                   r"(?:page|article|link|site)\s+say\b"
                   r"(?:\s+about\s+(?P<q2>.+))?", re.IGNORECASE),
        # Direct synonym: "what is this page about" / "what's this article about".
        re.compile(r"\bwhat(?:'?s| is)\s+(?:this|that|the)\s+(?:web\s*)?"
                   r"(?:page|article|link|site)\s+about\b", re.IGNORECASE),
        # MK: "прочитај ја страницата", "прочитај https://…", "сумирај ја статијата"
        re.compile(rf"\b(?:{_MK_READ}){mk.CLITICS}\s+(?P<url_mk>{_URL})", re.IGNORECASE),
        re.compile(rf"\b(?:{_MK_READ}){mk.CLITICS}\s+(?:{_MK_PAGE})\b", re.IGNORECASE),
        # MK: "земи https://…" — "get me that page".
        re.compile(rf"\b(?:преземи|донеси|земи){mk.CLITICS}\s+(?P<url_mk2>{_URL})",
                   re.IGNORECASE),
        # MK: "што пишува на https://…", "што пишува на страницата за батерии"
        re.compile(rf"\bшто\s+(?:пишува|стои)\s+(?:{mk.ON})\s+(?P<url_mk3>{_URL})"
                   rf"(?:\s+за\s+(?P<qmk>.+))?", re.IGNORECASE),
        re.compile(rf"\bшто\s+(?:пишува|стои)\s+(?:{mk.ON})\s+(?:{_MK_PAGE})"
                   rf"(?:\s+за\s+(?P<qmk2>.+))?", re.IGNORECASE),
    ]

    def __init__(self, config: WebFetchConfig | None = None,
                 summarize: Summarize | None = None) -> None:
        self._config = config or WebFetchConfig()
        self._summarize = summarize

    # -- the pieces execute() is made of --------------------------------------

    def _url_from(self, request: SkillRequest, groups: dict[str, Any]) -> str:
        """The URL this request is about, normalized ("" when there is none)."""
        raw = (request.args.get("url")
               or groups.get("url") or groups.get("url2")
               or groups.get("url_mk") or groups.get("url_mk2")
               or groups.get("url_mk3")
               # "read this page": the app tells us which page is on screen.
               or request.context.get("url") or "")
        return normalize_url(str(raw))

    @staticmethod
    def _question_from(request: SkillRequest, groups: dict[str, Any]) -> str:
        """The "…about X" part, if the user asked one rather than "summarize"."""
        return (request.args.get("question") or groups.get("q") or groups.get("q2")
                or groups.get("qmk") or groups.get("qmk2") or "").strip(" ?.!")

    def _confirmation(self, request: SkillRequest, host: str,
                      speak_mk: bool) -> SkillResult:
        """Ask before fetching a link that did not come from the user."""
        from_document = str(request.context.get("url_source", "")).lower() == "document"
        if speak_mk:
            lead = ("Тој линк доаѓа од документ, не од тебе." if from_document
                    else "Тој линк не доаѓа од тебе.")
            speech = f"{lead} Да ја земам {host}?"
        else:
            lead = ("That link came from a document, not from you."
                    if from_document else "That link didn't come from you.")
            speech = f"{lead} Fetch {host}?"
        return SkillResult(speech, needs_confirmation=True,
                           data={"host": host, "confirm": "url_source"})

    async def _speak_page(self, text: str, title: str, host: str, question: str,
                          speak_mk: bool) -> str:
        """Page text -> the sentence(s) MEDO says.

        With a summarizer this is an LLM pass (asked in the user's language, so
        the answer comes back in it). Without one — no model, or the model is
        offline — it degrades to the top of the page rather than failing: a
        partial answer beats "I can't".
        """
        body = text[:MAX_SUMMARY_CHARS]
        if self._summarize is None:
            lead = (f"Еве што пишува на {host}." if speak_mk
                    else f"Here's what {host} says.")
            return f"{lead} {trim_for_speech(body)}"
        if question:
            query = question
        elif speak_mk:
            query = f"Сумирај ја оваа веб страница на македонски: {title or host}"
        else:
            query = f"Summarize this web page: {title or host}"
        try:
            summary = (await self._summarize(query, body) or "").strip()
        except Exception:   # a broken summarizer must not lose the page
            logger.exception("web_fetch: summarizer failed for %s", host)
            summary = ""
        if summary:
            return summary
        lead = (f"Еве што пишува на {host}." if speak_mk
                else f"Here's what {host} says.")
        return f"{lead} {trim_for_speech(body)}"

    async def execute(self, request: SkillRequest) -> SkillResult:
        speak_mk = mk.is_cyrillic(request.text)
        groups = request.match.groupdict() if request.match else {}
        url = self._url_from(request, groups)
        if not url:
            return SkillResult(
                "Која страница да ја прочитам?" if speak_mk
                else "Which page should I read?", success=False)

        if not is_fetchable(url):
            # Refused outright, never offered as a confirmation: a file:// or
            # javascript: "page" has no reading to do — see docs/Decisions.md.
            scheme = urlparse(url).scheme or "that"
            return SkillResult(
                f"Не отворам {scheme} врски — читам само веб страници."
                if speak_mk else
                f"I only read web pages over http and https, not {scheme} links.",
                success=False, data={"url": url, "refused": "scheme"})

        host = urlparse(url).hostname or url
        if not url_is_trusted(request.context) and not request.context.get("confirmed"):
            logger.info("web_fetch: untrusted URL source for %s — asking first", host)
            return self._confirmation(request, host, speak_mk)

        try:
            html, _content_type = await fetch_page(url, self._config)
        except FetchError as exc:
            # Expected, classified failures: the reason + its specifics (sizes,
            # status) go to the log; the user hears one clean sentence.
            logger.info("web_fetch: %s failed (%s)", host, exc)
            english, macedonian = _FAILURES[exc.reason]
            return SkillResult(
                (macedonian if speak_mk else english).format(host=host),
                success=False, data={"url": url, "error": exc.reason})
        except Exception:
            # Anything fetch_page did NOT anticipate. Log with the traceback so
            # the exact failing line is captured for debugging, and still answer
            # gracefully instead of crashing the turn / leaking a stack trace.
            logger.exception("web_fetch: unexpected error reading %s", url)
            return SkillResult(
                "Не успеав да ја прочитам таа страница."
                if speak_mk else
                f"Something unexpected went wrong reading {host}.",
                success=False, data={"url": url, "error": "unexpected"})

        text = extract_text(html)
        if not text:
            english, macedonian = _FAILURES["empty"]
            return SkillResult((macedonian if speak_mk else english).format(host=host),
                               success=False, data={"url": url})

        title = extract_title(html)
        question = self._question_from(request, groups)
        speech = await self._speak_page(text, title, host, question, speak_mk)
        return SkillResult(speech, data={"url": url, "host": host, "title": title,
                                         "chars": len(text), "text": text})

    def tool_schema(self) -> dict[str, Any]:
        # ``url_source`` is deliberately NOT a parameter. Provenance is set by
        # whatever handed MEDO the link (the importer, the document index, the
        # voice loop) — never by the model, which is the component a malicious
        # page talks to. A tool argument the model fills in would let an
        # injected page declare its own link "from the user".
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The http(s) URL of the page to read.",
                        },
                        "question": {
                            "type": "string",
                            "description": "Optional question to answer from the "
                                           "page. Omit for a summary.",
                        },
                    },
                    "required": ["url"],
                },
            },
        }
