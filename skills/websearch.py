"""Web search — in the browser for a command, out loud for a question.

Two utterances wear the same verb, and they want opposite things:

* **"search up the Mazda RX-7"** is a COMMAND. The answer is the results page,
  in front of you, clickable. Fetching five snippets and having a 30B model
  narrate them took ~20 s to produce a paragraph you can't click — and, being
  a paragraph, it quietly dropped everything you'd have scrolled to. So an
  explicit search verb now opens the configured engine in MEDO's browser (the
  same window ``open_website`` and ``site_search`` use, so "click the first
  result" still lands on it) and says one sentence.
* **"what's the latest on X"**, a bare factual question reaching this skill by
  MEANING, or the model calling ``web_search`` mid-answer, are QUESTIONS. They
  want text: fetch, then summarize (or hand the raw results back on the tool
  path, so the model reasons over real snippets instead of inventing them).

``web_search.browser_scope`` in config.yaml is the dial between those; set it
to "always" and questions open the browser too (tool calls still get their
text as well, so the model can never answer from nothing).

Offline, DuckDuckGo raises and we return a clean spoken message.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

from core import mk
from skills.base import Skill, SkillRequest, SkillResult
from skills.sites import SITES, Site, build_url, clean_query, load_sites, open_with, resolve_site

logger = logging.getLogger(__name__)

Summarize = Callable[[str, str], Awaitable[str]]
_OFFLINE = "I can't search the web right now. I appear to be offline."
_OFFLINE_MK = "Не можам да пребарувам сега — изгледа дека сум офлајн."
_MAX_RESULTS = 5

#: Engines that aren't in the site table, as ``key -> (spoken label, template)``.
#: The table itself is checked first, so ``engine: wikipedia`` (or any of your
#: own ``skills.sites`` entries) works without appearing here.
ENGINES: dict[str, tuple[str, str]] = {
    "duckduckgo": ("DuckDuckGo", "https://duckduckgo.com/?q={q}"),
    "ddg": ("DuckDuckGo", "https://duckduckgo.com/?q={q}"),
    "bing": ("Bing", "https://www.bing.com/search?q={q}"),
    "brave": ("Brave Search", "https://search.brave.com/search?q={q}"),
    "startpage": ("Startpage", "https://www.startpage.com/sp/search?query={q}"),
    "ecosia": ("Ecosia", "https://www.ecosia.org/search?q={q}"),
    "yandex": ("Yandex", "https://yandex.com/search/?text={q}"),
    "perplexity": ("Perplexity", "https://www.perplexity.ai/search?q={q}"),
}


def resolve_engine(engine: str, sites: tuple[Site, ...] = SITES) -> Site:
    """``web_search.engine`` as a :class:`~skills.sites.Site` to search.

    Three spellings, most specific first: a name from the site table (so
    ``wikipedia``, ``youtube``, or one of your own ``skills.sites`` entries
    works, Cyrillic aliases included, and Wikipedia still switches to its
    Macedonian mirror for a Cyrillic query), one of the :data:`ENGINES`
    shorthands, or a full URL template containing ``{q}``.

    Anything else falls back to Google: a typo in config.yaml should cost you
    your preferred engine, not the ability to search at all.
    """
    wanted = (engine or "").strip()
    site = resolve_site(wanted, sites)
    if site is not None:
        return site
    key = wanted.lower()
    if key in ENGINES:
        label, template = ENGINES[key]
    elif "{q}" in wanted:
        host = (urlparse(wanted).hostname or "").lower()
        label = host[4:] if host.startswith("www.") else (host or "the web")
        template = wanted
    else:
        if wanted:
            logger.warning("unknown web_search.engine %r — using Google", engine)
        google = resolve_site("google", sites)
        if google is not None:
            return google
        label, template = "Google", "https://www.google.com/search?q={q}"
    return Site(key=key or "web", label=label, search=template,
                home=template.split("{q}")[0].rstrip("?&=/"))


def ddg_text_search(query: str, max_results: int = _MAX_RESULTS) -> list[dict[str, str]] | None:
    """Raw DuckDuckGo text results (None when unreachable/offline).

    The single ddgs entry point — this skill summarizes them for voice, and the
    companion API's /search/web serves them to the HUD. When the ddgs API or
    parser changes again, this is the only place to fix.
    """
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception:  # network down, rate limit, parser change, etc.
        return None


class WebSearchSkill(Skill):
    name = "web_search"
    description = "Search the web and summarize the results for a query."
    # Reached by MEANING for factual/current-info questions that trip no
    # search verb ("who composed the music for Gran Turismo", "how much is a
    # Raspberry Pi 5") — exactly the questions a small LLM answers from stale
    # memory or invents. semantic_from_text: the whole utterance is the query.
    semantic_from_text = True
    # NOT controls_pc, even though the command path opens a browser. That flag
    # would bar this skill from the semantic tier entirely (see
    # Router._semantic_safe), and the questions the semantic tier catches are
    # exactly the ones that DON'T open anything. The PC-control master switch
    # is honoured at execute time instead: with it off, a command falls back to
    # the spoken summary rather than opening a window.
    routing_phrases = [
        "who composed the music for that game",
        "how much does a raspberry pi cost right now",
        "when is the next spacex launch",
        "find out who won the match last night",
        "look into the best beginner soldering iron",
        "what year did the first arduino come out",
        "what's the current price of a spool of filament",
        "find out what's going on with the new graphics cards",
    ]

    #: Scope words that sit between the verb and the query and are NOT part of
    #: it. "search the internet for X" was searching for "internet for X", and
    #: "search up some info on X" for "info on X" — junk that was invisible
    #: while the answer was a spoken paragraph and is now in the address bar.
    #: They live here rather than in ``clean_query`` because "on"/"about" are
    #: not safe to strip generically: "search youtube for info ON cats" must
    #: keep its query.
    _SCOPE = (r"(?:up\s+)?"
              r"(?:(?:on\s+)?(?:the\s+)?(?:web|internet)\s+)?"
              r"(?:online\s+)?"
              r"(?:(?:some\s+|more\s+|any\s+)?(?:info(?:rmation)?|details?|stuff)\s+)?"
              r"(?:(?:for|about|on)\s+)?")

    patterns = [
        # The lookahead keeps the phrasal "look up TO (your heroes)", "look up
        # WHEN/AT", "google IS a great company", and "search your feelings/heart"
        # idioms off the fast path — search/google/look-up are otherwise verbs.
        re.compile(rf"\b(?:search|google|look\s+up)\s+{_SCOPE}"
                   r"(?!(?:to|when|at|is|are|was|were)\b|your\s+(?:feelings?|heart|soul)\b)"
                   r"(?P<q>.+)", re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+the\s+latest\s+(?:on|about)\s+(?P<q2>.+)", re.IGNORECASE),
        # MK: "барај рецепт за пица", "гугни цена на филамент". Registered last
        # in the registry, so the site/file/app skills have already had their
        # turn at these verbs — whatever reaches here really is a web search.
        re.compile(rf"\b(?:{mk.SEARCH}){mk.CLITICS}\s+(?:на\s+интернет\s+(?:за\s+)?)?(?P<q3>.+)",
                   re.IGNORECASE),
        re.compile(r"\bшто\s+(?:е\s+)?ново\s+(?:за|околу|со)\s+(?P<q4>.+)", re.IGNORECASE),
        # The same instruction to go and look, in words the verb list above
        # doesn't own. These reached the LLM and came back as a spoken
        # paragraph while "search up X" opened the browser — the same ask
        # answered two different ways, which reads as MEDO being arbitrary.
        # Narrow on purpose: "look into it/my inbox" is not a web search, and
        # "check the internet CONNECTION" is why "for" is mandatory below.
        re.compile(r"\blook\s+into\s+(?!my\b|our\b|it\b|this\b|that\b)(?P<q5>.+)",
                   re.IGNORECASE),
        re.compile(r"\bfind\s+(?:me\s+)?(?:some\s+|more\s+|any\s+)?"
                   r"(?:info(?:rmation)?|details?)\s+(?:on|about|for)\s+(?P<q6>.+)",
                   re.IGNORECASE),
        re.compile(r"\bcheck\s+(?:the\s+)?(?:web|internet|online)\s+for\s+(?P<q7>.+)",
                   re.IGNORECASE),
        # MK: "најди ми информации за X"
        re.compile(r"\b(?:најди|пронајди)(?:\s+(?:ми|ме))?\s+"
                   r"(?:информации|инфо|детали)\s+(?:за|околу|на)\s+(?P<q8>.+)",
                   re.IGNORECASE),
        # MK: "провери на интернет за X", "види на интернет за X"
        re.compile(r"\b(?:провери|види|погледни)\s+(?:на\s+)?интернет\s+"
                   r"(?:за\s+)?(?P<q9>.+)", re.IGNORECASE),
    ]

    #: Which capture group carried the query says what KIND of utterance it was.
    #: "search up X" / "барај X" are commands — go and look. "what's the latest
    #: on X" / "што е ново за X" are questions, and a question wants an answer,
    #: not a browser window. A group listed in neither is ignored entirely, so
    #: a pattern added later fails loudly in the tests rather than quietly
    #: searching for "".
    _COMMAND_GROUPS = ("q", "q3", "q5", "q6", "q7", "q8", "q9")
    _QUESTION_GROUPS = ("q2", "q4")

    def __init__(self, summarize: Summarize | None = None, *,
                 config: Any = None, opener=None, safety: Any = None,
                 extra_sites: dict[str, Any] | None = None) -> None:
        self._summarize = summarize
        # No config => summarize everything, exactly as before. The browser path
        # is opt-in at the wiring point (main.py passes settings.web_search), so
        # a bare WebSearchSkill() can never surprise a caller with a window.
        self._config = config
        self._safety = safety
        self._sites = load_sites(extra_sites)
        self._engine = resolve_engine(getattr(config, "engine", "google"), self._sites)
        if opener is None:
            import webbrowser

            opener = webbrowser.open
        self._opener = opener

    # --- routing ------------------------------------------------------------

    def _kind(self, request: SkillRequest, gd: dict[str, Any]) -> str:
        """How this skill was reached: "tool", "command", "question", "meaning"."""
        if request.context.get("via") == "tool":
            return "tool"
        if request.match is None:
            return "meaning"
        if any(gd.get(g) for g in self._COMMAND_GROUPS):
            return "command"
        return "question"

    def _wants_browser(self, kind: str) -> bool:
        """Should this utterance land in the browser instead of in speech?"""
        cfg = self._config
        if cfg is None or not getattr(cfg, "open_in_browser", False):
            return False
        # The HUD's PC CONTROL switch means "don't reach out and touch the
        # machine". Opening a window is touching it, so honour the switch here
        # the way the router honours it for every controls_pc skill.
        if self._safety is not None and \
                not getattr(self._safety, "pc_control_enabled", True):
            return False
        if getattr(cfg, "browser_scope", "commands") == "always":
            return True
        return kind == "command"

    def _search(self, query: str) -> list[dict[str, str]] | None:
        limit = int(getattr(self._config, "max_results", 0) or _MAX_RESULTS)
        return ddg_text_search(query, limit)

    # --- execution ----------------------------------------------------------

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        speak_mk = mk.is_cyrillic(request.text)
        kind = self._kind(request, gd)
        raw = request.args.get("query") or ""
        if not raw:
            # Iterated rather than or-chained so adding a pattern can't leave
            # its group unread — the group lists are the single source of truth.
            for group in (*self._COMMAND_GROUPS, *self._QUESTION_GROUPS):
                if gd.get(group):
                    raw = gd[group]
                    break
        if not raw and not request.match and not request.args:
            # Reached by MEANING (semantic tier): no regex groups and no tool
            # args, so the whole utterance IS the query.
            raw = request.text
        # "search up for the Mazda RX-7" captures "up for the Mazda RX-7" — the
        # particle belongs to the phrasal verb, and searching for it measurably
        # worsens the results. Same cleaner the site searches use.
        query = clean_query(raw)
        if not query:
            return SkillResult(
                "Што да пребарам?" if speak_mk else "What should I search for?",
                success=False)

        if self._wants_browser(kind):
            url = build_url(self._engine, query)
            ok = await open_with(self._opener, url)
            if ok is False:
                # No browser, or a blocked host. Never claim a page we didn't
                # open — fall through and answer out loud instead.
                logger.warning("browser search for %r didn't open — summarizing", query)
            elif kind != "tool":
                return SkillResult(
                    f"Пребарувам {self._engine.label} за {query}." if speak_mk
                    else f"Searching {self._engine.label} for {query}.",
                    data={"url": url, "query": query,
                          "engine": self._engine.key, "opened": True})
            # kind == "tool" under scope "always": the window is up, but the
            # model still needs the snippets or it will answer from nothing.

        results = await asyncio.to_thread(self._search, query)
        if results is None:
            return SkillResult(_OFFLINE_MK if speak_mk else _OFFLINE, success=False)
        if not results:
            return SkillResult(
                f"Не најдов ништо за {query}." if speak_mk
                else f"I couldn't find anything about {query}.", success=False)

        block = "\n".join(
            f"- {r.get('title', '')}: {r.get('body', '')}" for r in results
        )

        # Called by the model as a tool: hand back raw results for it to summarize.
        if kind == "tool" or self._summarize is None:
            return SkillResult(block, data={"query": query, "results": results})

        summary = await self._summarize(query, block)
        return SkillResult(summary, data={"query": query, "results": results})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "the search query"}
                    },
                    "required": ["query"],
                },
            },
        }
