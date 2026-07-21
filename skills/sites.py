"""Search *inside* a site — "search drone motors on YouTube", "барај мачки на јутјуб".

MEDO could already open a site (``web_open``) and search the web server-side
(``websearch``), but not search within a specific site, which is what people
actually ask for: YouTube for a video, Gmail for an email, Steam for a game,
Reddit for an opinion. Each site gets its native search URL opened in the
browser — no scraping, no API keys, and the results land where the user can
click them.

This module owns the site table because two skills need it: this one and
``web_open`` (so "отвори јутјуб" knows where YouTube lives). Every entry
carries the spoken aliases MEDO is likely to hear in **both** languages —
Whisper writes "YouTube" as "јутјуб" when you're speaking Macedonian, so the
Cyrillic forms are data, not an afterthought.

Add your own without touching code, in config.yaml under ``skills.sites``::

    skills:
      sites:
        pazar3:
          label:   "Pazar3"
          search:  "https://www.pazar3.mk/oglasi?q={q}"
          home:    "https://www.pazar3.mk"
          aliases: ["пазар3", "пазар"]

This is an act-on-the-PC skill (``controls_pc``): the HUD's PC CONTROL switch
gates it, like every other skill that reaches out and touches the machine.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, quote_plus

from core import mk
from skills.base import Skill, SkillRequest, SkillResult


@dataclass(frozen=True)
class Site:
    """One searchable site: where its search lives and what it's called aloud."""

    key: str
    label: str                        # spoken back to the user ("YouTube")
    search: str                       # URL template; "{q}" gets the query
    home: str                         # opened when there's no query
    aliases: tuple[str, ...] = ()     # extra spoken forms, EN + Cyrillic
    #: Optional localized template used when the *query* is Cyrillic — the
    #: Macedonian Wikipedia is a far better answer for a Macedonian question.
    search_mk: str | None = None

    def spoken_names(self) -> tuple[str, ...]:
        return (self.key, self.label.lower(), *self.aliases)


#: Curated defaults. Only sites whose search URL is stable and public — a
#: guessed URL that 404s is worse than not offering the site at all.
SITES: tuple[Site, ...] = (
    # --- video / audio -------------------------------------------------------
    Site("youtube", "YouTube",
         "https://www.youtube.com/results?search_query={q}",
         "https://www.youtube.com",
         ("yt", "you tube", "јутјуб", "јутуб", "ју туб", "јутјубе")),
    Site("spotify", "Spotify",
         "https://open.spotify.com/search/{q}",
         "https://open.spotify.com",
         ("спотифај", "спотифи", "спотифај музика")),
    Site("twitch", "Twitch",
         "https://www.twitch.tv/search?term={q}",
         "https://www.twitch.tv",
         ("твич",)),
    Site("netflix", "Netflix",
         "https://www.netflix.com/search?q={q}",
         "https://www.netflix.com",
         ("нетфликс",)),
    Site("imdb", "IMDb",
         "https://www.imdb.com/find/?q={q}",
         "https://www.imdb.com",
         ("имдб",)),
    # --- search / reference --------------------------------------------------
    Site("google", "Google",
         "https://www.google.com/search?q={q}",
         "https://www.google.com",
         ("гугл", "гоогл")),
    Site("images", "Google Images",
         "https://www.google.com/search?tbm=isch&q={q}",
         "https://images.google.com",
         ("google images", "гугл слики", "слики", "фотографии")),
    Site("maps", "Google Maps",
         "https://www.google.com/maps/search/{q}",
         "https://www.google.com/maps",
         ("google maps", "гугл мапи", "мапи", "карти", "мапа")),
    Site("translate", "Google Translate",
         "https://translate.google.com/?sl=auto&tl=en&op=translate&text={q}",
         "https://translate.google.com",
         ("google translate", "гугл транслејт", "преведувач", "транслејт")),
    Site("wikipedia", "Wikipedia",
         "https://en.wikipedia.org/w/index.php?search={q}",
         "https://en.wikipedia.org",
         ("wiki", "википедија", "википедиа", "вики"),
         search_mk="https://mk.wikipedia.org/w/index.php?search={q}"),
    Site("arxiv", "arXiv",
         "https://arxiv.org/search/?searchtype=all&query={q}",
         "https://arxiv.org",
         ("арксив", "архив на трудови")),
    # --- mail / drive --------------------------------------------------------
    Site("gmail", "Gmail",
         "https://mail.google.com/mail/u/0/#search/{q}",
         "https://mail.google.com",
         ("email", "e-mail", "mail", "my email", "my mail", "inbox",
          "гмаил", "мејл", "е-мејл", "емаил", "пошта", "мејлот", "поштата")),
    Site("drive", "Google Drive",
         "https://drive.google.com/drive/search?q={q}",
         "https://drive.google.com",
         ("google drive", "гугл драјв", "драјв")),
    # --- social / forums -----------------------------------------------------
    Site("reddit", "Reddit",
         "https://www.reddit.com/search/?q={q}",
         "https://www.reddit.com",
         ("редит", "реддит")),
    Site("x", "X",
         "https://x.com/search?q={q}",
         "https://x.com",
         ("twitter", "x.com", "твитер", "твитор")),
    Site("instagram", "Instagram",
         "https://www.instagram.com/explore/search/keyword/?q={q}",
         "https://www.instagram.com",
         ("insta", "инстаграм", "инста")),
    Site("facebook", "Facebook",
         "https://www.facebook.com/search/top?q={q}",
         "https://www.facebook.com",
         ("fb", "фејсбук", "фејзбук", "фб")),
    Site("tiktok", "TikTok",
         "https://www.tiktok.com/search?q={q}",
         "https://www.tiktok.com",
         ("тикток", "тик ток")),
    Site("linkedin", "LinkedIn",
         "https://www.linkedin.com/search/results/all/?keywords={q}",
         "https://www.linkedin.com",
         ("линкдин", "линкедин")),
    Site("pinterest", "Pinterest",
         "https://www.pinterest.com/search/pins/?q={q}",
         "https://www.pinterest.com",
         ("пинтерест",)),
    Site("hackernews", "Hacker News",
         "https://hn.algolia.com/?q={q}",
         "https://news.ycombinator.com",
         ("hacker news", "hn", "хакер њуз")),
    # --- shopping ------------------------------------------------------------
    Site("amazon", "Amazon",
         "https://www.amazon.com/s?k={q}",
         "https://www.amazon.com",
         ("амазон",)),
    Site("ebay", "eBay",
         "https://www.ebay.com/sch/i.html?_nkw={q}",
         "https://www.ebay.com",
         ("ибеј", "ебеј", "ебај")),
    Site("aliexpress", "AliExpress",
         "https://www.aliexpress.com/wholesale?SearchText={q}",
         "https://www.aliexpress.com",
         ("ali express", "алиекспрес", "али експрес")),
    Site("etsy", "Etsy",
         "https://www.etsy.com/search?q={q}",
         "https://www.etsy.com",
         ("етси",)),
    # --- games ---------------------------------------------------------------
    Site("steam", "Steam",
         "https://store.steampowered.com/search/?term={q}",
         "https://store.steampowered.com",
         ("стим", "стеам", "стим стор")),
    # --- code / making (this is a workshop assistant) ------------------------
    Site("github", "GitHub",
         "https://github.com/search?q={q}",
         "https://github.com",
         ("гитхаб", "гит хаб", "гитхуб")),
    Site("stackoverflow", "Stack Overflow",
         "https://stackoverflow.com/search?q={q}",
         "https://stackoverflow.com",
         ("stack overflow", "стак оверфлоу", "стековерфлоу")),
    Site("pypi", "PyPI",
         "https://pypi.org/search/?q={q}",
         "https://pypi.org",
         ("pip", "python packages", "пајпиај")),
    Site("npm", "npm",
         "https://www.npmjs.com/search?q={q}",
         "https://www.npmjs.com",
         ("нпм",)),
    Site("dockerhub", "Docker Hub",
         "https://hub.docker.com/search?q={q}",
         "https://hub.docker.com",
         ("docker hub", "docker", "докер")),
    Site("thingiverse", "Thingiverse",
         "https://www.thingiverse.com/search?q={q}",
         "https://www.thingiverse.com",
         ("тингиверс", "тхингиверс")),
    Site("printables", "Printables",
         "https://www.printables.com/search/all?q={q}",
         "https://www.printables.com",
         ("принтаблес",)),
    Site("grabcad", "GrabCAD",
         "https://grabcad.com/library?query={q}",
         "https://grabcad.com",
         ("граб кад", "грабкад")),
    Site("tinkercad", "Tinkercad",
         "https://www.tinkercad.com/search?q={q}",
         # Bare host on purpose: "open tinkercad" has resolved here since the
         # browser-control work and the HUD/tests expect exactly this.
         "https://tinkercad.com",
         ("тинкеркад", "тинкер кад")),
)


def load_sites(extra: dict[str, Any] | None = None) -> tuple[Site, ...]:
    """:data:`SITES` plus any ``skills.sites`` entries from config.yaml.

    A config entry reusing a built-in key replaces it, so a user can point
    ``wikipedia`` at their own mirror. Malformed entries (no ``search``) are
    skipped rather than crashing startup — one bad YAML block should never
    cost you the other thirty sites.
    """
    sites = {s.key: s for s in SITES}
    for key, spec in (extra or {}).items():
        if not isinstance(spec, dict) or not spec.get("search"):
            continue
        search = str(spec["search"])
        sites[key] = Site(
            key=key,
            label=str(spec.get("label", key.title())),
            search=search,
            home=str(spec.get("home") or search.split("{q}")[0].rstrip("/?&=")),
            aliases=tuple(str(a).lower() for a in spec.get("aliases", ())),
            search_mk=spec.get("search_mk"),
        )
    return tuple(sites.values())


def _norm(spoken: str) -> str:
    """Spoken site name -> lookup key ("  You  Tube. " -> "you tube")."""
    return re.sub(r"\s+", " ", spoken.strip().strip(".,!?;:\"'")).lower()


def resolve_site(spoken: str, sites: tuple[Site, ...] = SITES) -> Site | None:
    """Find the site behind a spoken name, in either language. None if unknown."""
    wanted = _norm(spoken)
    if not wanted:
        return None
    for site in sites:
        if wanted in site.spoken_names():
            return site
    return None


def alias_alternation(sites: tuple[Site, ...] = SITES) -> str:
    """Regex alternation over every spoken name, longest first.

    Longest-first matters: without it "google" would swallow the "google" in
    "google maps" and the user's map search would open a plain web search.
    """
    names = {n for site in sites for n in site.spoken_names()}
    ordered = sorted(names, key=len, reverse=True)
    return "|".join(re.escape(n) for n in ordered)


def build_url(site: Site, query: str) -> str:
    """The site's search URL for ``query``, correctly escaped.

    Path-style templates (Spotify, Maps) need ``quote``; query-string ones need
    ``quote_plus`` so spaces become ``+`` rather than ``%20``. Which one applies
    is decided by whether the placeholder sits after a "?".
    """
    template = site.search
    if site.search_mk and mk.is_cyrillic(query):
        template = site.search_mk
    head = template.split("{q}")[0]
    escaped = quote_plus(query) if "?" in head else quote(query)
    return template.replace("{q}", escaped)


async def open_with(opener, url: str):
    """Call an opener that may be sync or async, and return what it returned.

    Two kinds live behind this: ``webbrowser.open`` (blocking, so it goes to a
    thread) and the controlled browser's navigate coroutine (already async).
    Awaiting a coroutine function inside ``to_thread`` would silently do
    nothing and hand back a truthy coroutine object — i.e. MEDO would claim it
    opened a page it never touched.
    """
    if inspect.iscoroutinefunction(opener):
        return await opener(url)
    result = await asyncio.to_thread(opener, url)
    return await result if inspect.isawaitable(result) else result


#: Trailing politeness that would otherwise become part of the search query.
_TRAILING = re.compile(
    r"\s*(?:please|for me|te molam|те молам|ве молам|молам|ајде)\s*$", re.IGNORECASE
)


def clean_query(query: str) -> str:
    """Strip spoken filler and punctuation from a captured query."""
    query = query.strip().strip(".,!?;:\"' ")
    previous = None
    while previous != query:  # "…, please" then "…, for me" — peel both
        previous = query
        query = _TRAILING.sub("", query).strip().strip(".,!?;:\"' ")
    return query


class SiteSearchSkill(Skill):
    """Open a site's own search results for a spoken query."""

    name = "site_search"
    controls_pc = True
    description = (
        "Search inside a specific website and open the results in the browser: "
        "YouTube, Gmail, Reddit, Steam, GitHub, Amazon, Wikipedia, Google Maps "
        "and more. Use whenever the user names a site to search on."
    )

    def __init__(self, opener=None, extra_sites: dict[str, Any] | None = None) -> None:
        # Injectable for tests; the default opens the OS default browser.
        if opener is None:
            import webbrowser

            opener = webbrowser.open
        self._opener = opener
        self._sites = load_sites(extra_sites)
        alt = alias_alternation(self._sites)
        verbs = r"search|look\s+up|look\s+for|find|browse|check"
        self.patterns = [
            # "search drone motors on youtube", "find sniper elite on steam"
            re.compile(rf"\b(?:{verbs})\s+(?:for\s+)?(?P<q>.+?)\s+"
                       rf"(?:on|in|at|through)\s+(?:the\s+|my\s+)?(?P<site>{alt})\b",
                       re.IGNORECASE),
            # "search youtube for drone motors", "check my email for the invoice"
            re.compile(rf"\b(?:{verbs})\s+(?:on\s+|in\s+|my\s+)?(?P<site2>{alt})\s+"
                       rf"(?:for\s+)?(?P<q2>.+)", re.IGNORECASE),
            # "youtube search drone motors"
            re.compile(rf"\b(?P<site3>{alt})\s+search\s+(?:for\s+)?(?P<q3>.+)",
                       re.IGNORECASE),
            # MK: "барај мачки на јутјуб", "најди ми игра на стим"
            re.compile(rf"\b(?:{mk.SEARCH_OR_FIND}|{mk.CHECK}){mk.CLITICS}\s+(?P<qm>.+?)\s+"
                       rf"(?:{mk.ON})\s+(?P<sitem>{alt})\b", re.IGNORECASE),
            # MK: "барај на јутјуб мачки", "провери во пошта за сметката"
            re.compile(rf"\b(?:{mk.SEARCH_OR_FIND}|{mk.CHECK}){mk.CLITICS}\s+(?:{mk.ON})\s+"
                       rf"(?P<sitem2>{alt})\s+(?:за\s+)?(?P<qm2>.+)", re.IGNORECASE),
        ]

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        spoken = (request.args.get("site") or gd.get("site") or gd.get("site2")
                  or gd.get("site3") or gd.get("sitem") or gd.get("sitem2") or "")
        query = clean_query(
            request.args.get("query") or gd.get("q") or gd.get("q2")
            or gd.get("q3") or gd.get("qm") or gd.get("qm2") or ""
        )
        # Answer in the language we were asked in — the same bilingual rule the
        # briefing and bench skills follow.
        speak_mk = mk.is_cyrillic(request.text)

        site = resolve_site(spoken, self._sites)
        if site is None:
            return SkillResult(
                f"Не знам за сајтот {spoken}." if speak_mk
                else f"I don't know a site called {spoken}.", success=False)

        if query:
            url = build_url(site, query)
            speech = (f"Пребарувам {site.label} за {query}." if speak_mk
                      else f"Searching {site.label} for {query}.")
        else:  # named the site but not what to look for — just open it
            url = site.home
            speech = (f"Отворам {site.label}." if speak_mk
                      else f"Opening {site.label}.")

        ok = await open_with(self._opener, url)
        if ok is False:  # webbrowser.open returns False when no browser exists
            return SkillResult(
                "Не најдов прелистувач да отворам." if speak_mk
                else "I couldn't find a browser to open.", success=False)
        return SkillResult(speech, data={"url": url, "site": site.key, "query": query})

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "site": {
                            "type": "string",
                            "enum": [s.key for s in self._sites],
                            "description": "Which site to search.",
                        },
                        "query": {
                            "type": "string",
                            "description": "What to search for. Omit to just "
                                           "open the site.",
                        },
                    },
                    "required": ["site"],
                },
            },
        }
