"""Open websites in the user's default browser ("open tinkercad.com").

The gap this closes: MEDO could *launch* known apps and *search* the web
server-side, but had no way to actually open a site in the browser — the LLM
would truthfully answer "I can't open a browser". Anything that looks like a
domain opens directly; anything else becomes a browser search. Uses the
stdlib ``webbrowser`` module (default browser, new tab).

Bare names resolve through the shared site table in :mod:`skills.sites`, so
"open youtube" and "отвори јутјуб" both land on YouTube without this module
keeping its own copy of the URLs.

This is an act-on-the-PC skill (``controls_pc``), so the HUD's PC CONTROL
switch gates it.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import quote_plus

from core import mk
from skills.base import Skill, SkillRequest, SkillResult
from skills.sites import SITES, alias_alternation, resolve_site

#: something.tld[/path...] — enough to tell a site from a search phrase.
_DOMAIN_RE = re.compile(r"^(?:https?://)?[\w-]+(?:\.[\w-]+)+(?:/\S*)?$", re.IGNORECASE)

SEARCH_URL = "https://duckduckgo.com/?q={query}"

#: Bare names that open as <name>.com without saying "dot com", on top of the
#: sites in :mod:`skills.sites`. A CURATED list on purpose: a generic
#: "open <word>" rule would steal app launches ("open notepad") and folder
#: commands ("open downloads"). The LLM path otherwise tends to *claim* it
#: opened the site without calling any tool.
SITE_SHORTCUTS = (
    "chatgpt", "claude", "canva", "figma", "discord", "whatsapp",
    "twitch", "gmail", "wikipedia",
)

#: Spoken separators -> real ones. Whisper hears "tinkercad dot com" as words.
_SPOKEN_DOT = re.compile(r"\s+(?:dot|точка)\s+", re.IGNORECASE)


def to_url(target: str) -> str:
    """A spoken target -> the URL to open (site directly, else a web search)."""
    target = target.strip().strip(".?!,\"' ")
    if _DOMAIN_RE.match(target):
        return target if target.lower().startswith(("http://", "https://")) \
            else f"https://{target}"
    return SEARCH_URL.format(query=quote_plus(target))


def shortcut_url(name: str) -> str | None:
    """Bare spoken name -> its home page, or None when it isn't a known site.

    Checks the shared site table first (which carries the Cyrillic aliases),
    then the extra curated names above.
    """
    site = resolve_site(name)
    if site is not None:
        return site.home
    bare = name.strip().lower()
    return f"https://{bare}.com" if bare in SITE_SHORTCUTS else None


class OpenWebsiteSkill(Skill):
    name = "open_website"
    description = (
        "Open a website in the user's browser (e.g. tinkercad.com), or run a "
        "web search in the browser. Use for any 'open/go to <site>' request."
    )
    controls_pc = True

    def __init__(self, opener=None) -> None:
        # Injectable for tests; the default opens the OS default browser.
        if opener is None:
            import webbrowser

            opener = webbrowser.open
        self._opener = opener
        # Known bare names: every alias in the site table plus the extras above.
        shortcuts = "|".join((alias_alternation(SITES),
                              *(re.escape(s) for s in SITE_SHORTCUTS)))
        domain = r"(?:https?://)?[\w-]+(?:\.[\w-]+)+(?:/\S*)?"
        spoken = r"[\w-]+(?:\s+(?:dot|точка)\s+[\w-]+)+"
        en_open = r"open|go\s+to|visit|pull\s+up|bring\s+up"
        mk_open = rf"{mk.OPEN}|{mk.GO_TO}"
        self.patterns = [
            # "open tinkercad.com", "go to docs.python.org/3" — needs a dot.
            re.compile(rf"\b(?:{en_open})\s+(?:the\s+)?(?P<url>{domain})",
                       re.IGNORECASE),
            # Voice: Whisper transcribes "tinkercad.com" as "tinkercad dot com" —
            # words, not a dot — which used to fall through to the LLM.
            re.compile(rf"\b(?:{en_open})\s+(?:the\s+)?(?P<spoken>{spoken})\b",
                       re.IGNORECASE),
            # "open the website tinkercad" — explicit keyword, no dot needed.
            re.compile(r"\bopen\s+(?:the\s+)?(?:web\s?site|web\s?page)\s+(?P<name>.+)$",
                       re.IGNORECASE),
            # "open tinkercad" — curated bare names only.
            re.compile(rf"\b(?:{en_open})\s+(?:the\s+)?(?P<shortcut>{shortcuts})\b",
                       re.IGNORECASE),
            # "search for arduino sensors in the browser".
            re.compile(r"\bsearch\s+(?:for\s+)?(?P<query>.+?)\s+in\s+(?:the\s+|my\s+)?browser\b",
                       re.IGNORECASE),
            # MK: "отвори тинкеркад.com", "оди на youtube.com"
            re.compile(rf"\b(?:{mk_open}){mk.CLITICS}\s+(?P<url_mk>{domain})",
                       re.IGNORECASE),
            # MK: "отвори тинкеркад точка ком"
            re.compile(rf"\b(?:{mk_open}){mk.CLITICS}\s+(?P<spoken_mk>{spoken})\b",
                       re.IGNORECASE),
            # MK: "отвори јутјуб", "оди на редит"
            re.compile(rf"\b(?:{mk_open}){mk.CLITICS}\s+(?P<shortcut_mk>{shortcuts})\b",
                       re.IGNORECASE),
            # MK: "отвори ја страната тинкеркад"
            re.compile(rf"\b(?:{mk.OPEN}){mk.CLITICS}\s+(?:страна|страната|сајт|сајтот|"
                       rf"веб\s+страна)\s+(?P<name_mk>.+)$", re.IGNORECASE),
            # MK: "барај ардуино сензори во прелистувач"
            re.compile(rf"\b(?:{mk.SEARCH})\s+(?P<query_mk>.+?)\s+(?:{mk.ON})\s+"
                       rf"(?:{mk.BROWSER})\b", re.IGNORECASE),
        ]

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        speak_mk = mk.is_cyrillic(request.text)
        shortcut = gd.get("shortcut") or gd.get("shortcut_mk")
        target = (request.args.get("url") or request.args.get("query")
                  or gd.get("url") or gd.get("url_mk")
                  or gd.get("spoken") or gd.get("spoken_mk")
                  or shortcut or gd.get("name") or gd.get("name_mk")
                  or gd.get("query") or gd.get("query_mk") or "").strip()
        if not target:
            return SkillResult(
                "Која страна да отворам?" if speak_mk
                else "Which website should I open?", success=False)

        # Spoken separators back to real ones ("tinkercad dot com" -> ".").
        target = _SPOKEN_DOT.sub(".", target)

        # A known bare name ("youtube", "јутјуб") beats a web search for it.
        site = resolve_site(target)
        if site is None and "." in target and mk.is_cyrillic(target):
            # Macedonian speech turns "tinkercad dot com" into "тинкеркад.ком" —
            # a Cyrillic pseudo-domain nobody owns. When the label names a site
            # we know, that's plainly what was meant.
            site = resolve_site(target.split(".")[0])
        if site is not None:
            url, shown = site.home, site.label
        else:
            url = shortcut_url(target) or to_url(target)
            shown = re.sub(r"^https?://", "", url).split("/?q=")[0]

        ok = await asyncio.to_thread(self._opener, url)
        if ok is False:  # webbrowser.open returns False when no browser exists
            return SkillResult(
                "Не најдов прелистувач да отворам." if speak_mk
                else "I couldn't find a browser to open.", success=False)
        return SkillResult(
            f"Отворам {shown}." if speak_mk else f"Opening {shown} in your browser.",
            data={"url": url})

    def tool_schema(self) -> dict[str, Any]:
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
                            "description": "Site to open (tinkercad.com) or a "
                                           "search phrase for the browser.",
                        }
                    },
                    "required": ["url"],
                },
            },
        }
