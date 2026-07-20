"""Open websites in the user's default browser ("open tinkercad.com").

The gap this closes: MEDO could *launch* known apps and *search* the web
server-side, but had no way to actually open a site in the browser — the LLM
would truthfully answer "I can't open a browser". Anything that looks like a
domain opens directly; anything else becomes a browser search. Uses the
stdlib ``webbrowser`` module (default browser, new tab).

This is an act-on-the-PC skill (``controls_pc``), so the HUD's PC CONTROL
switch gates it.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import quote_plus

from skills.base import Skill, SkillRequest, SkillResult

#: something.tld[/path...] — enough to tell a site from a search phrase.
_DOMAIN_RE = re.compile(r"^(?:https?://)?[\w-]+(?:\.[\w-]+)+(?:/\S*)?$", re.IGNORECASE)

SEARCH_URL = "https://duckduckgo.com/?q={query}"


def to_url(target: str) -> str:
    """A spoken target -> the URL to open (site directly, else a web search)."""
    target = target.strip().strip(".?!,\"' ")
    if _DOMAIN_RE.match(target):
        return target if target.lower().startswith(("http://", "https://")) \
            else f"https://{target}"
    return SEARCH_URL.format(query=quote_plus(target))


class OpenWebsiteSkill(Skill):
    name = "open_website"
    description = (
        "Open a website in the user's browser (e.g. tinkercad.com), or run a "
        "web search in the browser. Use for any 'open/go to <site>' request."
    )
    controls_pc = True

    patterns = [
        # "open tinkercad.com", "go to docs.python.org/3" — needs a dot.
        re.compile(r"\b(?:open|go\s+to|visit)\s+(?:the\s+)?"
                   r"(?P<url>(?:https?://)?[\w-]+(?:\.[\w-]+)+(?:/\S*)?)",
                   re.IGNORECASE),
        # "open the website tinkercad" — explicit keyword, no dot needed.
        re.compile(r"\bopen\s+(?:the\s+)?(?:web\s?site|web\s?page)\s+(?P<name>.+)$",
                   re.IGNORECASE),
        # "search for arduino sensors in the browser".
        re.compile(r"\bsearch\s+(?:for\s+)?(?P<query>.+?)\s+in\s+(?:the\s+|my\s+)?browser\b",
                   re.IGNORECASE),
    ]

    def __init__(self, opener=None) -> None:
        # Injectable for tests; the default opens the OS default browser.
        if opener is None:
            import webbrowser

            opener = webbrowser.open
        self._opener = opener

    async def execute(self, request: SkillRequest) -> SkillResult:
        gd = request.match.groupdict() if request.match else {}
        target = (request.args.get("url") or request.args.get("query")
                  or gd.get("url") or gd.get("name") or gd.get("query") or "").strip()
        if not target:
            return SkillResult("Which website should I open?", success=False)
        url = to_url(target)
        ok = await asyncio.to_thread(self._opener, url)
        if ok is False:  # webbrowser.open returns False when no browser exists
            return SkillResult("I couldn't find a browser to open.", success=False)
        shown = re.sub(r"^https?://", "", url).split("/?q=")[0]
        return SkillResult(f"Opening {shown} in your browser.", data={"url": url})

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
