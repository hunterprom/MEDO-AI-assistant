"""Macedonian spoken-command vocabulary, shared by every skill.

MEDO is bilingual (EN + MK), but the fast path is regex — so a skill that only
knows "open" is deaf to "отвори". faster-whisper hands the router Macedonian in
Cyrillic, which means every English verb pattern needs its Macedonian twin.

Those verb lists live here rather than copy-pasted into a dozen regexes: adding
a synonym once fixes apps, websites, site search, files and web search at the
same time. Each constant is a bare alternation meant to be interpolated into an
f-string pattern::

    re.compile(rf"\\b(?:{mk.OPEN}){mk.CLITICS}\\s+(?P<app>chrome|spotify)\\b")

Longer forms come first inside each alternation — Python's ``|`` takes the first
branch that matches, so ``пребарај`` must be tried before ``барај`` or the
prefix would be left behind as part of the query.
"""

from __future__ import annotations

import re

#: Any Cyrillic letter. The cheap "is this Macedonian?" test the skills use to
#: pick which language to answer in (Latin script => English).
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")

#: open / launch / start / turn on — "отвори хром", "стартувај спотифај".
OPEN = r"отворете|отвори|стартувај|стартуј|стартирај|вклучи|подигни|активирај|пушти"

#: close / quit / turn off — "затвори хром", "исклучи спотифај".
CLOSE = r"затворете|затвори|исклучи|изгасни|изгаси|угаси|прекини"

#: search — "барај …", "пребарај …", "гугни …".
SEARCH = r"пребарувај|пребарај|побарај|барај|изгугли|гугни"

#: check — "провери во пошта …". Deliberately *not* part of :data:`SEARCH`:
#: "провери го времето" is a weather question, not a web search, so only the
#: patterns that name an explicit target opt into this verb.
CHECK = r"проверете|провери"

#: find / locate — "најди …". Separate from SEARCH because file skills want
#: "најди" but not "гугни".
FIND = r"пронајди|изнајди|најди"

#: Either of the two above — what most search-shaped patterns actually want.
SEARCH_OR_FIND = f"{SEARCH}|{FIND}"

#: go to / navigate to — "оди на јутјуб".
GO_TO = r"оди\s+на|оди\s+до|појди\s+на|појди\s+до|врати\s+се\s+на"

#: show me — "покажи ми …".
SHOW = r"покажи|прикажи"

#: Unstressed pronoun clitics that pile up after a Macedonian imperative
#: ("отвори ми го хром"). Optional and repeatable, so patterns can just append
#: this after the verb and forget about them.
CLITICS = r"(?:\s+(?:ми|ме|го|ја|ги|му|им|ни|ти|се))*"

#: Prepositions that introduce a target ("на јутјуб", "во прелистувач").
ON = r"на|во|од|кај|преку"

#: browser, spoken several ways — used by the "search … in the browser" patterns.
BROWSER = r"прелистувачот|прелистувач|пребарувачот|пребарувач|браузерот|браузер|интернет"


def is_cyrillic(text: str) -> bool:
    """True when ``text`` contains Cyrillic — i.e. the user spoke Macedonian.

    Skills use this to answer in the language they were asked in, the same
    bilingual rule the briefing and bench skills already follow.
    """
    return CYRILLIC_RE.search(text) is not None
