"""Did the model actually answer in the language we asked for?

Telling a small local model "reply in Macedonian" is a request, not a
guarantee. Left unchecked it produces exactly what a live transcript showed::

    YOU   отвори стима
    MEDO  да, можам да го instaliram.
    YOU   Направи апликација.
    MEDO  Морам да извежdam " winget" и тогаш можеш да ме направи апликација…
    YOU   Збори на македонски, зошто збориш на англиски.
    MEDO  I can help with that! Can you please provide more context…

Three different failures, one root cause: nothing downstream ever looked at the
reply. This module looks. It is deliberately *script*-based rather than a real
language ID — script is a coarse signal, but it is offline, instant, has no
model to load, and it catches the failures that actually happen: a reply in the
wrong alphabet, and Latin fragments welded into Cyrillic words ("извежdam").

Two things keep the coarseness honest:

* Proper nouns are exempt. "Отвори го Steam" is correct Macedonian, and so is a
  reply naming ``winget`` once — a capitalised, all-caps, digit-bearing or very
  short Latin token never counts against the reply.
* A ratio, not a count. One foreign token in a long sentence is normal speech;
  one in four is a broken reply.

The caller (``core/router.py``) uses this to retry ONCE with a firmer
instruction and keep whichever attempt scores better, so a false positive costs
one extra round and never a worse answer.
"""

from __future__ import annotations

import re
import unicodedata

#: Which script(s) a language is legitimately written in. Japanese mixes kana
#: and han, Korean mixes hangul and han — so this is a set per language, not a
#: single value. Everything absent from here is treated as Latin.
_SCRIPTS: dict[str, frozenset[str]] = {
    "mk": frozenset({"CYRILLIC"}),
    "ru": frozenset({"CYRILLIC"}),
    "el": frozenset({"GREEK"}),
    "zh": frozenset({"CJK"}),
    "ja": frozenset({"CJK", "KANA"}),
    "ko": frozenset({"HANGUL", "CJK"}),
    "hi": frozenset({"DEVANAGARI"}),
}
_LATIN = frozenset({"LATIN"})

#: Ratio of wrong-script words at which a reply stops being "a sentence with a
#: brand name in it" and starts being a broken reply. "да, можам да го
#: instaliram" is 1 word in 5 = 0.20; a correct Macedonian sentence mentioning
#: winget once sits far below it.
_WRONG_WORD_RATIO = 0.20

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _script_of_char(char: str) -> str | None:
    """The script family of one letter, or None if it isn't a letter."""
    if not char.isalpha():
        return None
    try:
        name = unicodedata.name(char)
    except ValueError:            # unnamed codepoint — not something we judge
        return None
    # unicodedata names letters "CYRILLIC SMALL LETTER A", "HIRAGANA LETTER A",
    # "CJK UNIFIED IDEOGRAPH-4E00" — the first word is the script, except for
    # the two kana syllabaries and CJK, which we fold together.
    head = name.split(" ", 1)[0]
    if head in ("HIRAGANA", "KATAKANA"):
        return "KANA"
    if head == "CJK":
        return "CJK"
    if head in ("LATIN", "CYRILLIC", "GREEK", "HANGUL", "DEVANAGARI", "ARABIC",
                "HEBREW", "THAI", "ARMENIAN", "GEORGIAN"):
        return head
    return None


def expected_scripts(lang: str | None) -> frozenset[str]:
    """The script(s) a reply in ``lang`` should be written in."""
    code = (lang or "").strip().lower()[:2]
    return _SCRIPTS.get(code, _LATIN)


def _scripts_in(word: str) -> set[str]:
    return {s for s in (_script_of_char(c) for c in word) if s}


def _is_proper_noun_shaped(word: str) -> bool:
    """Tokens that stay in their own alphabet inside any language.

    Brand and app names ("Steam", "YouTube"), acronyms ("USB", "LED"), anything
    carrying a digit ("ESP32"), and one- or two-letter fragments. Judging these
    as foreign would flag correct sentences.
    """
    if len(word) <= 2:
        return True
    if any(c.isdigit() for c in word):
        return True
    return word[:1].isupper()


def mixed_script_words(text: str) -> list[str]:
    """Words built from two alphabets at once — "извежdam", "следENE".

    Never legitimate in any language, so this alone condemns a reply.
    """
    return [w for w in _WORD_RE.findall(text or "") if len(_scripts_in(w)) > 1]


def off_language(text: str, lang: str | None) -> bool:
    """True when ``text`` does not read as a reply written in ``lang``.

    Conservative by construction: a reply is only called wrong when it welds
    two alphabets inside a word, or when enough of its ordinary (non-proper-noun)
    words sit in the wrong alphabet that it can't be read aloud as ``lang``.
    """
    text = (text or "").strip()
    if not text or not lang:
        return False
    if mixed_script_words(text):
        return True

    wanted = expected_scripts(lang)
    words = _WORD_RE.findall(text)
    # Only ordinary words vote. Proper nouns keep their own script in every
    # language, so counting them would flag "Отвори го Steam".
    ordinary = [w for w in words if not _is_proper_noun_shaped(w)]
    if not ordinary:
        return False
    wrong = [w for w in ordinary if _scripts_in(w) and not (_scripts_in(w) & wanted)]
    if not wrong:
        return False
    return (len(wrong) / len(ordinary)) >= _WRONG_WORD_RATIO


def violation_score(text: str, lang: str | None) -> int:
    """How badly ``text`` misses ``lang`` — lower is better, 0 is clean.

    Used to pick between an original reply and a re-asked one, so a retry can
    never leave the user with something worse than what it replaced.
    """
    text = (text or "").strip()
    if not text or not lang:
        return 0
    wanted = expected_scripts(lang)
    words = _WORD_RE.findall(text)
    ordinary = [w for w in words if not _is_proper_noun_shaped(w)]
    wrong = sum(1 for w in ordinary
                if _scripts_in(w) and not (_scripts_in(w) & wanted))
    # A welded word is a harder error than a foreign one — it is not a word in
    # any language, and it is what makes MEDO unintelligible rather than merely
    # inconsistent.
    return 3 * len(mixed_script_words(text)) + wrong
