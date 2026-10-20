"""Saying "that was a bug" and getting a file you can hand to a developer.

The gap this closes is not "MEDO has bugs". It is that the evidence evaporates.
A bug shows up in the middle of using MEDO by voice: it answered instead of
opening the browser, it routed "the current approach" to the power skill, it
read a Macedonian reply in an English voice. By the time anyone sits down to
write it up, the exact phrasing is gone, which tier fired is gone, and what is
left is "it did something weird with search yesterday" — unactionable.

So the report is written **at the moment of the complaint**, from what MEDO
already recorded, with no typing:

    You:  search up the Mazda RX-7
    MEDO: The Mazda RX-7 is a Japanese sports car...
    You:  that was a bug, it should have opened the browser
    MEDO: Saved as bug-2026-08-25-1547-search-up-the-mazda-rx-7.md.

Three things make that file worth more than a sentence in a chat window:

* **It captures the turn BEFORE the complaint.** "That was a bug" is about what
  just happened, never about itself. The router hands every skill the previous
  question and reply in ``context``, and the metrics store has the same turn
  with the routing tier, the matched skill, the match confidence and the
  latency attached — the four fields that say *where in the code* to look.
* **It captures the turns before that.** A misroute is often set up by the
  preceding exchange; three turns of run-up cost nothing and repeatedly turn
  out to be the whole story.
* **It is a file, not a database row.** The point is to hand it to someone.

**Privacy.** A report contains what was said in the room. The directory is
gitignored, nothing is ever sent anywhere, and this module writes no
credentials of its own — the environment block is deliberately platform and
version only. What the user chooses to do with the file afterwards is the
user's decision, which is exactly why it is a file they can read first.
"""

from __future__ import annotations

import logging
import platform
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

#: When this process started, near enough — this module is imported during
#: startup, so import time is launch time to within a second.
_STARTED = time.time()

#: Where MEDO's own code lives, for the staleness check below. Tests and docs
#: are excluded: editing them cannot change what a running MEDO does.
_SOURCE_DIRS = ("core", "skills", "llm", "studio", "voice", "security",
                "software", "remote", "ui", "vision")

#: Longest slug taken from an utterance for the filename. Long enough to tell
#: two reports apart at a glance in a directory listing, short enough that the
#: whole name still fits on one line of a terminal.
SLUG_CHARS = 48

#: Caps on what goes INTO a report. The metrics store already truncates
#: (500/2000), but the context fallback does not, and a bug report that quotes
#: a 40 KB document summary in full is one nobody will read.
MAX_TEXT = 600
MAX_REPLY = 2500

#: Reports kept on disk before the oldest are pruned. This is a bug drawer, not
#: an archive; without a cap a stuck "that was a bug" loop (or a routine that
#: somehow reached the skill) fills the disk one small file at a time.
MAX_REPORTS = 500


def local_ts(stamp: str) -> str:
    """A metrics timestamp rendered in the reader's own clock.

    ``route_events.ts`` defaults to SQLite's ``datetime('now')``, which is UTC
    with no marker on it. Printed raw next to the report's local ``When:``, a
    turn answered at 10:55 read "recorded 08:55" — two hours adrift, in a
    document whose whole job is to say when things happened. Converted here
    rather than in the metrics store: that store's UTC is correct and shared by
    other readers; it is the DISPLAY that has to be local.

    Anything that doesn't parse is returned untouched — a timestamp MEDO cannot
    read is still better shown than swallowed.
    """
    text = (stamp or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True)
class Turn:
    """One exchange, with as much routing detail as was recorded.

    Everything but ``text`` is optional because the two sources fill in
    different amounts: the metrics store knows the tier and the latency, the
    router's ``context`` fallback knows only what was said. A report built from
    the poorer source is still a useful report, so nothing here is required.
    """

    text: str
    reply: str = ""
    tier: str = ""
    skill: str = ""
    confidence: float | None = None
    latency_ms: float | None = None
    ts: str = ""

    @classmethod
    def from_row(cls, row: dict) -> "Turn":
        """One :meth:`core.metrics.MetricsStore.history` row as a Turn."""
        return cls(
            text=str(row.get("text") or ""),
            reply=str(row.get("reply") or ""),
            tier=str(row.get("tier") or ""),
            skill=str(row.get("skill") or ""),
            confidence=row.get("confidence"),
            latency_ms=row.get("latency_ms"),
            ts=local_ts(str(row.get("ts") or "")),
        )

    @property
    def empty(self) -> bool:
        return not (self.text.strip() or self.reply.strip())


def slugify(text: str, limit: int = SLUG_CHARS) -> str:
    """A filename-safe stub of an utterance.

    ASCII-only on purpose. Cyrillic survives a modern filesystem perfectly
    well, but these filenames get pasted into terminals, chat windows and issue
    trackers on machines whose encoding is anyone's guess, and a mangled name is
    a file nobody can find. A Macedonian report is therefore named by its
    timestamp with a transliterated stub where one is available.
    """
    text = (text or "").strip().lower()
    text = _transliterate(text)
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    if len(text) > limit:
        # Cut on a word boundary so the name doesn't end mid-syllable.
        text = text[:limit].rsplit("-", 1)[0] or text[:limit]
    return text or "report"


#: Macedonian Cyrillic -> Latin, for filenames only. This is the ordinary
#: transliteration people already use when writing Macedonian on a keyboard
#: without Cyrillic; it is never used for anything MEDO says or stores.
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ѓ": "gj", "е": "e",
    "ж": "zh", "з": "z", "ѕ": "dz", "и": "i", "ј": "j", "к": "k", "л": "l",
    "љ": "lj", "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "ќ": "kj", "у": "u", "ф": "f", "х": "h", "ц": "c",
    "ч": "ch", "џ": "dj", "ш": "sh",
}


def _transliterate(text: str) -> str:
    return "".join(_TRANSLIT.get(ch, ch) for ch in text)


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " […]"


def git_revision(root: Path) -> str:
    """``branch @ abc1234``, or "" when this isn't a checkout.

    Read straight out of ``.git`` rather than shelled out to ``git``: this runs
    on the voice path, where spawning a process costs more than the answer is
    worth, and a machine that installed MEDO from a zip has no git at all.
    """
    head_file = root / ".git" / "HEAD"
    try:
        head = head_file.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not head.startswith("ref:"):
        return head[:12]                       # detached: the SHA itself
    ref = head.split(":", 1)[1].strip()
    branch = ref.rsplit("/", 1)[-1]
    try:
        sha = (root / ".git" / ref).read_text(encoding="utf-8").strip()
    except OSError:
        # A ref that has been packed has no loose file; find it in packed-refs.
        sha = ""
        try:
            for line in (root / ".git" / "packed-refs").read_text(
                    encoding="utf-8").splitlines():
                if line.endswith(" " + ref):
                    sha = line.split(" ", 1)[0]
                    break
        except OSError:
            pass
    return f"{branch} @ {sha[:7]}" if sha else branch


def environment(root: Path) -> dict[str, str]:
    """Machine facts worth knowing, and nothing else.

    Explicitly NOT a system dump. Every field here is chosen because it has
    actually explained a bug in this project — the OS decides which key chords
    and mouse backend are live, the Python version decides which asyncio
    behaviour applies, and the revision says which code the reporter was
    running. No paths, no environment variables, no tokens: a bug report is
    meant to be shareable, and the way to keep it shareable is to put nothing
    private in it in the first place.
    """
    env = {
        "Platform": f"{platform.system()} {platform.release()} ({sys.platform})",
        "Python": platform.python_version(),
    }
    revision = git_revision(root)
    if revision:
        env["MEDO revision"] = revision
    env["Running code"] = running_code(root)
    return env


def newest_source(root: Path) -> float:
    """Modification time of the newest of MEDO's own source files, or 0.0.

    Only the directories that hold code MEDO executes — a doc or a test can be
    edited all day without changing what the running assistant does.
    """
    newest = 0.0
    for name in _SOURCE_DIRS:
        directory = Path(root) / name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                newest = max(newest, path.stat().st_mtime)
            except OSError:
                continue
    for name in ("main.py", "config.yaml"):
        try:
            newest = max(newest, (Path(root) / name).stat().st_mtime)
        except OSError:
            continue
    return newest


def running_code(root: Path, *, started: float | None = None,
                 now: float | None = None) -> str:
    """Whether the code in this process is the code on disk.

    The single most expensive false trail this project has produced: a fix is
    made, the source on disk is right, the test suite is green — and the
    running MEDO keeps behaving exactly as before, because Python imported its
    modules at launch and a file edited afterwards is simply not in the
    process. Two rounds of bug reports were spent on it, and the report itself
    could not tell: the revision line reads the same either way, since the
    change was never committed. So say it outright.
    """
    started = _STARTED if started is None else started
    newest = newest_source(root)
    if not newest:
        return "started " + _clock(started)
    stale = newest - started
    if stale <= 0:
        return f"started {_clock(started)} · sources unchanged since"
    return (f"started {_clock(started)}, but MEDO's source files were last "
            f"changed {_clock(newest)} — {_duration(stale)} LATER. This "
            f"process is running OLD CODE; restart MEDO before trusting any "
            f"routing shown here.")


def _clock(stamp: float) -> str:
    return datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M:%S")


def _duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 1:
        return "seconds"
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours, rest = divmod(minutes, 60)
    return f"{hours}h{rest:02d}m"


def source_of(skill_name: str, registry, root: Path) -> str:
    """Repo-relative file a skill lives in, for the "where to look" line.

    Derived from the live registry rather than a hand-kept name->file table,
    because a table is wrong the first time a skill moves and nobody notices
    until the pointer sends someone to a file that no longer exists.
    """
    if not skill_name or registry is None:
        return ""
    try:
        skill = registry.get(skill_name)
        module = sys.modules.get(type(skill).__module__) if skill else None
        path = Path(getattr(module, "__file__", "") or "")
        if not path:
            return ""
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, AttributeError, TypeError):
        return ""                              # outside the repo, or a mock


def _turn_block(turn: Turn, *, heading: str) -> list[str]:
    """One exchange rendered as markdown."""
    lines = [f"### {heading}", ""]
    lines.append(f"**You said:** {_clip(turn.text, MAX_TEXT) or '(not recorded)'}")
    lines.append("")
    lines.append(f"**MEDO replied:** {_clip(turn.reply, MAX_REPLY) or '(not recorded)'}")
    facts = []
    if turn.ts:
        # WHEN the faulty turn happened, which is not always "just now". A
        # fresh session's newest recorded turn can be yesterday's — someone
        # launches MEDO and says "that was a bug" about what it did last
        # night. Printing the turn's own timestamp next to the report's makes
        # that visible to whoever reads it instead of quietly implying the two
        # are the same moment.
        facts.append(f"recorded {turn.ts}")
    if turn.tier:
        facts.append(f"routed via **{turn.tier}**")
    if turn.skill:
        facts.append(f"skill `{turn.skill}`")
    if turn.confidence is not None:
        # The share of the sentence the winning regex actually matched. A low
        # number next to a complaint is the signature of a fast-path misfire,
        # which is why it is worth carrying into the report.
        facts.append(f"match confidence {turn.confidence}")
    if turn.latency_ms is not None:
        facts.append(f"{round(float(turn.latency_ms))} ms")
    if facts:
        lines += ["", "- " + " · ".join(facts)]
    return lines


def render(faulty: Turn, *, note: str = "", preceding: list[Turn] | None = None,
           when: datetime | None = None, language: str = "en",
           env: dict[str, str] | None = None, source: str = "",
           reported_with: str = "") -> str:
    """The whole report as markdown. Pure — no clock, no disk, no config.

    Split out from the writing so the format can be tested exactly, and so the
    same rendering serves the file, the ``--collect`` output and anything that
    later wants to show a report in the HUD.
    """
    when = when or datetime.now().astimezone()
    title = _clip(faulty.text.strip(), 70) or "unspecified"
    out = [f"# Bug: {title}", ""]

    meta = [f"- **When:** {when.isoformat(timespec='seconds')}"]
    if reported_with:
        meta.append(f"- **Reported by saying:** \"{_clip(reported_with, 200)}\"")
    meta.append(f"- **Conversation language:** {language}")
    for key, value in (env or {}).items():
        meta.append(f"- **{key}:** {value}")
    out += meta + [""]

    if note:
        # The user's own words go HIGH in the document. Whoever reads this
        # wants the human sentence before the telemetry, and a note buried
        # under three tables is one that gets skimmed past.
        out += ["## What went wrong, in your words", "", _clip(note, MAX_TEXT), ""]

    out += ["## The turn being reported", ""]
    out += _turn_block(faulty, heading="The exchange")
    out += [""]

    if source:
        out += ["### Where to look", "",
                f"- Skill source: `{source}`",
                f"- Routing tier: {faulty.tier or 'unknown'} "
                f"(`core/router.py`)", ""]

    if preceding:
        out += ["## What led up to it", "",
                "_Oldest first. A misroute is often set up by the exchange "
                "before it._", ""]
        for index, turn in enumerate(preceding, start=1):
            ago = "1 turn earlier" if index == 1 else f"{index} turns earlier"
            out += _turn_block(turn, heading=ago)
            out += [""]

    out += ["---", "",
            "_Written by MEDO's bug-report skill (`skills/bug_report.py`). "
            "Nothing here left this machine._", ""]
    return "\n".join(out)


class BugReportStore:
    """Where reports live, and the one method that makes one.

    The store owns the directory and the pruning; deciding *that* a bug is
    being reported is the skill's job, and deciding what a report says is
    :func:`render`'s. Kept separate because the first is voice-shaped, the
    second is IO-shaped and the third is a pure function worth testing on its
    own.
    """

    def __init__(self, directory: str | Path, *, metrics=None, registry=None,
                 root: Path | None = None, context_turns: int = 3,
                 max_reports: int = MAX_REPORTS) -> None:
        self._dir = Path(directory)
        self._metrics = metrics
        self._registry = registry
        self._root = root or Path(__file__).resolve().parent.parent
        self._context_turns = max(0, int(context_turns))
        self._max_reports = max(1, int(max_reports))

    @property
    def directory(self) -> Path:
        return self._dir

    def list(self) -> list[Path]:
        """Every report on disk, newest first."""
        try:
            return sorted(self._dir.glob("bug-*.md"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return []

    def recent_turns(self, limit: int, *, exclude_skill: str = "") -> list[Turn]:
        """The last ``limit`` turns, newest first, from the metrics store.

        ``exclude_skill`` drops the bug-report skill's own turns. Without it,
        a second "that was a bug" reports the FIRST bug report as the bug —
        which is both useless and exactly what an annoyed user does when the
        first report didn't seem to register.
        """
        if self._metrics is None or limit <= 0:
            return []
        try:
            rows = self._metrics.history(limit=limit + 6)
        except Exception:  # noqa: BLE001 - a report must not need a healthy DB
            logger.debug("bug report: could not read routing history",
                         exc_info=True)
            return []
        turns = []
        for row in rows:
            if exclude_skill and str(row.get("skill") or "") == exclude_skill:
                continue
            turn = Turn.from_row(row)
            if not turn.empty:
                turns.append(turn)
            if len(turns) >= limit:
                break
        return turns

    def capture(self, *, note: str = "", reported_with: str = "",
                language: str = "en", fallback: Turn | None = None,
                exclude_skill: str = "", when: datetime | None = None) -> Path | None:
        """Write one report. Returns its path, or None when there was nothing.

        ``fallback`` is the turn the ROUTER knows about (``context`` carries the
        previous question and reply on every request). It is used when routing
        telemetry is switched off or the database is unavailable, so turning
        telemetry off costs the report its tier and latency but never the
        report itself.
        """
        turns = self.recent_turns(self._context_turns + 1,
                                  exclude_skill=exclude_skill)
        faulty = turns[0] if turns else None
        preceding = turns[1:] if len(turns) > 1 else []
        if faulty is None or faulty.empty:
            faulty, preceding = fallback, []
        if faulty is None or faulty.empty:
            return None

        body = render(faulty, note=note, preceding=preceding, when=when,
                      language=language, env=environment(self._root),
                      source=source_of(faulty.skill, self._registry, self._root),
                      reported_with=reported_with)
        return self._write(faulty, body, when=when)

    def _write(self, faulty: Turn, body: str,
               when: datetime | None = None) -> Path | None:
        when = when or datetime.now().astimezone()
        stamp = when.strftime("%Y-%m-%d-%H%M%S")
        stem = f"bug-{stamp}-{slugify(faulty.text)}"
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._unique(stem)
            path.write_text(body, encoding="utf-8", newline="\n")
        except OSError:
            logger.warning("could not write a bug report to %s", self._dir,
                           exc_info=True)
            return None
        self._prune()
        return path

    def _unique(self, stem: str) -> Path:
        """A path that is not already taken, suffixing ``-2``, ``-3``, ...

        The name is a one-SECOND timestamp plus a slug of the utterance, so two
        reports about the same faulty turn inside the same second collide
        exactly — which is what happens when someone says "that was a bug"
        twice because the first one didn't seem to register, or re-sends it
        from the HUD. Writing straight to the name silently overwrote the
        earlier report: a bug-tracking feature losing bug reports.
        """
        path = self._dir / f"{stem}.md"
        counter = 2
        while path.exists() and counter < 100:
            path = self._dir / f"{stem}-{counter}.md"
            counter += 1
        return path

    def _prune(self) -> None:
        """Drop the oldest reports past the cap. Never fatal."""
        reports = self.list()
        for path in reports[self._max_reports:]:
            try:
                path.unlink()
            except OSError:
                pass

    def collect(self) -> str:
        """Every report concatenated into one document, oldest first.

        This is the "give it to you" step. Handing over eleven files means
        eleven attachments and a reader who reads three of them; handing over
        one document means the whole picture arrives together.
        """
        reports = list(reversed(self.list()))
        if not reports:
            return "No bug reports have been saved.\n"
        parts = [f"# MEDO bug reports ({len(reports)})", ""]
        for path in reports:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            parts += [f"<!-- {path.name} -->", "", text.strip(), "", "---", ""]
        return "\n".join(parts) + "\n"


def _main(argv=None) -> int:  # pragma: no cover - operator utility
    """``python -m core.bugreports`` — list, print, or collect saved reports."""
    import argparse

    from core.config import PROJECT_ROOT, load_settings

    parser = argparse.ArgumentParser(description="MEDO bug reports")
    parser.add_argument("--collect", action="store_true",
                        help="print every report as one document (for pasting)")
    parser.add_argument("--latest", action="store_true",
                        help="print the most recent report")
    parser.add_argument("--dir", default=None, help="reports directory")
    args = parser.parse_args(argv)

    settings = load_settings()
    directory = Path(args.dir) if args.dir else \
        PROJECT_ROOT / settings.bug_reports.directory
    store = BugReportStore(directory)
    reports = store.list()

    if args.collect:
        print(store.collect())
        return 0
    if args.latest:
        if not reports:
            print("No bug reports have been saved.")
            return 1
        print(reports[0].read_text(encoding="utf-8"))
        return 0

    if not reports:
        print(f"No bug reports in {directory}.\n"
              f'Say "that was a bug" to MEDO right after something goes wrong.')
        return 0
    print(f"{len(reports)} report(s) in {directory}, newest first:\n")
    for path in reports:
        first = ""
        try:
            first = path.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError):
            pass
        print(f"  {path.name}\n      {first.removeprefix('# ').strip()}")
    print("\npython -m core.bugreports --collect   # one document to hand over")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
