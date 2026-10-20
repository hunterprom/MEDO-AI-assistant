"""«That was a bug» — the complaint that writes itself up.

What these pin down, in order of how much it would hurt to get wrong:

1. **The report is about the PREVIOUS turn.** A bug report that quotes the
   complaint instead of the fault is worthless, and saying it twice must not
   file the first report as the second bug.
2. **It does not steal the correction detector's phrases.** "That's wrong" and
   "no, I meant..." are how ``core/corrections.py`` measures fast-path
   misfires. Claiming them here would file a duplicate report AND corrupt that
   statistic, so the trigger word is mandatory.
3. **It degrades instead of failing.** Telemetry off, an unwritable directory,
   nothing said yet — each has a different honest answer, and none of them is
   a traceback in the middle of a conversation.
4. **It leaks nothing.** No tokens, no paths, no environment variables in the
   report; the directory is gitignored.
"""

from __future__ import annotations

import re

import pytest

from core.bugreports import (
    BugReportStore,
    Turn,
    environment,
    git_revision,
    render,
    slugify,
)
from core.config import PROJECT_ROOT, load_settings
from core.corrections import is_correction, is_refinement
from core.events import EventBus
from core.metrics import MetricsStore
from core.router import Router
from skills.base import Skill, SkillRegistry, SkillRequest, SkillResult
from skills.bug_report import BugReportSkill


class _Search(Skill):
    """Stands in for the skill that misbehaved."""

    name = "web_search"
    description = "search the web"
    patterns = [re.compile(r"search up (?:for )?(?P<q>.+)", re.IGNORECASE)]

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("The Mazda RX-7 is a Japanese sports car.")


def _store(tmp_path, *, metrics=True, registry=None, **kw) -> BugReportStore:
    return BugReportStore(
        tmp_path / "bugs",
        metrics=MetricsStore(tmp_path / "m.db") if metrics else None,
        registry=registry, root=PROJECT_ROOT, **kw)


def _router(tmp_path, *, telemetry: bool = True, enabled: bool = True):
    settings = load_settings()
    settings.memory.db_path = str(tmp_path / "m.db")
    settings.logging.routing_stats = True
    settings.logging.route_telemetry = telemetry
    settings.bug_reports.enabled = enabled
    registry = SkillRegistry()
    registry.register(_Search())
    store = _store(tmp_path, registry=registry)
    registry.register(BugReportSkill(store, config=settings.bug_reports))
    router = Router(settings, registry, None, EventBus())  # type: ignore[arg-type]
    router.model = None
    router.metrics = MetricsStore(tmp_path / "m.db") if telemetry else None
    return router, store


def _only_report(store: BugReportStore) -> str:
    reports = store.list()
    assert len(reports) == 1, [p.name for p in reports]
    return reports[0].read_text(encoding="utf-8")


# --- what gets reported ------------------------------------------------------


@pytest.mark.asyncio
async def test_the_report_quotes_the_turn_before_the_complaint(tmp_path):
    router, store = _router(tmp_path)
    await router.route("search up for Mazda RX-7")
    result = await router.route("that was a bug")

    report = _only_report(store)
    assert "search up for Mazda RX-7" in report
    assert "Japanese sports car" in report
    # The complaint itself is recorded as the REPORTING phrase, never as the bug.
    assert report.splitlines()[0] == "# Bug: search up for Mazda RX-7"
    assert result.data["file"].endswith(".md")


@pytest.mark.asyncio
async def test_the_routing_detail_is_what_says_where_to_look(tmp_path):
    router, store = _router(tmp_path)
    await router.route("search up for Mazda RX-7")
    await router.route("that was a bug")

    report = _only_report(store)
    assert "routed via **FAST**" in report
    assert "skill `web_search`" in report
    assert "match confidence" in report
    assert "Where to look" in report


@pytest.mark.asyncio
async def test_saying_it_twice_does_not_report_the_first_report(tmp_path):
    # The obvious way to write this feature files the bug report as the bug the
    # second time — and saying it twice is exactly what someone does when the
    # first one didn't seem to register.
    router, store = _router(tmp_path)
    await router.route("search up for Mazda RX-7")
    await router.route("that was a bug")
    await router.route("that was a bug")

    reports = store.list()
    assert len(reports) == 2
    for path in reports:
        text = path.read_text(encoding="utf-8")
        assert text.splitlines()[0] == "# Bug: search up for Mazda RX-7"
        assert "that was a bug" not in text.split("## The turn being reported")[1]


@pytest.mark.asyncio
async def test_the_run_up_turns_are_included_oldest_first(tmp_path):
    router, store = _router(tmp_path)
    await router.route("search up for one")
    await router.route("search up for two")
    await router.route("search up for three")
    await router.route("that was a bug")

    report = _only_report(store)
    assert report.splitlines()[0] == "# Bug: search up for three"
    led_up = report.split("## What led up to it")[1]
    assert led_up.index("search up for two") < led_up.index("search up for one")


@pytest.mark.asyncio
async def test_a_note_said_in_the_same_breath_is_kept(tmp_path):
    router, store = _router(tmp_path)
    await router.route("search up for Mazda RX-7")
    await router.route("that was a bug, it should have opened the browser")

    report = _only_report(store)
    assert "## What went wrong, in your words" in report
    assert "it should have opened the browser" in report
    # ...and the trigger words are not swallowed into the note.
    assert "that was a bug, it should" not in report.split("in your words")[1][:200]


@pytest.mark.asyncio
async def test_a_bare_complaint_still_produces_a_full_report(tmp_path):
    """"Just say that was a bug" is the whole point — no note is required."""
    router, store = _router(tmp_path)
    await router.route("search up for Mazda RX-7")
    await router.route("that was a bug.")

    report = _only_report(store)
    assert "## What went wrong, in your words" not in report
    assert "search up for Mazda RX-7" in report


# --- the phrases -------------------------------------------------------------


@pytest.mark.parametrize("phrase, note", [
    ("that was a bug", ""),
    ("that's a bug", ""),
    ("this is a bug", ""),
    ("that was a bug — the browser never opened", "the browser never opened"),
    ("report that as a bug", ""),
    ("log a bug", ""),
    ("save this as a bug", ""),
    ("write that down as a bug", ""),
    ("bug report: it spoke Macedonian in English", "it spoke Macedonian in English"),
    ("тоа беше баг", ""),
    ("ова е баг", ""),
    ("тоа беше грешка", ""),
    ("запиши го ова како баг", ""),
    ("пријави баг", ""),
    ("пријави грешка, не го отвори прелистувачот",
     "не го отвори прелистувачот"),
])
def test_the_phrases_people_actually_say(phrase, note, tmp_path):
    skill = BugReportSkill(_store(tmp_path))
    match = skill.match(phrase)
    assert match is not None, f"{phrase!r} did not trigger a bug report"
    request = SkillRequest(text=phrase, match=match)
    assert BugReportSkill._note_of(request) == note


@pytest.mark.parametrize("phrase", [
    # These belong to core/corrections.py — they are how the fast-path misfire
    # rate is measured. Claiming them here would both duplicate that signal and
    # corrupt it.
    "no, I meant the weather",
    "that's not what I meant",
    "that's the wrong one",
    "make it shorter",
    "не, мислев на времето",
    "не тоа",
    # ...and these are ordinary questions that merely contain the word.
    "what's wrong with my code",
    "is there a bug in this circuit",
    "debug my python script",
    "what does a bug bounty pay",
    # A bare Macedonian "грешка" is far too common a word to claim.
    "имам грешка во кодот",
    "каква е таа грешка",
])
def test_what_must_not_trigger_a_bug_report(phrase, tmp_path):
    skill = BugReportSkill(_store(tmp_path))
    assert skill.match(phrase) is None, f"{phrase!r} wrongly filed a bug report"


def test_the_correction_detectors_phrases_are_left_alone(tmp_path):
    """Belt and braces on the boundary above, read from the OTHER module.

    Written this way so that adding a phrase to core/corrections.py and
    accidentally claiming it here fails HERE, rather than silently halving the
    misfire statistic months later.
    """
    skill = BugReportSkill(_store(tmp_path))
    for phrase in ("no, I meant the timer", "that's not it", "I didn't say that",
                   "actually, I wanted the weather", "не мислев тоа"):
        assert is_correction(phrase) or is_refinement(phrase)
        assert skill.match(phrase) is None, phrase


def test_the_skill_cannot_be_reached_by_meaning(tmp_path):
    """No routing_phrases => the semantic tier never indexes it.

    A complaint-shaped question ("something's wrong with the weather skill")
    deserves an answer, not a silently written file.
    """
    skill = BugReportSkill(_store(tmp_path))
    assert not skill.routing_phrases
    assert Router._semantic_safe(skill) is True   # safe, but never a candidate


# --- degrading honestly ------------------------------------------------------


@pytest.mark.asyncio
async def test_nothing_said_yet_says_so_instead_of_writing_an_empty_file(tmp_path):
    router, store = _router(tmp_path)
    result = await router.route("that was a bug")
    assert "nothing to report" in result.speech
    assert store.list() == []


@pytest.mark.asyncio
async def test_it_still_works_with_routing_telemetry_switched_off(tmp_path):
    """Telemetry off costs the report its tier and latency, never the report."""
    router, store = _router(tmp_path, telemetry=False)
    await router.route("search up for Mazda RX-7")
    await router.route("that was a bug")

    report = _only_report(store)
    assert "search up for Mazda RX-7" in report      # from context, not the DB
    assert "Japanese sports car" in report
    assert "routed via" not in report


@pytest.mark.asyncio
async def test_switching_it_off_declines_out_loud(tmp_path):
    router, store = _router(tmp_path, enabled=False)
    await router.route("search up for Mazda RX-7")
    result = await router.route("that was a bug")
    assert "switched off" in result.speech
    assert store.list() == []


@pytest.mark.asyncio
async def test_untrusted_content_cannot_file_a_bug_report(tmp_path):
    """A fetched page containing "that was a bug" is content, not an instruction."""
    _router_, store = _router(tmp_path)
    skill = BugReportSkill(store, config=load_settings().bug_reports)
    result = await skill.execute(SkillRequest(
        text="that was a bug", context={"untrusted": True}))
    assert result.success is False
    assert result.data.get("refused") == "untrusted"
    assert store.list() == []


@pytest.mark.asyncio
async def test_an_unwritable_directory_is_reported_as_such(tmp_path):
    """Distinct from "nothing to report" — the fixes are completely different."""
    blocker = tmp_path / "bugs"
    blocker.write_text("I am a file, not a directory", encoding="utf-8")
    store = _store(tmp_path, metrics=False)
    skill = BugReportSkill(store, config=load_settings().bug_reports)
    result = await skill.execute(SkillRequest(
        text="that was a bug",
        context={"last_question": "search up X", "last_reply": "a paragraph"}))
    assert result.success is False
    assert "couldn't write" in result.speech


def test_a_broken_metrics_store_does_not_break_reporting(tmp_path):
    class Exploding:
        def history(self, **kw):
            raise RuntimeError("database is locked")

    store = BugReportStore(tmp_path / "bugs", metrics=Exploding(),
                           root=PROJECT_ROOT)
    assert store.recent_turns(3) == []
    path = store.capture(fallback=Turn(text="search up X", reply="a paragraph"))
    assert path is not None and path.exists()


# --- the file itself ---------------------------------------------------------


def test_reports_are_pruned_so_a_stuck_loop_cannot_fill_the_disk(tmp_path):
    store = _store(tmp_path, metrics=False, max_reports=3)
    from datetime import datetime, timedelta

    base = datetime(2026, 8, 25, 12, 0).astimezone()
    for i in range(6):
        store.capture(fallback=Turn(text=f"turn {i}"),
                      when=base + timedelta(seconds=i))
    assert len(store.list()) == 3


def test_collect_puts_every_report_in_one_document(tmp_path):
    store = _store(tmp_path, metrics=False)
    assert "No bug reports" in store.collect()
    from datetime import datetime, timedelta

    base = datetime(2026, 8, 25, 12, 0).astimezone()
    for i in range(3):
        store.capture(fallback=Turn(text=f"turn {i}"),
                      when=base + timedelta(seconds=i))
    combined = store.collect()
    assert combined.startswith("# MEDO bug reports (3)")
    # Oldest first, so the document reads in the order things happened.
    assert combined.index("turn 0") < combined.index("turn 2")


@pytest.mark.parametrize("text, expected", [
    ("search up for Mazda RX-7", "search-up-for-mazda-rx-7"),
    ("какво е времето", "kakvo-e-vremeto"),
    ("   ", "report"),
    ("!!!???", "report"),
])
def test_slugify_makes_a_findable_ascii_filename(text, expected):
    assert slugify(text) == expected


def test_a_long_utterance_is_cut_on_a_word_boundary():
    slug = slugify("please search the entire internet for absolutely "
                   "everything about rotary engines")
    assert len(slug) <= 48
    assert not slug.endswith("-")
    assert slug.split("-")[-1] in {"about", "for", "absolutely", "everything",
                                   "internet", "entire"}


def test_the_report_carries_no_secrets(tmp_path):
    """The environment block is platform + version, deliberately nothing else."""
    env = environment(PROJECT_ROOT)
    # "Running code" is two timestamps and a duration — whether the process
    # predates its own source files. No paths, no filenames, no environment.
    assert set(env) <= {"Platform", "Python", "MEDO revision", "Running code"}
    assert str(PROJECT_ROOT) not in " ".join(env.values())
    blob = render(Turn(text="x", reply="y"), env=env).lower()
    for forbidden in ("token", "secret", "api_key", "password", "bearer",
                      "authorization"):
        assert forbidden not in blob


def test_git_revision_degrades_to_nothing_outside_a_checkout(tmp_path):
    assert git_revision(tmp_path) == ""
    assert git_revision(PROJECT_ROOT)          # this repo IS one


def test_the_bugs_directory_is_gitignored():
    """The reports quote what was said in the room. Committing them is a
    mistake you cannot take back, so the ignore rule is part of the feature."""
    rules = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert re.search(r"^bugs/$", rules, re.MULTILINE)


def test_render_is_pure_and_survives_a_turn_it_knows_nothing_about():
    body = render(Turn(text="", reply=""))
    assert "# Bug: unspecified" in body
    assert "(not recorded)" in body


def test_a_giant_reply_is_clipped_rather_than_pasted_whole():
    body = render(Turn(text="x", reply="y" * 9000))
    assert "[…]" in body
    assert len(body) < 4000


# --- against the REAL registry ----------------------------------------------
#
# The tests above use a two-skill registry, which cannot see the failure this
# project has actually shipped twice: registration ORDER decides the fast path,
# so a phrase another skill already claims never reaches the skill that wants
# it, silently, and you get that other skill's answer instead. Both example
# routines hit exactly this. These run against every skill MEDO registers.


def _real_registry():
    from main import Announcer, build_registry
    return build_registry(load_settings(), Announcer())


@pytest.mark.parametrize("phrase", [
    "that was a bug", "that's a bug", "this is a bug",
    "report that as a bug", "log a bug", "save this as a bug",
    "write that down as a bug", "bug report: the browser never opened",
    "that was a bug, it should have opened the browser",
    "тоа беше баг", "ова е баг", "тоа беше грешка",
    "запиши го ова како баг", "пријави баг", "пријави грешка",
])
def test_no_other_skill_claims_the_bug_phrases_first(phrase):
    hit = _real_registry().find_match(phrase)
    assert hit is not None and hit[0].name == "bug_report",         f"{phrase!r} was claimed by {hit[0].name if hit else None}"


@pytest.mark.parametrize("phrase, expected", [
    # Things that merely contain the word, against the full skill list.
    ("search up how to fix a bug in python", "web_search"),
    ("google bug bounty programs", "web_search"),
    ("there's a bug on my screen", None),
    ("what does a software bug cost", None),
    ("debug my python script", None),
    ("поправи ја грешката", None),
    ("имам грешка во кодот", None),
    # ...and the everyday commands that must keep working unchanged.
    ("what time is it", "datetime"),
    ("какво е времето", "weather"),
    ("set a timer for five minutes", "timers"),
    ("close yourself", "quit"),
    ("take a note buy milk", "notes"),
])
def test_the_bug_skill_steals_nothing_from_the_real_registry(phrase, expected):
    hit = _real_registry().find_match(phrase)
    assert (hit[0].name if hit else None) == expected


def test_the_faulty_turns_own_timestamp_is_shown():
    """A fresh session's newest recorded turn can be from LAST night.

    Someone launches MEDO and says "that was a bug" about what it did
    yesterday. Only the turn's own timestamp, printed next to the report's,
    tells the reader those are two different moments.
    """
    body = render(Turn(text="what time is it", reply="It's 10:55 AM.",
                       ts="2026-09-03 22:14:07"))
    assert "recorded 2026-09-03 22:14:07" in body


def test_a_recorded_timestamp_is_shown_in_the_readers_own_clock():
    """`route_events.ts` is SQLite's `datetime('now')` — UTC, unmarked.

    Printed raw beside the report's local `When:`, a turn answered at 10:55
    read "recorded 08:55". Two hours adrift, in the one document whose job is
    to say when things happened. Found in a real report the moment the feature
    was used.
    """
    from datetime import datetime, timezone

    from core.bugreports import local_ts

    utc = "2026-09-04 08:55:35"
    expected = (datetime.fromisoformat(utc).replace(tzinfo=timezone.utc)
                .astimezone().strftime("%Y-%m-%d %H:%M:%S"))
    assert local_ts(utc) == expected
    # ...and it must survive the trip through a history row.
    assert Turn.from_row({"text": "x", "ts": utc}).ts == expected


@pytest.mark.parametrize("value", ["", "not a date", "2026-13-45 99:99:99"])
def test_an_unreadable_timestamp_is_shown_rather_than_swallowed(value):
    from core.bugreports import local_ts

    assert local_ts(value) == value.strip()


# -- "the fix is in the file but not in the process" ---------------------------

def test_a_report_says_when_the_running_code_is_stale(tmp_path):
    """The most expensive false trail this project has produced.

    A fix lands, the source is right, the suite is green — and MEDO keeps
    behaving exactly as before, because Python imported its modules at launch.
    The revision line cannot show it (the change isn't committed), so two
    rounds of bug reports were spent chasing routing that was already fixed.
    """
    from core.bugreports import running_code
    (tmp_path / "core").mkdir()
    source = tmp_path / "core" / "router.py"
    source.write_text("x = 1", encoding="utf-8")
    edited = source.stat().st_mtime

    said = running_code(tmp_path, started=edited - 7200)
    assert "OLD CODE" in said and "restart" in said.lower()
    assert "2h00m" in said


def test_a_report_stays_quiet_when_the_process_is_current(tmp_path):
    from core.bugreports import running_code
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "router.py").write_text("x = 1", encoding="utf-8")
    edited = (tmp_path / "core" / "router.py").stat().st_mtime

    said = running_code(tmp_path, started=edited + 60)
    assert "OLD CODE" not in said
    assert "started" in said


def test_only_code_MEDO_runs_counts_as_stale(tmp_path):
    """Editing a test or a doc changes nothing about a running assistant, so it
    must not raise a false "restart me" on every report."""
    from core.bugreports import newest_source
    (tmp_path / "core").mkdir()
    code = tmp_path / "core" / "router.py"
    code.write_text("x = 1", encoding="utf-8")
    import os
    os.utime(code, (1_000_000, 1_000_000))

    for ignored in ("tests", "docs"):
        (tmp_path / ignored).mkdir()
        later = tmp_path / ignored / "thing.py"
        later.write_text("y = 2", encoding="utf-8")
        os.utime(later, (2_000_000, 2_000_000))

    assert newest_source(tmp_path) == 1_000_000


def test_the_staleness_line_is_in_the_report_environment(tmp_path):
    from core.bugreports import environment
    env = environment(tmp_path)
    assert "Running code" in env
    # ...and it still leaks nothing: no paths, no env vars.
    assert str(tmp_path) not in " ".join(env.values())
