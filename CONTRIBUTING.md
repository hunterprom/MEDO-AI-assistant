# Contributing to MEDO

MEDO is a local-first voice assistant: everything runs on the owner's own
hardware, and nothing leaves the machine unless they asked it to. That single
property decides most of what follows.

## Before you start

- **Read `docs/decisions.md`.** It records why things are the way they are.
  Several of the odder-looking choices are load-bearing, and the file exists so
  you don't have to rediscover that by breaking one.
- **`docs/architecture.md` describes contracts, not suggestions.** The HUD's
  render path and the vision sidecar's isolation are both frozen; changing them
  needs a decision record, not a patch.
- **`docs/extending.md`** covers the usual reasons to be here — a new skill, a
  plugin, a specialist, a routine — with working examples.

## Setting up

```bash
git clone https://github.com/hunterprom/MEDO.git
cd MEDO
run.bat            # Windows — builds both venvs, pulls the models
./run.command      # macOS / Linux
```

First run downloads roughly 24 GB of models. It only happens once;
`scripts/setup_marker.py` records what was installed, and later launches
reinstall only what a changed `requirements*.txt` actually invalidated.

```bash
.venv\Scripts\python -m pytest -q          # the suite (Windows)
.venv/bin/python -m pytest -q              # macOS / Linux
.venv\Scripts\python -m ruff check .       # lint
python main.py --once "what time is it"    # route one request, print, exit
```

## The rules that are not style preferences

**Test in both languages.** MEDO is English and Macedonian. The fast path is
regex, and a pattern written in English is simply deaf to the Macedonian
phrasing — the request falls through to the LLM, which then narrates, denies a
capability it has, or invents something. This is not hypothetical: four
consecutive review rounds passed a 2264-test suite while MEDO was broken for
nearly every Macedonian utterance, because every test was in English. If your
change touches routing, add a Macedonian case.

**A skill is dual-path.** Regex `patterns` *and* a `tool_schema`. One path only
means half the assistant can't reach your feature.

**Nothing destructive runs without a spoken yes.** If a skill acts on the
machine, declare `controls_pc`; if it is irreversible, declare
`requires_confirmation` and let the router hold it — don't self-gate inside
`execute` and rely on that. `tests/test_destructive_routes.py` drives every
routing tier at a deliberately careless skill and fails if any of them reaches
it. Adding a route means adding a case there.

**Nothing leaves the machine by default.** A new outbound call needs a config
flag, off by default, and a line in `docs/security.md` saying what is sent.

**Degrade, don't crash.** Ollama is down, the camera is busy, the network is
gone, the model is cold — all normal. Say so and carry on.

## Commits and pull requests

- One change per commit, present tense, saying what changed for the *user*:
  `fix: a search command opens the browser, a question still gets answered`.
- Update `README-detailed.md` when user-visible behaviour changes, and
  `docs/decisions.md` when you make a choice the next person would question.
- New behaviour needs tests. The suite is fast (~3 minutes) and is expected to
  be green before you push.
- Never commit `*.db`, `secrets.local.yaml*`, `models/*`, `screenshots/`, or
  anything under `data/`. They hold personal data, credentials, or a private
  key.

## Reporting a security issue

Don't open a public issue. MEDO can control the machine it runs on, so a bug in
the policy engine, the trust boundary, or the companion API's auth is worth
handling quietly first. Email matejilioskiid@gmail.com with what you found and
how to reproduce it. `docs/security.md` §8 has the threat model, including the
limits that are known and accepted.
