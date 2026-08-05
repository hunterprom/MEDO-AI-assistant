# Opus Fixlist — verified findings from the Fable review loop (2026-07-31)

> **STATUS 2026-07-31 — round 1 (bugs 1–7) FIXED, round 2 FIXED** (Opus pass,
> full suite **2258 passed**, ruff clean). #8 is a documented known-limit.
> Commits: `c2129e9`, `30e85f5`, `d127883`, `27c4572`. The original findings are
> kept verbatim below as the record of what was wrong and why.
>
> **Round 2** re-reviewed the fixes themselves plus the untouched core and found
> more, all now fixed: a **data-loss bug the round-1 orphan sweep introduced**
> (a corrupt `project.json` made the sweep delete the entire version history —
> now skipped on the recovery path, plus atomic metadata writes); **dev mode
> turning "discard my change" into "merge it"** (`self_dev` used `confirmed` to
> pick a branch, and `security.yolo` injects it on every utterance); **"set
> volume to 10" turning the volume UP** on any machine without an audio
> endpoint; **PowerSkill announcing "Shutting down now." when the OS refused**;
> `/provider` switching the brain even when it returns 400; `PlaySkill` claiming
> it opened results when the fallback also failed; `done_succeeded` wrong in
> both directions; and a full rework of the 3D fast path (24 verified false
> positives killed — no modifier blocklist can separate "a case for my phone"
> from "a case for my promotion", so idiom-prone nouns now need a *fabrication*
> verb).
>
> ## STILL OPEN — verified, not yet fixed (for the next pass)
>
> 1. **MED-HIGH — a streamed model preamble is spoken before the tool that
>    contradicts it.** `core/router.py:1214` passes `on_delta` on *every* tool
>    round; the ollama/openai streaming paths emit content deltas before
>    `tool_calls` are known (the Anthropic path guards correctly at
>    `llm/client.py:309`). Scenario: "open youtube" → the model streams *"Sure,
>    opening YouTube"* → `webbrowser.open` returns False → MEDO then says *"I
>    couldn't find a browser."* It already claimed the action. Fix: suppress
>    deltas on rounds where tools are offered, or buffer until the round resolves.
>    Note `tests/test_bugfix_deep_debug4.py:246` locks in that the preamble DOES
>    stream (it guards the opposite bug), so that test needs rethinking together.
> 2. **MED — `safety.confirm_destructive: false` makes every destructive skill a
>    permanent dead end.** `core/router.py:813` and `:1325` only arm the pending
>    confirmation when the flag is on, but the skills still self-gate on
>    `context["confirmed"]` — so they keep returning the prompt and the "yes"
>    never routes back. Turning confirmations OFF disables the actions instead of
>    auto-running them. Fix: when the flag is off, inject `confirmed=True` (like
>    dev mode does) rather than skipping the pending state.
> 3. **LOW-MED — a confirmation mid-batch silently discards the rest of a
>    multi-tool turn.** `core/router.py:1325-1331` returns from inside the
>    `for call in tool_calls` loop, throwing away already-collected speech and
>    never running (or mentioning) later calls. "Take a screenshot and then shut
>    down" → after "yes" the user believes both ran.
> 4. **LOW-MED — two MEDO Link devices can collide on a skill name.**
>    `link/registry.py:159` builds `f"{device_id}_{cap}"`, so `robo`+`dog_sit`
>    and `robo_dog`+`sit` both yield `robo_dog_sit`; `SkillRegistry.register`
>    raises → HTTP 500, the device is half-registered, and the shared capability
>    dispatches to the FIRST device. `load_persisted` swallows the same error at
>    startup, so the fleet silently loses capabilities after a reboot.
> 5. **LOW — `press_keys` drops trailing words and turns sequences into chords.**
>    `skills/desktop.py:43-65` — "press control s and close the window" saves and
>    silently ignores the rest; "press a and then b" presses them
>    *simultaneously* and reports "Pressed a b."

Context for a fresh session: repo `MEDO-main`, branch `medo-main-work` @ `fbd0227`
(full suite **2201 passed, 1 skipped**; ruff `--select F` clean). These findings
were verified by reading the code — each has a file:line, a concrete failure
scenario, and a suggested fix. They are all in code written AFTER the big
5-agent hardening pass (`6d5413c`), i.e. Maker Studio S2–S5 and the hardening
fixes themselves. Fix, add a regression test per item, run the full suite, then
commit with the usual trailers and push to `origin medo-main-work`
(fetch + reconcile first — parallel sessions push here).

---

## Bugs (fix these)

### 1. MED — Studio clarifying-question drops the user's answer
`skills/maker_studio.py:66` returns `SkillResult(result.question, await_reply=True)`,
but `_StudioSkill.execute` has **no `captured_reply` handler** — compare
`skills/app_builder_skill.py:76`, `skills/second_opinion.py:233`,
`skills/self_dev_skill.py:92`, which all check `request.context.get("captured_reply")`.

- **Scenario:** LLM tool path calls `model_3d` with a too-short description →
  the domain's `build_brief` asks "What should I model?" (`await_reply=True`) →
  the user answers "a gear" → the router routes the reply back with
  `context['captured_reply']` → `_description()` reads only `match`/`args`
  (never `request.text`), returns "" → the skill re-asks **without**
  `await_reply` → dead end; the answer is silently dropped.
- **Fix:** at the top of `execute`, if `request.context.get("captured_reply")`,
  use `request.text` as the description (mirror `MakeAppSkill`). Also honour a
  cancel ("never mind") like `app_builder_skill._CANCEL`.

### 2. MED — 3D fast-path hijacks idioms ("make a case for…")
`skills/maker_studio.py:117` — the noun alternation includes `case|stand|box|
hook|handle` with verbs `design|make|create|print|model`.

- **Scenario (studio.enabled on):** "make a case for hiring more engineers" →
  `desc="case for hiring more engineers"` → MEDO starts a 3D build (or says
  Studio is disabled) instead of routing the sentence to the LLM. Same for
  "make a stand against X", "take/make a box …".
- **Fix:** drop the idiom-prone bare nouns (`case`, `stand`, `box`, `hook`,
  `handle`) from the *generic* alternation and keep them only when a physical
  qualifier is present (e.g. `phone case`, `case for my ESP32` — require
  `for\s+(?:my|the|this|an?)\s+\w+` **plus** a device-ish word, or require a
  `3d|print|model` cue for those five). The LLM tool path still covers the
  natural phrasings. Add negative tests: "make a case for hiring", "make a
  stand against corruption" must NOT match.

### 3. LOW-MED — files.py extension guard false-positives on versioned app names
`skills/files.py:181` — `re.search(r"\.[a-z0-9]{1,6}$", app)` treats ANY
dot-suffix as a file extension.

- **Scenario:** "open a file in Python 3.12" → app tail `"Python 3.12"` ends in
  `.12` → looks like an extension → open-with skipped → doomed filename search
  ("I couldn't find a file matching 'in Python 3.12'"). Same for "Arduino IDE
  2.3.2".
- **Fix:** replace the generic pattern with a curated real-extension list, e.g.
  `\.(png|jpe?g|gif|bmp|webp|txt|md|pdf|docx?|xlsx?|pptx?|csv|json|xml|ya?ml|
  py|ino|c|cpp|h|js|ts|html|css|zip|rar|7z|mp3|wav|mp4|mkv|stl|step|svg)$`.
  Keep the existing regression test (`screenshot in chrome.png`) and add
  "open a file in Python 3.12" → must resolve the app.

### 4. LOW — Studio project LISTING is blocked by the PC-control switch
`skills/maker_studio.py:178` — `StudioProjectsSkill.controls_pc = True` gates the
whole skill, but listing projects is pure sensing; only the "open the folder"
action actuates.

- **Scenario:** PC control OFF → "show my studio projects" is refused, though it
  changes nothing on the machine.
- **Fix:** set `controls_pc = False`; inside `execute`, before `open_path`, check
  `settings.safety.pc_control_enabled` — when off, speak the path instead of
  opening it ("It's at ~/MEDO-studio/…; turn PC control on and I'll open it").

### 5. LOW — Studio "open my last …" unreachable from the LLM tool path
`skills/maker_studio.py:194` — the open branch requires
`request.match is not None and "open" in request.text.lower()`; a tool call has
`match=None`, so the model can only ever list.

- **Fix:** add an optional `"open": bool` (or `action`) property to
  `tool_schema` and honour `request.args.get("open")` in `execute`.

### 6. LOW — orphan version dirs when a Studio run crashes mid-way
`studio/projects.py` — `begin_version()` reserves `next_n` and `mkdir`s `vN`
immediately; if the engine crashes before `finalize_version`, `vN/` stays on
disk forever, invisible to `versions()` and never pruned.

- **Fix (cheap):** in `StudioProject.open`/`create`, garbage-collect `v*` dirs
  whose `n` is < `next_n` but absent from `versions[]` (or simply document).

### 7. LOW-MED (hallucination residual) — agents' `done.success` defaults to True
`skills/screen_agent.py` and `skills/browser.py` — `ok = bool(action.get("success", True))`.
A small local model that gives up but **omits** the `success` field (while
`say`="I can't find the button") is still returned as `success=True`.

- **Fix idea:** keep the default (compat), but when `success` is absent AND
  `say` matches `\b(can'?t|cannot|couldn'?t|unable|impossible|fail(ed)?)\b`,
  treat it as failure. Add tests for both directions (an omitted flag with a
  happy `say` must stay success).

### 8. INFO — mic dedup can't split two devices with IDENTICAL 31-char MME names
`voice/audio.py` `_dedup_inputs` — when two *distinct* mics truncate to the same
31-char MME name, they're indistinguishable **by name** at the MME layer; the
first index wins. Inherent to name-keying; the distinguishable-full-name case is
already fixed. No action — documenting so nobody "rediscovers" it.

---

## Verify on a real machine (not fixable from here)
- `schemdraw` / `cadquery` are NOT installed → Studio S2/S3 currently degrade to
  "install X". Install (`pip install -r requirements-studio.txt`), enable
  `studio.enabled`, and run one real end-to-end schematic + one 3D model.
- Long code-generation on qwen3:30b may exceed the LLM client timeout — verify
  `llm` timeout config is generous enough for Studio generation, or set
  `studio.model` to a faster model.
- `CodeBuildSkill` overlaps `MakeAppSkill` (app_builder) on "build a tool that…"
  when BOTH are enabled — registration order decides. Document or disambiguate.

---

## Cool feature proposal — "Watch & Learn" (teach-by-demonstration macros)

MEDO already learns WHERE an app's controls are (`software/knowledge.py` maps +
`locate_in_app`/`navigate_in_app`). The natural next step is learning **flows**:

> **"Watch me — I'll show you how to export."** MEDO records the sequence of
> controls the user activates in the focused app (UIA focus/invoke events, or a
> narrated walkthrough: user clicks + says "step"), and saves it as a named,
> per-app **macro** next to the UI map (`data/software_maps/<app>-macros.json`).
> Later: **"do it like I showed you"** / "export the video like I showed you" →
> MEDO replays the steps via the existing `navigate_in_app` machinery — each
> step re-resolved against the LIVE window, with the same policy gate,
> confirm-before-destructive, and untrusted-content refusal.

Why it fits: reuses the learned-map + app-context foundation; replay is just a
sequence of already-gated single-step navigations; honest by construction (a
step that can't be found stops the macro and says which step failed). This turns
"MEDO knows where things are" into "MEDO knows how you do things" — the CapCut
"edit a video for me" goal — WITHOUT the brittleness of a vision-only agent.
Suggested surface: `learn_flow` ("watch me…"), `run_flow` ("do it like I showed
you"), plus a macros list in the HUD software panel.

---

## Improvements (smaller, high-value)
1. **Wire `engine.iterate` to voice.** The engine supports parametric iteration
   (`StudioEngine.iterate` → vN+1 with history) but NO skill calls it — only
   tests do. Add a follow-up skill/context: after a Studio build, "make the wall
   3 mm" / "add a pull-up on GPIO4" should `iterate` on the LAST project
   (remember it in a session field, like app-context does).
2. **Background Studio builds.** Generation + sandbox execution currently holds
   the turn (minutes on a 30B model). Adopt the `MakeAppSkill` pattern: kick off
   `asyncio.ensure_future`, announce on completion, guard with a busy flag.
3. **HUD Studio previews.** The panel lists projects text-only; S5's spec wanted
   previews. Serve the newest version's `schematic.svg` / a rendered STL thumb
   via a `/studio/file?path=` endpoint (whitelist to `projects_dir`!) and show
   an `<img>` per row.
4. **KiCad netlist export** (S2 optional output) — verify the KiCad CLI from
   live docs before wiring; don't invent flags.
5. **HUD nudge when a Studio lib is missing** — the panel shows DISABLED, but
   could also show "pip install -r requirements-studio.txt" when the skills
   degrade for a missing library.
