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
> ## Round 5 (2026-08-07) — the Macedonian transcript. FIXED.
>
> A live session held almost entirely in Macedonian was almost entirely wrong,
> and the review rounds above had missed all of it because **every round was
> conducted in English**. Five defects, one root cause per layer:
>
> * **The fast path was English-only.** Making verbs, part nouns and app-name
>   patterns existed in English and nowhere else, so "Направи апликација",
>   "може да ми направиш 3D модел", and "отвори стима" reached no skill at all
>   and fell through to the LLM — which narrated, denied, or invented. MEDO
>   told the user it was "not capable" of 3D models with the whole Maker Studio
>   sitting behind it. Fixed with `mk.MAKE`, `_MAKE_APP_MK`, MK patterns on all
>   three Studio domains, and a Macedonian alias set for apps.
> * **Declension broke every spoken launch.** Macedonian glues the definite
>   article onto a borrowed name ("стим" → "стимот" / "стима"), and the `\b`
>   after the alternation matched neither. `mk.NOUN_ENDING` + `mk.undeclined()`
>   now fold the forms back to the stem.
> * **A too-greedy weather pattern** claimed a bare "прогноза", so a request to
>   *build a forecast app* was answered with the current temperature — three
>   times in one transcript. `WeatherSkill.match` now declines build requests.
> * **Nothing ever checked the reply.** "Reply in Macedonian" is a suggestion to
>   a small local model: it answered in English, or welded Latin stems into
>   Cyrillic words ("извежdam", "да го instaliram"). New `core/reply_language.py`
>   scores a reply by script and `Router._repair_language` re-asks once, keeping
>   whichever attempt scores better — so a false positive costs one round and
>   never a worse answer.
> * **A Latin city in a Cyrillic sentence** ("Во Skopje е 34 степени") — the
>   geocoder and the configured default city are both Latin. `mk.to_cyrillic()`
>   reverses the romanisation, and *refuses* when it can't be done cleanly
>   (a "w" or "y" has no Macedonian letter), so a half-converted name is never
>   spoken.
>
> Also: the system prompt now forbids inventing a product to fit misheard
> speech — "О-пен-с-теам" (spoken "open Steam") had produced a confident
> description of "Open Pen Team, a popular accessibility tool", a product that
> does not exist.
>
> Full suite **2322 passed**, ruff clean. Regression tests:
> `tests/test_macedonian_routing.py` (58).
>
> **Known limit, not fixed:** garbled STT that transcribes *confidently* still
> reaches the model, which may act on it ("Opening Obsidian." for an
> unintelligible line). Whisper's own filter drops a segment only when
> `no_speech_prob` is high AND `avg_logprob` low; a confident mis-transcription
> passes both. The prompt rule mitigates it; a real gibberish gate would risk
> rejecting valid Macedonian speech, which is the very thing being fixed here.
>
> ## Round 6 (2026-08-08) — the English transcript. FIXED.
>
> An English session, so none of the round-5 Macedonian work applied. Seven
> defects, each reproduced before it was touched:
>
> * **The app builder lost its spec to whichever skill the answer resembled.**
>   "make me an app" → "What should the app do?" → "I want the app to track the
>   weather in Macedonia…" → *"It's 22 degrees and clear in Jonesboro."* The
>   build was silently dropped. `Router._route_inner` abandoned a reply capture
>   whenever the answer also matched a fast-path skill — right for a narrow
>   question, fatal for an open one, because an app spec names the very things
>   MEDO has skills for. `SkillResult.reply_is_open` now marks a question whose
>   answer is free text; make_app and Maker Studio set it, and the capture wins.
>   (Self-dev's "what should I fix?" is deliberately left narrow: it edits
>   MEDO's own code, so letting a real command escape is the safer default.)
> * **The geocoder answered a country with a village in Louisiana.**
>   `_lookup` asked Open-Meteo for `count=1` and trusted it, but the API leads
>   with fuzzy alternate-name hits: "Macedonia" returned Jonesboro, LA (pop.
>   4587) ahead of two exact matches. It now fetches several and ranks them —
>   exact name, then capital, then population — and a small alias table maps the
>   names people say to the ones the gazetteer files ("Macedonia" → "North
>   Macedonia", renamed 2019).
> * **"Give me a morning brief" took 115 seconds.** Measured: weather 0.9 s,
>   news 8.4 s, **rewrite on qwen3:30b 72.5 s** — a thinking model reformatting
>   sentences that were already final. The rewrite runs on the light tier now
>   (llama3.2:3b, ~9 s) and the two network sections are fetched concurrently
>   instead of one after the other. Measured after: **12.2 s**.
> * **An unrelated fact was announced as the day's schedule.** "Worth
>   remembering: I don't drink coffee." `FactsStore.relevant()` returns its
>   whole store when the store is smaller than the limit — correct for padding
>   an LLM prompt, wrong as a query. With exactly one fact stored it matched
>   everything. New `FactsStore.search()` filters by cosine score and is allowed
>   to return nothing; the briefing uses it.
> * **MEDO read a JSON function schema out loud**, braces, quotes and
>   `"type": "function"` included — the system prompt has forbidden that since
>   round 1, and nothing enforced it. New `core/speech_text.py`:
>   `strip_markup()` runs on every reply before anyone says it, and
>   `looks_like_markup()` triggers one re-ask for a spoken answer (stripping
>   alone leaves the prose around a hole). `FenceFilter` carries fence state
>   across streamed chunks, because sentence-streaming TTS otherwise voices the
>   inside of a code block long before the closing ``` arrives.
> * **An internal tool name was spoken**: "I can locate_in_app weather app…".
>   `leaked_tool_names()` checks a reply against the live registry, so ordinary
>   snake_case in a filename isn't a false positive.
> * **The semantic tier answered a question about MEDO with a security skill.**
>   "What features would you like added to your system?" → *"Lion mode is off."*
>   The tier matched `security_updates` on the shared vocabulary (system, apps,
>   add). Questions ABOUT MEDO now skip the semantic tier entirely and go to the
>   LLM, which is the only thing that can actually answer them.
>
> Full suite **2378 passed**, 1 skipped, ruff clean. Regression tests:
> `tests/test_transcript_round6.py` (56).
>
> **Known limit, not fixed:** the streamed markup guard is best-effort for
> *non-fenced* markup. A bullet or a bare `{` that arrives mid-stream is
> stripped per sentence, but a JSON blob written without a fence can still have
> its first sentence spoken before the reply is whole. Catching that would mean
> buffering the entire reply and giving up sentence-streaming — the latency win
> that makes the LLM path bearable.
>
> ## STILL OPEN — verified, not yet fixed (for the next pass)
>
> **Round 3 cleared #2/#4/#5; round 4 addressed #1 and #3 — this list is
> now CLEAR.** Full suite 2264 passed.
>
> Residual on #1 (documented, not a regression): the streaming paths now
> stop speaking the moment a tool call appears in the stream, so a
> preamble can no longer keep narrating an action that hasn't run. Text
> the model emitted BEFORE the tool call arrives in the same chunk
> sequence is still spoken — eliminating that entirely would mean
> buffering every round to its end, which would undo the
> speak-as-it-generates latency win. The anti-narration system-prompt
> rule is the other half of this mitigation. Revisit only if a real
> transcript still shows a false claim.
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
