# Bug Log

Bugs found and fixed while building the super project (2026-07-03), plus the
honest list of what still isn't perfect. See [[roadmap|Roadmap]] for planned work.

## Fixed during the merge

| # | Bug | Fix |
|---|-----|-----|
| 1 | qwen3:30b's `<think>` reasoning leaked into TTS, conversation memory, and the yes/no confirmation gate (v2 had no stripping at all) | `strip_think()` at the single `chat()` choke point in `llm/client.py`; unterminated-`<think>` truncation handled; tool_calls preserved |
| 2 | 30B unloaded after Ollama's 5-minute default → 15–30 s reload before every delayed reply | `keep_alive: 30m` in every chat payload |
| 3 | `default_model: null` triggered an interactive startup picker — broke one-click launch and headless serving | ships `qwen3:30b`; graceful fallback to first installed model |
| 4 | STT hardcoded `language: "en"` — Macedonian input impossible (v1 regression) | `stt.language: null` = per-utterance auto-detect; nullable in config schema |
| 5 | English `initial_prompt` biased Macedonian decodes once auto-detect was on | prompt nulled by default, documented |
| 6 | LLM returning empty content (truncated mid-think) was answered with the misleading "can't reach my model" | honest `EMPTY_REPLY` fallback |
| 7 | v1's cursor mapping mirrored X unconditionally — double-mirror when the camera already flips (`flip: true`) | mirror is flip-conditional in `vision/pointer.to_screen` |
| 8 | Discrete gestures (point_up→volume up, pinch→lock!) would fire while steering the cursor | discrete map suspended in pointer mode |
| 9 | `pyautogui` in the sidecar would add a 0.1 s pause per cursor move (~10× too slow for 15 fps) and a FAILSAFE corner trap | ctypes `SetCursorPos`/`mouse_event` instead; zero new sidecar deps |
| 10 | `pyautogui.write()` silently drops Cyrillic — "type" was ASCII-only | non-ASCII goes through clipboard paste (pyperclip), clipboard restored after |
| 11 | Polling `/status` for HUD stats would hammer Ollama every 2 s (it lists models per hit) | new cached `GET /sys` with a background psutil/nvidia-smi collector |
| 12 | HUD replies rendered twice when typed (fetch response + SSE `routed`) | chat renders only from SSE |
| 13 | v1 `.env` was dead config — the launcher ran `npm run server`, which never loaded it (num_ctx 8192 vs effective 4096) | one config source (`config.yaml`); num_ctx pinned 4096 (8192 OOMs this GPU) |
| 14 | v1 "100% local" claim was false: browser Web-Speech STT sends audio to Google | whisper is genuinely local |
| 15 | v1 `open_app` built a shell string from user text (injection-shaped) | config-mapped app launch table (v2 design) kept |
| 16 | "forget it" would have deleted every fact containing "it" | vague-pronoun guard in ForgetFactSkill |
| 17 | "what do you remember" could be *stored as a fact* | recall/forget register before remember; remember is start-anchored |
| 18 | Windows volume broken with newer pycaw (`AudioDevice` has no `.Activate`) | 3-way endpoint fallback (session fix, carried over) |
| 19 | "close browser" killed a nonexistent `browser.exe` | image name derived from the launch command (session fix, carried over) |
| 20 | Missing Piper voice crashed `--voice` and took the HUD+API down with it | launcher downloads the voice; TTS optional; servers survive voice-stack failure (session fixes, carried over) |


## Fixed in the evaluator-hardening pass (2026-07-14)

| # | Bug | Fix |
|---|-----|-----|
| 21 | Companion API on `0.0.0.0:8710` was fully unauthenticated — any LAN device could type, screenshot, or power off the PC | bearer-token auth on every endpoint (token minted into `secrets.local.yaml` on first serve); 127.0.0.1 exempt so HUD/sidecar stay zero-config; fails closed; `remote.auth_enabled: false` restores old behavior |
| 22 | Confirmation gate was **English-only** — the bilingual assistant could not confirm or cancel a destructive action in Macedonian ("да" fell through to "ask again", "не"/"откажи" could never cancel) | Cyrillic + Latin-transliteration yes/no sets in `core/safety.py`; Whisper-proof normalization (case, `.!,?…`, whitespace); still whole-reply matching, so "не знам" stays ambiguous and re-asks |
| 23 | Pointer mode was **Windows-only**: the engine's backend was raw user32 (`vision/winmouse.py`), so on macOS/Linux `ctypes.windll` raised and pointer mode silently reported "unavailable on this system" (2026-07-17) | platform dispatcher `vision/mouse.py` → `winmouse` (win32) / new `macmouse` (raw CoreGraphics via ctypes + osascript volume/media; drags post `LeftMouseDragged`) / new `anymouse` (pyautogui, pause + fail-safe disabled). macOS additionally needs the Accessibility permission — the sidecar log now says exactly that instead of nothing moving in silence |
| 24 | "type <Cyrillic>" **silently typed nothing on macOS**: the clipboard-paste fallback pressed Ctrl+V, a no-op on a Mac (paste is Cmd+V). The tests "passed" because they pinned the Windows chord on every OS | platform-aware paste modifier in `TypeTextSkill`; tests now assert the running platform's chords |
| 25 | Window actions and the spoken "command"/"windows" key were **Windows chords on every OS** — Alt+F4, Win+Down, Win+D are dead keys on a Mac, and "press command …" resolved to a `win` key | per-OS `WindowActionSkill.ACTIONS` (Cmd+M / Ctrl+Cmd+F / Cmd+W / F11 / Cmd+Tab on macOS); `KEY_MAP` maps windows/command/cmd to this machine's super key |
| 26 | `python -m voice.wakeword` printed a mic **priority list** as `device #['A25', 'FHD Webcam']` (looked like a broken index) | tester now prints "first available of A25 > FHD Webcam" |
| 27 | Macedonian STT only understood trivial phrases ("Како си?" worked, complex sentences decoded as garbage): Whisper `small` is too weak for mk, auto-detect misheard mk as Bulgarian/Serbian (wrong tokenizer → garbled decode), CUDA math DLLs were missing (CPU int8), and the 1.0 s silence gate cut sentences at mid-thought pauses | `large-v3-turbo` on GPU — pip `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` wheels + DLL-dir registration in stt.py; en/mk detection clamp (one forced re-decode on misdetection); silence gate 1.4 s. A/B on a synthesized complex mk sentence: small → gibberish, turbo → the full sentence (decode 1.5 s, warm load 3.1 s) |
| 28 | "what do you see" answered "I couldn't make anything out" — **moondream decodes to 2-token garbage** (`'?".'`) on EVERY path: valid frames and synthetic images, /api/generate and /api/chat, GPU and CPU, fresh re-pull, Ollama 0.31.1 and 0.32.1. The currently published moondream blobs are simply broken (the old working install was ~1.7 GB; the registry now serves 1.2 GB) | vision model swapped to `qwen2.5vl:3b` (verified: describes a synthetic scene correctly, first try). Bonus: much better text reading for see_bench part markings. Ollama upgraded to 0.32.1 along the way |


## Found by MEDO's own bug reports (2026-09-04)

The first session using `"that was a bug"` (see [[extending|Extending]] and
`skills/bug_report.py`) produced ten reports in half an hour. They collapsed
into three causes:

| # | Bug | Fix |
|---|-----|-----|
| 29 | **A CLI brain has none of MEDO's skills.** `claude-code`/`codex` are handed a flat prompt and read back as text — `_chat_cli` takes no `tools` argument — so "turn on pointer mode", "build me a counter app" and "build me a 3D model" were all answered "I don't have a tool for that here", while the same pointer command hit the fast path and worked one turn later. One turn claimed "Pointer mode off." having called nothing | the tool-brain borrow that already existed for live-info queries now also fires for a request to ACT (`Router._needs_tool_brain`); questions keep the selected brain, which answers them better |
| 30 | **"I have no tool for that" leaked into answering.** The anti-over-offering rule ended "your tools are the complete list of what you can do", so MEDO refused to recommend a perfume or explain how to build a motor driver | the rule is scoped to ACTING, with an explicit line that explaining/advising/recommending need no tool |
| 31 | **Bug reports showed UTC timestamps as if they were local.** `route_events.ts` is SQLite's `datetime('now')`; printed beside the report's local `When:`, a turn answered at 10:55 read "recorded 08:55" | `core.bugreports.local_ts` converts for display; the store keeps UTC |
| 32 | **The 3D modelling and schematic skills did not exist at runtime.** `studio.enabled` defaults to False and `config.yaml` had no `studio:` section at all, so `model_3d` / `design_circuit` / `build_code` / `studio_projects` were never registered — every 3D request routed to nothing and the LLM truthfully said it had no tool | documented `studio:` block in `config.yaml`, enabled; a test asserts the four skills exist iff the switch is on |
| 33 | **Every finished 3D model was reported as a failure.** CadQuery's OpenCascade bindings segfault while UNLOADING on Windows (exit 0xC0000005, reproducible on a bare `import cadquery`), so a complete watertight STL scored `returncode != 0` — and the engine then spent its whole retry budget re-generating a script that was never broken | `Model3DDomain.execute` judges success on a complete, parseable STL; the crash is logged and kept in the record. Timeouts, policy refusals and truncated exports are never salvaged |
| 34 | **Every schematic ran until the sandbox killed it.** schemdraw's matplotlib backend ends a drawing with `fig.show()` + `plt.show()`, which starts the MPL event loop and blocks forever in a subprocess with no display | `MPLBACKEND=Agg` in the sandbox's scrubbed environment — true for any generated code that imports matplotlib. Same script: 60 s timeout -> 1.3 s |
| 35 | **A CLI-agent brain was told to speak when asked for code.** `_flatten_for_cli` appends "Reply with only your spoken answer, as plain text" to every CLI call — the last line the agent reads, so it beat the domain's "Output ONLY the Python code". A phone stand came back as an English paragraph describing one, which was written to `model.py` and executed | `chat(..., for_speech=False)` for callers that want an artifact; the Studio's Brain passes it |
| 36 | **Code generation was killed by the conversation timeout.** `llm.cli_timeout_s` is 180 s so nobody waits three minutes for a spoken answer; a real parametric phone stand is 195 s of genuine code writing, so it was cut off just short — four times, then reported as "I couldn't model that" | `studio.generate_timeout_s` (600 s) for the build path only; voice latency unchanged |
| 37 | **A build took the whole assistant with it.** `_StudioSkill.execute` awaited the engine inside the turn: 514 s measured end to end, during which MEDO could not hear or answer anything | background build + announce on completion, the shape `make_app` already used. A clarifying question is still asked in the turn that prompted it; without an announcer it stays synchronous |
| 38 | **`circuit_help` claimed the Maker Studio phrase the docs tell you to say.** It registers ~270 lines earlier than `design_circuit`, and registration order decides the fast path, so "design a circuit for an ESP32 blinking an LED" got prose and the schematic generator was unreachable | `CircuitSkill.match` stands aside for an explicit drafting verb + circuit noun while `studio.enabled`; wiring QUESTIONS still reach it, and with the Studio off it gives up nothing |
| 39 | **`make_app` missed most natural phrasings.** No room for an adjective between the article and the noun, so "build me a simple counter app", "make a small budgeting app" and "направи ми мала игра" matched nothing — 9 of 16 real phrasings | a bounded modifier slot in both languages; a non-filler modifier also becomes the brief, so "a counter app" builds instead of asking what it should do |
| 40 | **Naming an app named it one letter.** `called (?P<name>[\w -]+?)` is LAZY and the group after it OPTIONAL, so "a tool called budgeter that logs expenses" captured name="b" and dropped the spec entirely — every "called X" phrasing, including the skill docstring's own example | a lookahead anchors the name to the spec keyword or the end of the sentence |
| 41 | **`make_app` still missed a longer description.** The modifier run was capped at two words, so "can you build me a simple camera with filters app?" (four) matched nothing and fell to the LLM, which on a CLI brain has no tools and answers "I don't have the capability to create an app" | run widened to four, guarded by a DETERMINER test — "a note about **the** tool" and "a copy of **that** script" are different sentences, and a determiner inside the run ends the match. Macedonian uses the preposition list instead ("белешка **за** алатката") |
| 42 | **`learn_app` claimed any sentence containing "scan &lt;word&gt;".** "Turn off scan lines" was read as *learn the app called "lines"* and answered "I couldn't find lines open. Open it first, then say 'learn lines'." The verb was matched anywhere in the utterance, but these are imperatives | the three patterns anchor to the start of the utterance behind a polite/wake lead-in (which includes the wake word, since Whisper writes it into the transcript and mishears it as "Middle") |
| 43 | **"map out a plan" was an app called "out a plan".** The app group allows spaces, so the phrasal verb was swallowed whole | negative lookahead on `out` after the verb |
| 44 | **A circuit described BEFORE the noun reached nothing.** "Could you design an RC alternating current circuit?" — the noun is last, after three words of description — matched no pattern and fell to the LLM, which answered with the single sentence "Detailed design and explanations for an RC alternating current circuit." | `design_circuit` takes a determiner-guarded modifier run, and `_description` reads the brief from whichever half of the sentence carries it ("an RC ALTERNATING CURRENT circuit" or "a circuit FOR an ESP32") |
| 45 | **A bug report could not tell you the process was stale.** Python imports its modules at launch, so a fix on disk is not in a running MEDO — and the report's revision line reads identically either way when the change isn't committed. Two full rounds of reports were spent on routing that was already fixed | the environment block now carries **Running code**: process start vs. the newest mtime under the source directories, and says outright "This process is running OLD CODE; restart MEDO" when it is behind. Docs and tests are excluded — editing those changes nothing about a running assistant |
| 46 | **A failed printability check never got a retry.** The engine broke out of its loop as soon as the code RAN, then judged the artifact afterwards — so a rocket whose mesh had 2 open edges was reported as "I tried to model that but couldn't" on the FIRST attempt, with three attempts still unused. A non-watertight mesh is the commonest way a CAD script "succeeds", and exactly what the brain can fix once told | `check` moved inside the loop: a 'fail' advisory is fed back as the error and the next attempt gets it. Still bounded by `max_retries`, and a mesh that is broken at the end is still reported broken |

## Hallucinations found in one Macedonian exchange (2026-09-06)

Four turns, four different ways to sound like an answer without being one
(`bugs/bug-2026-09-06-112349-tvoja-boga.md`).

| # | Bug | Fix |
|---|-----|-----|
| 47 | **Every skill answered in English once the LLM tier dispatched it.** `dispatch_tool` rebuilds a stand-in utterance out of the tool name and the model's arguments — `"smalltalk"`, `"weather Skopje"` — and that stand-in is always English. The 28 skill files that choose a reply language with `mk.is_cyrillic(request.text)` were sniffing it, so a Macedonian question answered on the fast path came back in Macedonian and the same question answered through the model came back in English | `SkillRequest.utterance` carries what the user actually said (the router puts it in the tool context); all 28 sniff that instead. `text` still holds the stand-in, so pattern-driven skills keep finding their groups |
| 48 | **Small talk answered things that were not small talk.** `_category()` fell back to `"greet"` when no pattern matched, which is fine on the fast path (a pattern DID match to get there) and wrong via the model, which picks the tool for anything conversational. Asked "Твоја бога.", MEDO replied "At your service." — a reply to a greeting that never happened, in the wrong language | no match now returns `""` and the skill declines; the router already treats an unsuccessful tool as needing synthesis, so the model answers what was really said |
| 49 | **A placeholder was spoken as an answer.** Asked in Macedonian where the village Другово is, MEDO thought for 24 seconds and said "Nil." The tier repairs leaked tool-JSON, markdown and wrong-language replies, but a bare code token passed straight through — and it sounds like a fact, so the user cannot tell it from one | `_is_non_answer` treats a lone `nil`/`null`/`n/a`/`NaN`/`undefined`, or punctuation alone, as empty and falls back to the honest apology. Narrow on purpose: "No.", "Nothing." and "Unknown, sorry." are real answers and are left alone |
| 50 | **The apology itself was English-only.** `EMPTY_REPLY` is one string, so the fallback for a Macedonian turn was a language violation on top of a failure | `EMPTY_REPLY_BY_LANG` keyed by the turn's language, English as the floor |
| 51 | **A place name was translated instead of repeated.** "каде се наоѓа селото Другого" came back as "Слеза во **Other** се наоѓа во главната страница" — the name rendered as its dictionary meaning. `reply_language` could not catch it: "Other" is capitalised, so `_is_proper_noun_shaped` exempts it as a brand name (which is what keeps "Отвори го Steam" legal) | the non-English language instruction now says a name is not a word to translate, cites this exact failure, and to say so when the place or person is unknown. The proper-noun exemption is deliberately left alone |

| 52 | **A 3B model was driving the tool loop for every Macedonian turn.** `auto_model` downgrades a "simple" turn to `llm.fast_model` (`llama3.2:3b`), and `pick_model_tier` judges simplicity by length and English/Macedonian keywords — so a short Macedonian sentence is always "simple". Live: "Твоја бога." went to llama3.2:3b, which called `power(action=lock)` and **locked the screen**; "каде се наоѓа селото Другово" became `locate_in_app(app='Other App')`, the place name translated into an app that does not exist. These are not reasoning failures the strong model shares — the 3B is not reading the sentence, and it still gets the whole tool list | `pick_model_tier` returns "strong" for Cyrillic. On the same turns afterwards: no downgrade, no lock, and `web_search(query='Другово')` — the real name. Costs the 30B's load time on a Macedonian turn that misses the fast path; a multilingual `fast_model` would buy the tier back |
| 53 | **Locking the screen was the one power action with no confirmation.** That exemption belongs to the fast path, which only arrives by matching the words "lock the screen" — but the model can choose the tool too, and did, from an utterance containing no verb at all | `PowerSkill._asked_to_lock` requires the words (en + mk) or a regex match; anything else asks first. Sleep/shutdown/restart are unchanged. Note `SkillRequest.utterance` returns "" rather than the stand-in on the tool path — otherwise the check read "power lock", the synthetic text of the very call it was judging, and locked anyway |

| 54 | **A bare Macedonian "Паузирај." turned the volume down.** The MK media pattern needs a following noun (`паузирај ја песната`), and the bare-transport-verb pattern that catches English "pause"/"resume" on its own had no Macedonian twin — so "Паузирај." matched nothing, fell to the LLM and came back as `media_volume_down`. `_action` already knew the verbs; only the pattern was missing | a bare `паузирај`/`продолжи`/`следна`/`претходна` pattern, anchored to the whole utterance like the English one. `запри`/`стопирај` are deliberately excluded — bare, they are as likely to mean "stop talking" |
| 55 | **A garbled syllable armed a shutdown.** "Позе." → "Are you sure you want to shut down?"; "Паузирај а музиката, **не** да гузга сис компјутерот" ("pause the music, NOT the computer") → "Are you sure you want to put the machine to sleep?" — told in the same breath not to touch the computer. The confirmation reads like a backstop but is not one: it leaves the user a single misheard "yes"/"да" from a real shutdown | `PowerSkill._asked_for` generalises the lock guard to every action: the utterance must name it (en + mk) or a regex must have matched, otherwise the call is **refused outright** rather than confirmed. Genuine requests still reach their confirmation unchanged |
| 56 | **MEDO read a markdown essay aloud, for six minutes.** Asked in Macedonian how to memorise squares and cubes, it answered in English with markdown tables, `$11^2 = 100 + 20(1) + 1 = 121$`, `---` rules, a bullet list and a brain emoji — 361 s of it. `_repair_markup` never fired because `looks_like_markup` only knew code fences and JSON blobs | the detector also recognises table rows, LaTeX spans, ATX headings, horizontal rules and multi-line lists. Money (`$5`, `$5 to $10`) and a stray prose dash are explicitly not LaTeX or a rule |


## Known limitations (open, by design or deferred)

- **see_bench** recognition quality depends on lighting and moondream — a
  1.7 B vision model, not a parts database. Part-number OCR is best-effort:
  without the Tesseract engine (`winget install UB-Mannheim.TesseractOCR`,
  needs admin — the silent install was refused a UAC prompt) answers simply
  omit the markings line.
- Macedonian replies are **spoken with an English Piper voice** (no mk voice
  exists for Piper). Text in the HUD is correct. → [[roadmap|Roadmap]]
- ~~Wake word is pretrained "hey jarvis", not "MEDO".~~ *Fixed 2026-07-24:* custom `hey_medo.onnx`, trained on the owner's real voice, fires on both "medo" and "hey medo".
- Brightness: external monitors don't expose WMI brightness — graceful error.
- moondream↔qwen3 VRAM contention on 12 GB: one reload after vision calls.
- ~~Companion API has token auth but **no TLS**~~ — TLS added 2026-08-25
  (`remote.tls_enabled`, self-signed on first run), off by default. The HUD
  and vision sidecar are still plaintext on loopback; never forward either.
  port 8710.
- v1's two-finger chat scroll and open-palm-wake/fist-barge-in gestures are
  not carried over yet (need voice-loop integration). → [[roadmap|Roadmap]]

## Gestures fired on ambiguous poses; some gestures never dispatched

**Symptom.** Two complaints that turned out to share a cause. Gestures fired
while a hand was moving *between* poses — closing a fist would flash through
the horns pose and fire a zoom. And several defined gestures "never applied":
you could make them perfectly and nothing happened.

**Cause.** Recognition was an if-chain of loose heuristics. A finger counted as
extended if its tip was farther from the wrist than its middle joint, which is
true for every pose between fully open and fully closed — so a hand in transit
satisfied real branches by accident. Worse, the chain's order was the only
thing disambiguating overlapping poses, and that order was implicit.

The second half was simpler and more embarrassing: some gestures existed in the
classifier but were bound to no action at all. They classified correctly and
then did nothing, which from the far side of a camera is indistinguishable from
broken recognition.

**Fix.** Replaced the heuristics with an explicit `GestureSignature` table —
one entry per gesture, an exact state for all five digits, plus geometric
constraints. A pose matching no signature is UNKNOWN; there is deliberately no
"nearest gesture" fallback. Added a confidence score (how deep inside its band
each *required* finger sits, taking the minimum), a hold of 3 consecutive
frames, and hysteresis so a held pose is one event rather than a stream.

For the dispatch half: `dispatch_report()` maps every signature to where it
goes, `python -m vision.gestures --list` prints it, and the sidecar logs it at
startup. Anything UNBOUND is now visible instead of silently inert. A test
fails the build if any gesture is defined but unreachable.

**Also caught while fixing it:** in permissive (non-strict) mode a resolved
ambiguous finger scored zero confidence, so the mode that exists to be lenient
rejected everything.
