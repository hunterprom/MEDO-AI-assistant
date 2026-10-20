<p align="center">
  <img src="docs/img/medo-logo.png" alt="MEDO" width="128" height="111">
</p>

# MEDO — the detailed guide

This is the long-form reference. For the two-minute version see
[`README.md`](README.md); for the architecture notes, runbook, bug log and
decision records, open the Obsidian vault in [`docs/`](docs/).

MEDO is a **local-first, two-language (default English/Macedonian) voice +
gesture desktop assistant** in one Python codebase: a custom "medo" wake word
trained on my own voice, Whisper ears, a swappable LLM brain, neural voices in
either language, MediaPipe hand-tracking, a sci-fi HUD, a desktop presence
sphere — and a bench of expert "council" agents it can put a question to, or turn
loose on its *own* answers.

---

## Architecture at a glance

```
mic ──► openWakeWord ("medo") ──► record ──► faster-whisper (detect between
                                                     the 2 ACTIVE languages)
                                                        │
     HUD :8730  +  corner sphere  ◄── EventBus ◄────────┤   (both SSE observers)
                                                        ▼
  watch app ──► companion API :8710 /ask ──────► Intent Router
  vision sidecar gestures ──► POST /ask ────────►   │
                                                    ├─ FAST: regex skill (<1 ms)
                                                    ├─ SEMANTIC: reach a skill by
                                                    │   MEANING (local embeddings,
                                                    │   + learns from real usage)
                                                    └─ LLM: tool loop (strip_think,
                                                        ≤4 rounds, confirmation
                                                        gate, "let me think" filler)
                                                        │
                       local neural voice (Piper/Kokoro, sherpa-onnx) ──► speakers
```

Three processes, isolated on purpose: the main app (`.venv`), the MediaPipe
vision sidecar (`.venv-vision` — a quarantined `numpy<2` pin that talks only
HTTP), and Ollama. Every input — voice, watch, HUD text, hand gesture —
converges on the same Intent Router; every UI renders from the same event
stream. Full details in [`docs/architecture.md`](docs/architecture.md).

### The three routing tiers

Every utterance is routed through up to three tiers, and which one handled it is
recorded for the performance table:

1. **FAST** — a registered skill's regex matched. Deterministic, no model, <1 ms.
   ~30 commands: time, timers, notes, volume, media, apps, files, screenshots,
   windows, typing, clipboard, brightness, power.
2. **SEMANTIC** — no regex matched, but the utterance reaches a query-style skill
   by *meaning*. The utterance is embedded once (local `nomic-embed-text`) and
   matched against each opted-in skill's example phrases; an ambiguous near-tie
   is **declined on purpose** so it falls through rather than guess. With
   **adaptive route memory** (opt-in), a phrasing the brain resolved to a single
   safe query skill is remembered, so the same wording shortcuts straight to the
   skill next time — the middle tier grows from your own usage, fully offline.
   Destructive/actuation skills are **never** reachable here (a safety invariant
   re-asserted at the dispatch decision), and a learned route still passes the
   confirmation and PC-control gates.
3. **LLM** — the brain, in a bounded tool loop (≤4 rounds), with `</think>`
   reasoning stripped before anything is spoken, a spoken confirmation gate for
   destructive tools, and a "let me think" filler when a CLI agent is slow.

---

## What it does

### Voice
A **custom "medo" wake word** (openWakeWord, retrained on my own captured voice —
fires on "medo" and "hey medo") → faster-whisper → Intent Router → **neural
TTS**. Replies are spoken in the language they were asked in, by that language's
own neural voice — and a non-English reply is never read aloud by the English
voice. Fifteen of the sixteen voices synthesise **on the CPU** through
`sherpa-onnx` (Piper and Kokoro models, ~150–300 ms, zero VRAM, Apache-2.0, so
they work offline too). **Macedonian is the exception**: no permissively
licensed local mk voice exists yet, so it stays on edge-tts (the app trusts the
OS certificate store via `truststore`, so it works behind a TLS-intercepting
proxy). `tts.allow_cloud: false` shows those replies instead of speaking them,
and `tts.voices` re-picks any language's voice without touching code.
→ [`docs/voice.md`](docs/voice.md)

**Two-language mode**: MEDO runs with **at most two active languages** at a time
(default **English + Macedonian**) and constrains per-utterance detection to that
pair — two known candidates are far more reliable than open-ended language ID.
The pair is configurable (`languages.active`) with a picker on the HUD CONFIG
screen; `detection: fixed` forces one language when you only speak one.

**Barge-in**: say the wake word (or tap the HUD mic) while MEDO talks to cut it
off — decoupled from wake sensitivity (`audio.barge_wake_frames`) so a sensitive
wake word doesn't make MEDO cut *itself* off mid-sentence. The mic is a
**priority list** (`audio.input_device`) — headset first, webcam fallback —
hot-swapped within ~2 s and switchable live from the HUD.

### Brain — five interchangeable providers
Switchable live from the HUD: **Ollama** (local, default `qwen3:30b`), **GPT /
any OpenAI-compatible API**, the **Claude API**, and two local CLI agents —
**Claude Code** and **Codex** (their own login, no key in MEDO). `</think>`
reasoning is stripped before anything is spoken or remembered. When a CLI agent
is slow, MEDO speaks a **"let me think" filler** so the gap doesn't read as a
freeze.

### The council — an expert per major
A bench of specialist personas MEDO can put a question to. A discipline is a
**system prompt**, not a process, so "convening" three specialists costs three
prompts on the model you already have loaded — no extra processes on Ollama.

- **"ask the electrical engineer about grounding"** → one specialist answers,
  named out loud so you know who spoke.
- **"convene the council on battery sizing"** → MEDO picks the two or three
  specialists whose field the question actually touches (a deterministic
  word-list, no extra round-trip to decide), asks them in parallel, and
  synthesizes one spoken answer.
- **"who's on the council?" / "do you have a lawyer?"** → a registry read, no LLM.

The built-in bench (18 majors): electrical, robotics, mechanical, physics,
quantum, mathematics, software, law, finance, economics — plus **marketing,
design, data science, cybersecurity, chemistry, medicine, biology and writing**.
Add your own without touching code via `council.extra` in `config.yaml`; each is
a title, a prompt, and the subject words that route a question to it (English
and Macedonian). See [`core/council.py`](core/council.py).

### Second opinion — the council red-teams MEDO's *own* answer
Say **"second opinion"** / **"are you sure?"** / **"poke holes in that"** /
**"fact-check that"** (or the Macedonian *"дај второ мислење"* / *"сигурен ли
си"*) right after MEDO answers, and it hands its **own last answer** — and the
question that produced it — to the relevant specialists as a hostile panel. They
don't re-answer; for each claim they tag it **SUPPORTED / UNSUPPORTED / WRONG /
CANNOT_VERIFY** with a one-clause reason.

The confidence verdict is then computed **in Python** from those tags, not by a
model: WRONG dominates, any UNSUPPORTED makes it shaky, all-SUPPORTED is solid,
and anything unparseable or silent degrades to *"treat as unverified"* — never to
false confidence. MEDO speaks a headline-first verdict naming the single most
damaging flagged claim, then offers the full per-claim breakdown (read back from
cache, no second model call). It's the one council feature that **removes** a
generative surface instead of adding one. See
[`skills/second_opinion.py`](skills/second_opinion.py); toggle
`council.second_opinion`.

### MCP — plug in any application
Declare servers in `config.yaml → mcp.servers` (stdio command or Streamable HTTP
URL) and every tool they expose becomes a MEDO skill automatically. A curated set
ships **disabled by default** — GitHub, Google Maps, Google Drive,
sequential-thinking, filesystem, memory, Brave Search. Flip one on and drop its
secret into the git-ignored `secrets.local.yaml` under `mcp_env:` — injected into
the environment, never written to `config.yaml`.

### Software connectors — control local apps by command (not MCP)

Beyond MCP, MEDO can control other desktop apps by **direct local mechanisms** —
"play/pause", "minimize", "new tab", "close Notepad". A connector declares an app
and its actions; each becomes a routable skill gated by the policy engine, over a
robustness ladder (native API/CLI → app hotkeys → accessibility automation).
Sending or posting **always confirms first**, and only your own voice/text can
trigger control — never content from a document or another app. Off by default
(`software.enabled`) — flip it on without touching config from the HUD **CONFIG**
tab: an **App control** master toggle plus a **Software control** panel with a
per-connector switch, live availability, and the commands each app understands.
Adding an app is a new connector file, not a core edit. See
[`docs/software-connectors.md`](docs/software-connectors.md).

### Make things — documents, presentations, spreadsheets, apps
- **"make me a five-page document about X"**, **"create a presentation on solar
  batteries"**, **"make a spreadsheet of the team roster"** → MEDO composes the
  content with its model and renders it to Word (`.docx`), PowerPoint (`.pptx`),
  or a self-contained web page / slide deck, saved under `~/Documents/MEDO`. It
  reads a length hint ("five-page", "detailed", "short") and sizes the content to
  match. Cells are always written as literal text — a value starting with `=` is
  never a live spreadsheet formula.
- **Projects**: "start a project to build the robot dog", "add a task…", "how's
  the robot-dog project going" — a small sqlite-backed store, plus "plan the
  project" to break a goal into ordered tasks.
- **App builder**: "make an app that tracks my water intake" → a fresh folder
  under the configured `apps_dir`, scaffolded by the coding agent (files only).

### Self-programming (Lion mode, opt-in)
With `self_dev.enabled` on and **Lion mode** active, "work on your own code: fix
the bug where …" drives a coding CLI agent to make the change in an **isolated
git worktree** on a branch, gated by the test + lint suite, and **never** merged
into the live branch (nor does MEDO restart itself) until you approve it in the
HUD. Layered safety: worktree isolation, a safelist/denylist enforced on the
*actual* changed files, the human apply-approval, and a **sandboxed gate** — the
test/lint run happens in a scrubbed, HOME/TEMP-isolated environment (no secrets
in-env; a `~`-relative write can't reach the live repo), with an optional
`self_dev.sandbox_cmd` wrapper for true OS isolation. See
[`core/self_dev.py`](core/self_dev.py).

### Search, browse, remember, see
- **Search where you actually search**: "search drone motors on YouTube", "find
  Half-Life on Steam" open that site's own results — ~35 sites built in, extend
  `skills.sites` without code.
- **Drives the browser, not just opens it**: "click sign in", "read the page",
  "what can I click" — real Chrome through Playwright, working on the **DOM** so
  window size/zoom/theme don't matter. "On the site, …" hands a bounded
  multi-step task to the brain (spoken yes first, capped at `browser.max_steps`,
  `blocked_domains` re-checked after every redirect).
- **Bilingual commands**, not just bilingual chat: "отвори хром", "барај мачки на
  јутјуб", "најди ја датотеката извештај" — answered in the language asked.
- **Memory + RAG**: "remember that…" facts recalled **semantically**, and "what
  do my documents say about the lease?" searches whitelisted .txt/.md/.pdf by
  meaning, quoting the source file.
- **Pointer mode** (webcam): index finger drives the cursor · 🤏 pinch = drag/click
  · ✌ scroll · 🤘 zoom · three fingers = right-click · ✊ held ≈1 s = exit. Boots OFF.
- **Sight**: "what do you see" / "read my screen" → local **qwen2.5-VL** (Ollama).

### Routines — chain skills without writing code

A `.yaml` file in `routines/` becomes a skill. Say the trigger phrase, let it
fire on a schedule, or let the model call it as a tool:

```yaml
name: morning briefing
trigger:
  phrase: ["good morning", "добро утро"]
  at: "07:30"
  days: [mon, tue, wed, thu, fri]
steps:
  - skill: datetime
  - skill: weather
    args: {city: "{city}"}
  - skill: news
params: {city: Skopje}
```

Three examples ship in `routines/`. Validate yours with
`python -m core.composer --check`. A routine **cannot** call a skill that needs
confirmation — MEDO refuses to load it and names the step, because a prompt in
the middle of a chain has nobody to answer it.

### Cowork — hand over a whole task (off by default)

A routine is a chain you wrote in advance. **Cowork** is the one you delegate on
the spot:

> **"cowork: tell me the date, then start recording, then save the clip"**
> "Here's my plan: 1. say the date  2. start recording in OBS  3. save the
> clip. Shall I go ahead?" · **"skip step 3"** · **"yes"**

Bounded on purpose, because a local brain turned loose on an open-ended
multi-step task fails unpredictably — and a failure halfway through, in front of
people, is what they remember:

1. **Plan first, out loud.** Nothing runs before you approve it, and you can
   edit it by voice ("skip step 3", "change step 2 city to Ohrid").
2. **Only real capabilities.** Every step is resolved against the live registry
   — skills, connectors, MEDO Link devices. A step MEDO can't do stays in the
   plan carrying the reason instead of being swapped for the nearest thing that
   exists. Capped at six steps; a bigger task is refused, not truncated.
3. **Approval isn't a blank cheque.** Each step is re-checked at the policy gate
   when it runs, and the high-impact ones (power, run a command, write files,
   drive a device) ask again by name.
4. **Stop on failure.** MEDO says where it stopped and offers retry / skip /
   stop. It never invents another way to finish. "Stop" aborts at any point and
   reports what did and didn't happen.

The plan **locks** at approval, so nothing MEDO reads mid-task — a file, a web
page, a device's reply — can add a step. `"save that as a demo prep routine"`
turns an approved plan into a real `routines/*.yaml`, registered on the spot.

Planning reads the whole capability catalogue in one pass of the strong brain —
**60–120 s** on a 3060 with `qwen3:30b` — so MEDO says *"let me work out a
plan"* before the wait. Worth it for a task worth delegating; the wrong way to
ask for the time.

Switch it on with `cowork.enabled: true`. Full model, limits and rationale:
**[`docs/cowork.md`](docs/cowork.md)**.

### Remembering past conversations

At the end of a session the local model writes three or four sentences about
what happened, embeds them, and keeps *those* — not the transcript. So:

> **"What did we talk about yesterday?"** · **"За што зборувавме вчера?"**
> "Yesterday, you were planning the roof repair and we worked out that the
> guttering needs replacing before winter."

Summaries are pruned after 90 days unless pinned (`memory.episode_retention_days`).
A session too short to be a conversation is not summarised at all. Switch the
whole thing off with `memory.episodes_enabled: false`.

### Interrupting to correct

Cut in while MEDO is answering and say *"actually, make it shorter"* — it
re-answers the **original** question with the correction folded in, rather than
treating those four words as a new request. Works on an explicit correction
("no, I meant the weather" / "не, мислев на времето") without interrupting too.

### Speaking first — the proactive layer (off by default)

`proactive.enabled` lets MEDO start a conversation: the battery crossing a
threshold, a time-of-day nudge. Three rules it never breaks — never while you
are mid-conversation (it waits, then says it), never inside the do-not-disturb
window (`quiet_from` / `quiet_until`, which correctly spans midnight), never
twice inside a cooldown. Every unprompted line is counted:
`python -m core.metrics --proactive`.

### The history dashboard

`http://localhost:8710/dashboard` — every turn MEDO has routed, searchable
across both the question and the answer, filterable by routing tier, with a
re-run button. Read-only, behind the same token gate as the rest of the
companion API, and re-run goes through the ordinary `/ask` so a destructive
request still meets its confirmation.

### "That was a bug" — reporting a fault by saying so

A developer feature. The moment MEDO gets something wrong, say so:

```
You:  search up the Mazda RX-7
MEDO: The Mazda RX-7 is a Japanese sports car built around...
You:  that was a bug, it was supposed to open the browser
MEDO: Saved that as a bug report — the first one in the bugs folder.
```

`bugs/bug-<timestamp>-<what-you-said>.md` now holds the turn that went wrong —
what you asked, what MEDO answered, which tier routed it, which skill it
picked, how much of the sentence the pattern matched, how long it took, the
three turns of run-up, and the source file to go and look in. Hand that file to
whoever is fixing MEDO. Nothing is typed and nothing is reconstructed from
memory a day later, which is the whole point: the evidence is gathered while it
still exists.

The reason goes in the same breath if you have one ("...it should have opened
the browser"); a bare "that was a bug" is enough on its own.

```bash
python -m core.bugreports              # what has been saved
python -m core.bugreports --collect    # all of them as ONE document to hand over
```

Macedonian works the same: "тоа беше баг", "пријави грешка", "запиши го ова
како баг". The trigger word is mandatory — "that's wrong" and "no, I meant..."
belong to the correction detector that measures fast-path misfires, and
claiming them here would corrupt that statistic. `bugs/` is gitignored: a
report quotes what was actually said in the room, so sharing one is a decision
you make deliberately, per file.

### MEDO as an MCP server

The mirror of `mcp.servers`: other local AI tools can call MEDO's skills.

```json
{"mcpServers": {"medo": {
  "command": "…/MEDO-main/.venv/Scripts/python.exe",
  "args": ["-m", "mcp_servers.skills_server"]
}}}
```

**Read-only skills only.** Fourteen are offered — the weather, the news, the
council, the resistor decoder. Anything that writes to your memory, reveals the
screen or camera, reads back your conversation, or enumerates the machine is
withheld, whatever its capability flags say. `--list` shows both sides of that
line. Widen it one skill at a time with `mcp.server_expose`.

### The interfaces
- **HUD** (<http://localhost:8730>): three tabs — **CORE** (live diagnostics, the
  orb with ~300 real folder dots, and the chat side panel), **CLUSTER** (a
  cosmic-web sphere per capability domain, built from the live skill registry so
  it stays truthful — the council's specialists are its stars), and **CONFIG**
  (provider + key, mic picker, spoken-language picker, themes, MCP status) — all
  over server-sent events, zero build step. Installable as a PWA.
- **Desktop presence sphere** (`ui/overlay.py`): a small always-on-top CORE
  sphere pinned to a screen corner; shows state and bubbles up the last exchange;
  click to talk without the wake word, double-click to open the HUD. Its own
  process, so a UI crash can't take the assistant down.
- **Watch app** (`watch/`, Wear OS): one tap on "Pair with MEDO" finds the PC by
  UDP broadcast and swaps a 6-digit on-screen code for the auth token.
- **Plugins**: drop a `.py` in `plugins/` → it works by voice AND as an LLM tool.

---

## Security model

MEDO is **local-first and LAN-only** — these ports must never be forwarded
beyond your LAN. Within that boundary the trust model is:

- **TLS is available** (`remote.tls_enabled`). MEDO mints a self-signed
  certificate on first serve covering `localhost` and this machine's addresses;
  set `remote.tls_certfile`/`tls_keyfile` to use your own instead. It is **off
  by default** because turning it on changes the URL every paired client uses —
  but on any shared network it is worth the re-pair, since without it the bearer
  token crosses the wire in the clear on every request. `GET /tls` returns the
  certificate's SHA-256 so a client can pin it, which is what buys identity as
  well as privacy from a self-signed certificate.
- **Requests are rate-limited per caller** — per token, or per address for an
  unauthenticated one, so grinding token guesses doesn't get a fresh quota per
  guess (`remote.rate_limit_per_minute`, loopback exempt). Every refusal — bad
  token, missing token, throttled, cross-origin — is recorded:
  `python -m core.metrics --auth`. The rejected token itself is never stored.
- **Companion API (8710)** requires a bearer token for LAN clients (minted into
  `secrets.local.yaml` on first serve; loopback is exempt so the HUD keeps its
  zero-config startup). A **same-site Origin check** refuses any cross-origin
  browser request — even one arriving over 127.0.0.1 — which closes the "a
  website you're visiting drives your local assistant" drive-by; a domain `Host`
  header (DNS rebinding) is refused, and CORS echoes only the request's own
  allowed origin, never a wildcard. Add a trusted remote-HUD origin to
  `remote.allowed_origins`.
- **Switching the cloud provider's base URL** requires the API key to be
  re-supplied, so a stored key is never forwarded to a URL you didn't just
  authorize.
- **`/open`** reveals a file's folder rather than executing the file.
- **The HUD** serves the companion-API token to a **loopback** request only, so a
  LAN-exposed HUD can't hand the token to any device that loads the page.
- **The vision sidecar (8731)** binds `127.0.0.1` and refuses cross-site browser
  requests, so the webcam stream and the mouse-control toggle aren't exposed to
  the LAN or to a drive-by page.
- **Secrets** (`secrets.local.yaml`) are written owner-only (0600).
- **No destructive action can be reached without a spoken yes**, from any tier.
  The confirmation gate lives in the router, not in each skill, so a skill that
  forgets to check is held anyway. A scheduled routine can't answer its own
  prompt, and an unattended source is refused rather than left armed.
  `tests/test_destructive_routes.py` drives every route — regex, semantic, LLM
  tool call, MCP tool, plugin, connector, paired device, the companion API, a
  routine — at a deliberately careless skill and fails if any of them reaches it.
- **Self-programming** runs the coding agent in an isolated worktree and its
  test/lint gate in a scrubbed, HOME/TEMP-isolated environment **with no
  outbound network** (loopback stays open for the suite's own test servers),
  behind a denylist, a diff-size cap, and an approval recorded **on disk and
  bound to the exact diff** a human read. A merge that then breaks the suite
  rolls itself back. `self_dev.sandbox_cmd` adds real OS isolation; the
  in-process network guard is a speed bump, not containment, and says so.
- **MEDO as an MCP server** (`mcp_servers/skills_server.py`) offers only
  read-only skills, and "read-only" there is stricter than the capability flags:
  anything that writes to your memory, reveals the screen or camera, reads back
  your conversation, or enumerates the machine is withheld.

### The central policy engine

Those guards are unified behind **one deny-by-default policy engine**
(`security/`): every side-effectful action routes through
`PolicyEngine.check() → allow | confirm | deny` (pure, fail-closed), over a
capability vocabulary each skill/plugin/device declares. Additive and
config-gated, it adds:

- **Trust boundary** — documents, web pages, tool output and memory are DATA,
  never instructions: wrapped as untrusted before the model sees them, and a
  **post-LLM action gate** re-judges any high-impact action induced by that
  content — an injected *"delete my files"* is confirmed or denied, never
  auto-run.
- **Owner voice** *(optional)* — high-impact actions can require the enrolled
  owner's voice (fully local; only a voiceprint embedding is stored, never audio).
- **Plugin approval** *(optional)* — a new/changed plugin's capabilities are read
  statically and its code isn't imported until you approve it.
- **Cloud egress** — a cloud brain gets your question, not your local
  documents/memory, unless you opt that brain in.
- **Audit** *(optional)* — a hash-chained, tamper-evident log of security events
  (`python -m security.audit --report`).
- **Developer mode** — a HUD toggle (CONFIG → Developer mode) turns the whole
  layer off for local dev; **session-only** (resets to secure on restart), with a
  loud on-screen banner. Never ship it on.

See [`docs/security.md`](docs/security.md) for the full threat model and honest
limits.

---

## Performance

Every routed request is measured on the machine itself (`metrics` table in the
sqlite DB, `logging.routing_stats: true`); run `python -m core.metrics --report`
to regenerate the table. Fast-path commands never touch the model, which is why
they sit three orders of magnitude below the LLM path; the SEMANTIC tier sits in
between — one embed, no LLM round-trip.

---

## Install & first run

### Hardware

MEDO runs everything locally, so what you get depends on what the machine can
hold. Nothing below is a hard gate except the disk: a smaller brain is picked
automatically on a smaller GPU.

| | Minimum | Comfortable | What the difference buys |
|---|---|---|---|
| **CPU** | 4 cores | 6+ cores | Wake word, STT and TTS all run while the model generates |
| **RAM** | 8 GB | 16-32 GB | Room for the model, Whisper and the browser at once |
| **GPU** | none | 12 GB VRAM | With none, the brain runs on CPU: correct, and slow. 8 GB fits a 7-8B model; 12 GB fits the 30B partly offloaded; 24 GB fits it whole |
| **Disk** | 30 GB free | 50 GB | The models are ~24 GB of it. This one IS a gate |
| **Mic** | any | a headset or a decent USB mic | Wake-word accuracy is mostly a microphone problem |
| **Webcam** | none | any 720p | Only needed for gestures, pointer mode and "what do you see" |

Reference: the machine MEDO is developed on is a Ryzen 5 5600, 32 GB RAM,
RTX 3060 12 GB — qwen3:30b at `num_ctx` 4096, first audio in about two seconds.

### Prerequisites

- **Windows 10/11**, or **macOS/Linux**.
- **Python 3.12** on PATH — `py -3.12` (Windows) or `python3.12` (macOS/Linux).
- A package manager for the automatic app installs: **winget** (ships with
  Windows 11) or **Homebrew** (macOS/Linux). Without one MEDO still runs — the
  launcher prints the Ollama/Obsidian download links and continues.
- **~30 GB free disk** for the local models; an NVIDIA GPU helps but the 30B
  brain runs on CPU too (slower). On Windows the model store defaults to
  `D:\OllamaModels` — change `OLLAMA_MODELS` at the top of `run.bat` if you have
  no D: drive.

### 1. Get the code

```
git clone https://github.com/hunterprom/MEDO.git
cd MEDO
```

### 2. First launch — installs everything

Double-click **`run.bat`** (Windows) or **`run.command`** (macOS/Linux). The
**first** run is a one-time setup (~24 GB — be patient) that, in order:

1. installs **Ollama** and **Obsidian** if missing (winget / Homebrew);
2. pulls the four local models MEDO uses, smallest first so it's usable quickly
   (see the table below);
3. creates the two virtualenvs (`.venv`, `.venv-vision`) and installs the pip deps;
4. downloads the wake-word and Piper voice models;
5. starts Ollama (with `OLLAMA_MODELS=D:\OllamaModels`, `GGML_CUDA_NO_PINNED=1`),
   launches the vision sidecar, and opens the HUD at <http://localhost:8730>.

| Model | Size | Used for |
|-------|------|----------|
| `nomic-embed-text` | ~275 MB | semantic routing + fact memory |
| `llama3.2:3b` | ~2 GB | fast fallback brain |
| `qwen2.5vl:3b` | ~3 GB | vision — "what do you see" |
| `qwen3:30b` | ~18 GB | the main brain |

Setup is **marker-gated and idempotent**: it writes `.medo-setup-done` only once
`qwen3:30b` has landed, so **every later launch skips setup and boots straight to
the app**, while an interrupted first run simply resumes on the next launch
(`ollama pull` continues partial downloads; winget/pull steps no-op for anything
already present).

Then say **"medo"** — or type into the HUD.

> **Don't want the 18 GB `qwen3:30b`?** Create the marker yourself to skip the
> heavy pull — `type nul > .medo-setup-done` (Windows) or
> `touch .medo-setup-done` (macOS/Linux) — then point `llm.default_model` in
> `config.yaml` at a smaller model (e.g. `llama3.2:3b`). Delete the marker to
> re-run the full setup.

### If winget / Homebrew is missing

Install the two apps by hand, then re-launch (it detects them and jumps to the
model pulls):

- **Ollama** — <https://ollama.com/download>
- **Obsidian** — <https://obsidian.md>

### Updating

```
git pull
```

Re-launch; the venvs and models are reused. If `requirements.txt` changed, delete
`.venv` (and `.venv-vision` if `requirements-vision.txt` changed) and the next
launch rebuilds them.

## Packaging into a one-click app

For non-technical users, an app layer (`app/` + `winsetup/`) wraps MEDO into a
one-click Windows installer: a tray **launcher** that supervises Ollama + the
engine + the vision sidecar (restart-with-backoff, clean teardown — no orphans),
**hardware detection** that auto-fits a model profile to the machine (a weak
laptop gets a small brain; a 24 GB GPU gets the 30B), a **first-run wizard**
(installs Ollama, downloads the right model with a progress bar, tests the mic),
and a friendly Settings screen — no terminal, no `config.yaml`. Build with
`python winsetup/build.py`; see [`docs/packaging.md`](docs/packaging.md) for the
dependency notes (the mediapipe/numpy split) and a manual install checklist. The
dev workflow (`run.bat` + `config.yaml`) is untouched.

---

## Configuration

Everything lives in **`config.yaml`** (env-overridable, prefix `MEDO_`, nested
with `__`, e.g. `MEDO_LLM__DEFAULT_MODEL=qwen3:14b`). Highlights:
`llm.default_model`, `languages.active` and `languages.detection`,
`audio.input_device` (mic priority list), `audio.barge_wake_frames`,
`vision.pointer.*`, `council.*` (roster, `max_members`, `second_opinion`,
`extra`), `self_dev.*`, `app_builder.*`, `filler`, `ui.overlay`, `mcp.servers`,
`remote.allowed_origins`, `memory.max_facts`, `weather.default_city`. Per-machine
secrets persist in the git-ignored `secrets.local.yaml` (written 0600).

## Ports (LAN only — never forward these)

| Port | What |
|------|------|
| 8730 | HUD (127.0.0.1) — serves the API token to loopback only |
| 8710 | Companion API — bearer-token auth for LAN clients, same-site Origin gate for browsers, per-caller rate limit, loopback exempt. **TLS available** (`remote.tls_enabled`) and worth turning on for any shared network |
| 8731 | Vision sidecar (127.0.0.1) — MJPEG `/video`, `/frame.jpg`, `POST /pointer`, cross-site refused |
| 11434 | Ollama |

## Tests

```
.venv\Scripts\python -m pytest
```

Heavy audio/vision model tests skip cleanly on machines without the deps.

## Docs

`docs/` is an **Obsidian vault** — architecture notes, runbook, bug log, roadmap,
decision records, and the security model. The original design handoff is
preserved in `docs/design/`. Filenames are kebab-case, so links work outside
Obsidian too.

| Where to look | For |
|---|---|
| [`docs/extending.md`](docs/extending.md) | Adding a skill, plugin, specialist, routine or MCP server — with working examples |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Something isn't working |
| [`docs/architecture.md`](docs/architecture.md) | The pipeline, the processes, the frozen contracts |
| [`docs/decisions.md`](docs/decisions.md) | Why it is the way it is — read before refactoring |
| [`docs/security.md`](docs/security.md) | Policy engine, trust boundary, threat model |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) · [`CHANGELOG.md`](CHANGELOG.md) · [`LICENSE`](LICENSE) | Contributing, what changed, MIT |

---

## Troubleshooting

Moved to **[`docs/troubleshooting.md`](docs/troubleshooting.md)** — the stuck
wake word, an unreachable brain, a black camera feed, a watch that won't pair,
and the rest, each with the fix.

## Known limitations

- **Macedonian is the one voice that needs the internet** (edge-tts, free) —
  every other language synthesises locally. Offline, or with
  `tts.allow_cloud: false`, a Macedonian reply is shown and not spoken.
- The wake word is a **custom "medo" model** trained locally on my own voice; a
  very different voice/mic may want a quick re-capture
  (`scripts/capture_wake_samples.py`) and retrain.
- Brightness control needs a laptop-class display.
- On macOS, pointer mode needs the **Accessibility** permission.
- Loading the vision model (qwen2.5-VL) temporarily evicts the 30B from VRAM on a
  12 GB GPU — the next chat pays a one-time reload.
