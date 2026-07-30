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
                             edge-tts neural voice / Piper (offline) ──► speakers
```

Three processes, isolated on purpose: the main app (`.venv`), the MediaPipe
vision sidecar (`.venv-vision` — a quarantined `numpy<2` pin that talks only
HTTP), and Ollama. Every input — voice, watch, HUD text, hand gesture —
converges on the same Intent Router; every UI renders from the same event
stream. Full details in [`docs/Architecture.md`](docs/Architecture.md).

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
neural voice (edge-tts; the app trusts the OS certificate store via `truststore`,
so it works behind a TLS-intercepting proxy), with local **Piper** as the offline
fallback — and a non-English reply is never read aloud by the English voice.

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
(`software.enabled`); adding an app is a new connector file, not a core edit. See
[`docs/Software Connectors.md`](docs/Software%20Connectors.md).

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

MEDO is **local-first and LAN-only** — there is no TLS and these ports must never
be forwarded beyond your LAN. Within that boundary the trust model is:

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
- **Self-programming** runs the coding agent in an isolated worktree and its
  test/lint gate in a scrubbed, HOME/TEMP-isolated environment, behind a
  denylist and human apply-approval.

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

See [`docs/Security.md`](docs/Security.md) for the full threat model and honest
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
`python winsetup/build.py`; see [`docs/Packaging.md`](docs/Packaging.md) for the
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
| 8710 | Companion API — bearer-token auth for LAN clients, same-site Origin gate for browsers, loopback exempt, no TLS |
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
preserved in `docs/design/`.

---

## Troubleshooting — Q&A

Everything here is user-fixable — no code changes needed.

**Q: MEDO is stuck on STANDING BY / seems deaf.**
A: Run the live tester: `.venv\Scripts\python -m voice.wakeword`. It lists input
devices and prints mic level + wake score every frame. Level stuck near 0 →
wrong/muted microphone: fix `audio.input_device` or pick a mic in the HUD CONFIG
tab. Score peaking just under the threshold → lower `wakeword.threshold`. Or skip
the phrase entirely: the HUD **WAKE** button starts a turn without it.

**Q: Replies say "I can't reach my language model".**
A: The selected brain isn't reachable. Ollama: install/start it and pull a model
(`ollama pull llama3.2:3b`). GPT/Claude API: enter the key in the HUD CONFIG tab
and check the base URL. Claude Code / Codex: log in once on this machine. Fast-
path commands (time, volume, apps…) work regardless.

**Q: A council question / "second opinion" says the council can't meet.**
A: The council role-plays experts on the model that's loaded, so it needs a
reachable brain (or, for a CLI-agent brain, a local `tool_brain_model` on Ollama
so several specialists don't each spawn a slow process). Start Ollama or pick a
provider in the HUD.

**Q: "Type …" works in English but does nothing with Macedonian text.**
A: Cyrillic goes through a clipboard paste (Cmd+V on macOS). If it still fails,
the focused app may block programmatic paste; your clipboard is restored either
way.

**Q: The watch says "missing or invalid token".**
A: Re-pair from the watch (**Pair with MEDO** → type the 6-digit code MEDO shows
on the PC), or copy `remote.token` from the PC's `secrets.local.yaml`.

**Q: The camera feed is black / the sidecar exits immediately.**
A: On **macOS** this is almost always the **Camera permission** — grant it to the
app that launches the sidecar (Terminal / `run.command`) in System Settings →
Privacy & Security → **Camera**. Make sure no other app is holding the webcam.
The sidecar also needs its own venv (`run.command` builds `.venv-vision` on first
launch). Pointer mode additionally needs the **Accessibility** permission on
macOS.

**Q: A non-English reply isn't spoken (but shows in the HUD).**
A: The neural voices (edge-tts) need internet. Offline, English falls back to the
local Piper voice; a non-English reply is shown but not spoken rather than read
aloud with English phonemes. Text in the HUD stays correct either way.

**Q: A remote HUD (phone on the LAN) can't drive MEDO.**
A: By design — the API token is served to loopback only and cross-origin browser
requests are refused. Reach a remote HUD over an authenticated tunnel, or add its
origin to `remote.allowed_origins` on a trusted network.

## Known limitations

- Non-English replies are spoken with **neural voices via edge-tts** (free, needs
  internet); offline, English uses Piper and other languages are shown but not
  spoken.
- The wake word is a **custom "medo" model** trained locally on my own voice; a
  very different voice/mic may want a quick re-capture
  (`scripts/capture_wake_samples.py`) and retrain.
- Brightness control needs a laptop-class display.
- On macOS, pointer mode needs the **Accessibility** permission.
- Loading the vision model (qwen2.5-VL) temporarily evicts the 30B from VRAM on a
  12 GB GPU — the next chat pays a one-time reload.
