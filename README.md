# MEDO — Local AI Desktop Assistant

**MEDO is a two-language (default English/Macedonian, configurable pair) voice +
gesture assistant that runs entirely on my own hardware — a custom "medo" wake
word trained on my own voice, Whisper ears, a swappable LLM brain, neural voices
in either language, MediaPipe hand-tracking, a sci-fi HUD, and a desktop presence
sphere — in one Python codebase.**

> 🎥 [2-minute demo](docs/demo.md) — TODO: link

<!-- TODO(Matej): record the demo in this order — it shows breadth in 2 minutes:
     1. "medo" wake → "what time is it" (HUD CORE tab visible: FAST path,
        single-digit ms latency chip)
     2. a Macedonian question, answered aloud in a Macedonian neural voice
     3. "read my screen" (local qwen2.5-VL vision)
     4. "pointer on" → drive the cursor by hand, pinch-click, fist to exit -->

| ![HUD — CORE tab](docs/img/hud-core.png) | ![Pointer mode](docs/img/pointer-mode.png) |
|---|---|
| *HUD CORE tab — drop screenshot at `docs/img/hud-core.png`* | *Pointer mode — drop screenshot at `docs/img/pointer-mode.png`* |

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
                                                    └─ LLM: tool loop (strip_think,
                                                        ≤4 rounds, confirmation
                                                        gate, "let me think" filler)
                                                        │
                             edge-tts neural voice / Piper (offline) ──► speakers
```

Three processes, isolated on purpose: the main app (`.venv`), the MediaPipe
vision sidecar (`.venv-vision` — numpy<2 pin stays quarantined), and Ollama.
Every input — voice, watch, HUD text, hand gesture — converges on the same
Intent Router; every UI renders from the same event stream. Full details in
[`docs/Architecture.md`](docs/Architecture.md).

## Performance

Every routed request is measured on the machine itself (`metrics` table in the
assistant's sqlite DB, `logging.routing_stats: true`) and this table is
generated from it — not estimated:

<!-- run: python -m core.metrics --report  and paste its output below -->

| Path | Requests | Share | p50 | p95 |
|------|---------:|------:|--------:|--------:|
| FAST | *(run the report)* | | | |
| LLM  | *(run the report)* | | | |

Method: measured end-to-end inside the Intent Router (utterance in → reply
ready), on my machine — RTX 3060 12 GB, `qwen3:30b` via Ollama for the LLM
path. Fast-path commands never touch the model, which is why they sit three
orders of magnitude below it.

## What it does

- **Voice**: a **custom "medo" wake word** (openWakeWord, retrained on my own
  captured voice — fires on "medo" and "hey medo") → faster-whisper → Intent
  Router → **neural TTS**. Replies are spoken in the language they were asked
  in, by that language's neural voice (edge-tts; the app trusts the OS
  certificate store via `truststore`, so it works behind a TLS-intercepting
  proxy), with local **Piper** as the offline fallback — and a non-English
  reply is never read aloud by the English voice (that gibberish is caught).
  **Two-language mode**: MEDO runs with **at most two active languages** at a
  time (default **English + Macedonian**) and constrains per-utterance detection
  to that pair — two known candidates are far more reliable than open-ended
  language ID. The pair is configurable (`languages.active`, any two of the
  supported set) with a picker on the HUD CONFIG screen; `detection: fixed`
  forces one language when you only speak one. **Barge-in**: say the wake word
  (or tap the HUD mic) while MEDO talks to cut it off. The mic is a **priority
  list** (`audio.input_device`) — headset first, webcam fallback — hot-swapped
  within ~2 s and switchable live from the HUD.
- **Brain — five interchangeable providers**, switchable live from the HUD:
  **Ollama** (local, default `qwen3:30b` with 24 tools), **GPT / any
  OpenAI-compatible API**, the **Claude API**, and two local CLI agents —
  **Claude Code** and **Codex** (their own login, no key in MEDO).
  `</think>` reasoning is stripped before anything is spoken or remembered.
  When a CLI agent is slow (they can take 20-30 s), MEDO speaks a **"let me
  think" filler** so the gap doesn't read as a freeze — LLM path only, in the
  turn's language, configurable under `filler`.
- **MCP — plug in any application**: declare servers in `config.yaml →
  mcp.servers` (stdio command or Streamable HTTP URL) and every tool they
  expose becomes a MEDO skill automatically. A curated set ships **disabled by
  default** — GitHub, Google Maps, Google Drive, sequential-thinking,
  filesystem, memory, Brave Search (plus a Fusion 360 placeholder). Flip one on
  and drop its secret into the git-ignored `secrets.local.yaml` under
  `mcp_env:` — it's injected into the environment, never written to `config.yaml`.
- **Fast path**: ~30 regex commands run deterministically in <1 ms — time,
  timers, notes, volume, media, apps, files, screenshots, windows, typing,
  clipboard, brightness, power.
- **Search where you actually search**: "search drone motors on YouTube",
  "check my email for the invoice", "find Half-Life on Steam" open that site's
  own results — ~35 sites built in (YouTube, Gmail, Reddit, Steam, GitHub,
  Amazon, Wikipedia, Maps, Thingiverse…), extend the list in `config.yaml →
  skills.sites` without touching code.
- **Drives the browser, not just opens it**: "click sign in", "type quadruped
  robot into the search box", "read the page", "what can I click" — MEDO runs
  real Chrome through Playwright and works on the **DOM**, matching the button
  you named instead of guessing pixels, so window size, zoom and theme don't
  matter. A persistent profile means you log into a site once. "On the site,
  …" hands a bounded multi-step task to the brain (spoken yes first, capped at
  `browser.max_steps`, `blocked_domains` re-checked after every redirect).
- **Bilingual commands, not just bilingual chat**: the fast path answers
  Macedonian too — "отвори хром", "затвори спотифај", "отвори јутјуб",
  "барај мачки на јутјуб", "најди ја датотеката извештај" — and replies in the
  language it was asked in. Cyrillic site names ("јутјуб", "стим", "пошта")
  resolve like any other alias.
- **Memory + RAG**: "remember that…" facts recalled **semantically** (local
  embeddings), and "what do my documents say about the lease?" searches
  whitelisted .txt/.md/.pdf by meaning, quoting the source file.
- **Pointer mode** (webcam): index finger drives the cursor · 🤏 pinch =
  drag/click · ✌ scroll · 🤘 zoom · three fingers = right-click · ✊ held ≈1 s
  = exit. Always boots OFF.
- **Sight**: "what do you see" / "read my screen" / "can you see my screen" →
  local **qwen2.5-VL** (Ollama). First look after an idle spell retries once
  while the model warm-loads, so a cold start doesn't read as "no camera".
- **HUD** (<http://localhost:8730>): three tabs — **CORE** (live diagnostics,
  the orb, and the **chat as a side panel** under the activity log), **CLUSTER**
  (a cosmic-web sphere per capability domain, built from the live skill
  registry), and **CONFIG** (provider + key, mic picker, the **spoken-language
  picker**, themes, MCP status) — all over server-sent events, zero build step.
  The CORE orb carries **~300 real folder dots** — wheel-zoom toward the cursor,
  grab-drag through the cluster, click opens the folder. The search box searches
  files and the web side by side.
- **Desktop presence sphere** (`ui/overlay.py`): a small always-on-top CORE
  sphere pinned to a screen corner, so MEDO stays visible while you work in
  other apps. It shows state (idle / listening / thinking / speaking) and
  bubbles up what you said and what it answered; **click** to talk without the
  wake word, **double-click** to open the HUD. Its own process — a UI crash
  can't take the assistant down. Toggle under `ui.overlay`.
- **Extensible**: drop a `.py` in `plugins/` → it works by voice AND as an LLM
  tool; a Wear OS **watch app** (`watch/`) talks to the same API — one tap on
  **"Pair with MEDO"** finds the PC by UDP broadcast and swaps a 6-digit
  on-screen code for the auth token, so nothing is typed but the code.

## How I built this

<!-- TODO(Matej): fill the specifics marked TODO, keep it honest -->

- **v1** was Java/Node (jarvis-web): React HUD, Express, nut-js gestures. It
  proved the ideas but the stack fought me — two runtimes, a build step, and
  an unauditable "100% local" claim (browser Web-Speech STT actually sent
  audio to Google — bug #14).
- **v2** was a ground-up Python rearchitecture: event bus, one-implementation
  skill contract (regex pattern + LLM tool schema from the same class), safety
  confirmation gate, real tests.
- The **merge** ported v1's best features (pointer mode, think-stripping,
  facts, screen vision) onto v2's architecture — and deleted the Node stack.
- Hardest three problems (full war stories in [`docs/Bug Log.md`](docs/Bug%20Log.md)):
  qwen3's `<think>` reasoning **leaking into TTS and the yes/no safety gate**
  (fixed once at the `chat()` choke point); **MediaPipe's numpy<2 pin**
  colliding with the voice stack (solved with a quarantined sidecar venv that
  talks only HTTP); and **auditing the "local" claim** until it was true.
- **AI-assistance disclosure**: <!-- TODO(Matej): fill in specifics — e.g.
  "Pair-programmed with Claude (architecture reviews, test scaffolding, the
  HUD's canvas math); independently designed/decided X, Y, Z; every line
  reviewed by me and defended in docs/Decisions.md." Don't overclaim either
  direction. -->

## One-click run (Windows)

Double-click **`run.bat`**. First launch creates two virtualenvs, installs
dependencies, downloads the wake-word + Piper voice models, starts Ollama with
the right environment (`OLLAMA_MODELS=D:\OllamaModels`, `GGML_CUDA_NO_PINNED=1`),
launches the vision sidecar, and opens the HUD at <http://localhost:8730>.

Then say **"medo"** — or type into the HUD. (macOS/Linux: `run.command`.)

## Configuration

Everything lives in **`config.yaml`** (env-overridable, prefix `MEDO_`,
nested with `__`, e.g. `MEDO_LLM__DEFAULT_MODEL=qwen3:14b`). Highlights:
`llm.default_model`, `languages.active` (the ≤2 languages MEDO hears/speaks)
and `languages.detection` (`auto_pair` | `fixed`), `audio.input_device` (mic
priority list — first available wins, hot-swapped), `vision.pointer.*`
(sensitivity, smoothing, scroll/zoom gains, fist exit hold), `filler` (the
slow-answer fillers), `ui.overlay` (the corner sphere), `mcp.servers`,
`hud.max_dir_dots`, `memory.max_facts`, `weather.default_city`. Per-machine
secrets (online API key, picked mic, the companion-API token, and MCP tokens
under `mcp_env:`) persist in the git-ignored `secrets.local.yaml`.

## Ports (LAN only — never forward these)

| Port | What |
|------|------|
| 8730 | HUD (127.0.0.1) |
| 8710 | Companion API — `/ask`, `/status`, `/sys`, `/models`, `/model`, `/provider`, `/dirs`, `/open`, `/wake`, `/interrupt`, `/audio/devices`, `/audio/input`, `/control/languages`, `/search/files`, `/search/web` — **bearer-token auth** for LAN clients (token minted into `secrets.local.yaml` on first serve; localhost exempt; no TLS — LAN only) |
| 8731 | Vision sidecar — MJPEG `/video`, `/frame.jpg`, `POST /pointer` |
| 11434 | Ollama |

## Tests

```
.venv\Scripts\python -m pytest
```

Heavy audio/vision model tests skip cleanly on machines without the deps.

## Docs

`docs/` is an **Obsidian vault** — open it in Obsidian for the architecture
notes, runbook, bug log, roadmap, and decision records. The original design
handoff is preserved in `docs/design/`.

## Troubleshooting — Q&A

Everything here is user-fixable — no code changes needed.

**Q: Pointer mode is on but the cursor doesn't move (macOS).**
A: macOS blocks synthetic input until you grant the **Accessibility**
permission to whatever launches the sidecar (Terminal / `run.command`):
System Settings → Privacy & Security → **Accessibility** → enable it, then
restart the sidecar. The sidecar log says exactly this when it's the cause.
First runs also need the **Camera** permission (macOS prompts).

**Q: Pointer mode says "unavailable on this system" (Linux).**
A: The Linux backend needs pyautogui in the sidecar venv:
`.venv-vision/bin/pip install pyautogui` (X11; on Wayland enable XWayland).
Windows and macOS need nothing — their backends are built in.

**Q: MEDO is stuck on STANDING BY / seems deaf.**
A: Run the live tester: `.venv\Scripts\python -m voice.wakeword`. It lists
input devices and prints mic level + wake score every frame. Level stuck
near 0 → wrong/muted microphone: fix `audio.input_device` (it's a priority
list) or pick a mic in the HUD CONFIG tab. Score peaking just under the
threshold → lower `wakeword.threshold` a notch. Or skip the phrase entirely:
the HUD **WAKE** button starts a turn without it.

**Q: "Another MEDO is already running (port 8710 is busy)".**
A: An older MEDO window is still open — close it (or just re-run `run.bat`,
which replaces it). Otherwise something else grabbed the port: change
`remote.port` in config.yaml.

**Q: The watch says "Can't reach Jarvis".**
A: Watch and PC must be on the **same Wi-Fi/LAN**, and MEDO must be running
with the companion API (`--serve`, or `vision.enabled`/`remote.enabled`).
Then check the firewall allows inbound TCP 8710. Easiest re-setup: watch
settings → **Pair with MEDO** (fills address + token by itself).

**Q: The watch says "missing or invalid token".**
A: Auth is on (good). Re-pair from the watch (**Pair with MEDO** → type the
6-digit code MEDO shows on the PC), or copy `remote.token` from the PC's
`secrets.local.yaml` into the watch's token field.

**Q: Replies say "I can't reach my language model".**
A: Whichever brain is selected isn't reachable. Ollama: install/start it and
pull a model (`ollama pull llama3.2:3b`). GPT/Claude API: enter the key in
the HUD CONFIG tab and check the base URL. Claude Code / Codex: log in once
on this machine (`claude login` / `codex login`). Fast-path commands (time,
volume, apps…) work regardless.

**Q: The first question after picking the big model hangs ~30 s.**
A: That's the one-time cold load of the 30B into VRAM; MEDO pre-warms on
switch, and `keep_alive: 30m` keeps it loaded between turns. Repeated
reloads usually mean something else is evicting VRAM (see the vision-model
note under Limitations).

**Q: A non-English reply isn't spoken (but shows in the HUD).**
A: The neural voices (edge-tts) need internet. Online, every active language is
spoken in its own voice — behind a TLS-intercepting proxy that means the OS
trust store is used (`truststore`, injected at startup). Offline, English falls
back to the local Piper voice; a non-English reply is **shown but not spoken**
rather than read aloud with English phonemes (which was gibberish). Text in the
HUD stays correct either way.

**Q: `pip install` fails with SSL certificate errors.**
A: Corporate/filtered networks intercept TLS. Use
`pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org …`
— the app itself already trusts the OS certificate store (truststore).

**Q: The HUD is open but everything says API OFFLINE.**
A: The HUD (port 8730) is served but the companion API (8710) isn't —
start MEDO with `--serve` (or set `remote.enabled: true`), and hard-refresh
the page after a restart.

**Q: The camera feed is black / the sidecar exits immediately / "could not
open camera".**
A: On **macOS** this is almost always the **Camera permission** — and macOS
usually does *not* pop a prompt for OpenCV, it just denies silently. Grant it
to the app that launches the sidecar (Terminal / `run.command`) in System
Settings → Privacy & Security → **Camera**. If it isn't in the list, run
`tccutil reset Camera` in that terminal and relaunch so macOS re-asks. Also
make sure no other app is holding the webcam, and try `vision.camera_index: 1`
if you have more than one. (The sidecar also needs its own venv — a fresh
clone has no `.venv-vision` yet; `run.command` builds it on first launch.)

**Q: "Type …" works in English but does nothing with Macedonian text.**
A: Fixed — Cyrillic goes through a clipboard paste, which now uses Cmd+V on
macOS (was Ctrl+V, a no-op there). If it still fails, the focused app may
block programmatic paste; your clipboard is restored either way.

## Known limitations

- Non-English replies are spoken with **neural voices via edge-tts** (free,
  needs internet — uses the OS trust store to work behind a TLS proxy); offline,
  English uses the local Piper voice and other languages are shown but not
  spoken.
- The wake word is a **custom "medo" model** trained locally on my own voice
  (openWakeWord). It fires on "medo" and "hey medo"; it was tuned on a small
  set of my own recordings, so a very different voice/mic may want a quick
  re-capture (`scripts/capture_wake_samples.py`) and retrain.
- Brightness control needs a laptop-class display (external monitors usually
  don't support WMI brightness).
- On macOS, pointer mode needs the **Accessibility** permission for whatever
  launches the sidecar (Terminal / `run.command`): System Settings → Privacy
  & Security → Accessibility. Until granted, the sidecar log says so and the
  cursor stays put.
- Loading the vision model (qwen2.5-VL) temporarily evicts the 30B from VRAM
  (12 GB GPU) — the next chat pays a one-time reload.
