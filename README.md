# MEDO — Local AI Desktop Assistant

**MEDO is a bilingual (English/Macedonian) voice + gesture assistant that runs
entirely on my own hardware — wake word, Whisper ears, a swappable LLM brain,
Piper voice, MediaPipe hand-tracking, and a sci-fi HUD — in one Python codebase.**

> 🎥 [2-minute demo](docs/demo.md) — TODO: link

<!-- TODO(Matej): record the demo in this order — it shows breadth in 2 minutes:
     1. "hey jarvis" wake → "what time is it" (HUD CORE tab visible: FAST path,
        single-digit ms latency chip)
     2. a Macedonian question, answered in Macedonian
     3. "read my screen" (moondream vision)
     4. "pointer on" → drive the cursor by hand, pinch-click, fist to exit -->

| ![HUD — CORE tab](docs/img/hud-core.png) | ![Pointer mode](docs/img/pointer-mode.png) |
|---|---|
| *HUD CORE tab — drop screenshot at `docs/img/hud-core.png`* | *Pointer mode — drop screenshot at `docs/img/pointer-mode.png`* |

## Architecture at a glance

```
mic ──► openWakeWord ──► record until silence ──► faster-whisper (auto mk/en)
                                                        │
             HUD :8730 (SSE observer) ◄── EventBus ◄────┤
                                                        ▼
  watch app ──► companion API :8710 /ask ──────► Intent Router
  vision sidecar gestures ──► POST /ask ────────►   │
                                                    ├─ FAST: regex skill (<1 ms)
                                                    └─ LLM: Ollama tool loop
                                                        (strip_think, ≤4 rounds,
                                                         confirmation gate)
                                                        │
                                              Piper TTS ──► speakers
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

- **Voice**: openWakeWord → faster-whisper (auto-detects **Macedonian and
  English** per utterance) → Intent Router → Piper TTS. **Barge-in**: say the
  wake word (or tap the HUD mic) while MEDO talks to cut it off. The mic is a
  **priority list** (`audio.input_device`) — headset first, webcam fallback —
  hot-swapped within ~2 s and switchable live from the HUD.
- **Brain — five interchangeable providers**, switchable live from the HUD:
  **Ollama** (local, default `qwen3:30b` with 24 tools), **GPT / any
  OpenAI-compatible API**, the **Claude API**, and two local CLI agents —
  **Claude Code** and **Codex** (their own login, no key in MEDO).
  `</think>` reasoning is stripped before anything is spoken or remembered.
- **MCP — plug in any application**: declare servers in `config.yaml →
  mcp.servers` (stdio command or Streamable HTTP URL) and every tool they
  expose becomes a MEDO skill automatically — filesystem, Spotify, home
  automation, anything speaking the Model Context Protocol.
- **Fast path**: ~30 regex commands run deterministically in <1 ms — time,
  timers, notes, volume, media, apps, files, screenshots, windows, typing,
  clipboard, brightness, power.
- **Memory + RAG**: "remember that…" facts recalled **semantically** (local
  embeddings), and "what do my documents say about the lease?" searches
  whitelisted .txt/.md/.pdf by meaning, quoting the source file.
- **Pointer mode** (webcam): index finger drives the cursor · 🤏 pinch =
  drag/click · ✌ scroll · 🤘 zoom · three fingers = right-click · ✊ held ≈1 s
  = exit. Always boots OFF.
- **Sight**: "what do you see" / "read my screen" → local **moondream**.
- **HUD** (<http://localhost:8730>): live diagnostics, chat, config (provider
  + key, mic picker, themes, MCP status) over server-sent events, zero build
  step. The cosmic-web orb carries **~300 real folder dots** — wheel-zoom
  toward the cursor, grab-drag through the cluster, click opens the folder.
  The search box searches files and the web side by side.
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

Then say **"hey jarvis"** — or type into the HUD. (macOS/Linux: `run.command`.)

## Configuration

Everything lives in **`config.yaml`** (env-overridable, prefix `MEDO_`,
nested with `__`, e.g. `MEDO_LLM__DEFAULT_MODEL=qwen3:14b`). Highlights:
`llm.default_model`, `stt.language` (null = auto-detect),
`audio.input_device` (mic priority list — first available wins, hot-swapped),
`vision.pointer.*` (sensitivity, smoothing, scroll/zoom gains, fist exit hold),
`hud.max_dir_dots`, `memory.max_facts`, `weather.default_city`. Per-machine
secrets (online API key, picked mic, the companion-API token) persist in the
git-ignored `secrets.local.yaml`.

## Ports (LAN only — never forward these)

| Port | What |
|------|------|
| 8730 | HUD (127.0.0.1) |
| 8710 | Companion API — `/ask`, `/status`, `/sys`, `/models`, `/model`, `/provider`, `/dirs`, `/open`, `/wake`, `/interrupt`, `/audio/devices`, `/audio/input`, `/search/files`, `/search/web` — **bearer-token auth** for LAN clients (token minted into `secrets.local.yaml` on first serve; localhost exempt; no TLS — LAN only) |
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

## Known limitations

- Macedonian replies are spoken with a **neural mk-MK voice via edge-tts**
  (free, needs internet); offline they fall back to the English Piper voice.
- The wake word is openWakeWord's pretrained **"hey jarvis"** — a custom
  "hey MEDO" model is on the roadmap.
- Brightness control needs a laptop-class display (external monitors usually
  don't support WMI brightness).
- On macOS, pointer mode needs the **Accessibility** permission for whatever
  launches the sidecar (Terminal / `run.command`): System Settings → Privacy
  & Security → Accessibility. Until granted, the sidecar log says so and the
  cursor stays put.
- Loading moondream temporarily evicts the 30B from VRAM (12 GB GPU) — the
  next chat pays a one-time reload.
