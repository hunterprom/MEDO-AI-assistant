# MEDO — Local AI Desktop Assistant

**A local-first Jarvis for Windows: wake word, bilingual voice (English +
Macedonian), hand-gesture mouse control, a sci-fi HUD, and a pluggable skill
system — powered by Ollama on your own GPU, with a one-click switch to any
OpenAI-compatible cloud model.**

> 🎬 **2-minute demo — TODO: link**

<!-- Record for the demo: wake ("hey jarvis") -> one fast-path command with the
     HUD latency chip visible -> one Macedonian question (answered in the
     mk-MK voice) -> "read my screen" -> pointer mode (pinch click, scroll,
     zoom, held-fist exit). Upload, then replace the TODO above with the link. -->

| HUD — CORE tab | Pointer mode |
|---|---|
| ![HUD CORE tab](docs/img/hud-core.png) | ![Pointer mode](docs/img/pointer-mode.png) |

<!-- Screenshots: drop two PNGs at exactly docs/img/hud-core.png (the CORE tab
     with the orb + diagnostics) and docs/img/pointer-mode.png (the optical
     feed with a hand landmark overlay). ~1600 px wide reads well on GitHub. -->

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

Two processes, two venvs: the main app (`.venv` — voice stack + aiohttp
servers) and the vision sidecar (`.venv-vision` — MediaPipe pins numpy<2);
they talk only over HTTP. Full details in `docs/Architecture.md` (the `docs/`
folder is an Obsidian vault).

## Performance

Measured on my machine (RTX with 12 GB VRAM, `qwen3:30b` local /
`llama-3.3-70b-versatile` via Groq). Every routed request appends a row to
the shared sqlite DB; regenerate this table any time with
`python -m core.metrics --report`.

<!-- Run `python -m core.metrics --report` after using MEDO for a while and
     paste its output over the placeholder table below. -->

| Path | Requests | Share | p50 | p95 |
|------|---------:|------:|----:|----:|
| FAST | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| LLM  | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

Reference points from live testing: fast-path skills answer in ~2–15 ms,
Groq replies land in ~0.5–0.7 s, the warm local 30B in ~7–9 s (first audio
starts after the first sentence thanks to streaming TTS, ~0.8 s).

## What it does

- **Voice**: openWakeWord → faster-whisper (auto-detects **Macedonian and
  English** per utterance) → Intent Router → Piper TTS; Macedonian replies
  are spoken by a neural mk-MK voice (edge-tts, online). **Barge-in**: say
  the wake word while MEDO is talking to cut it off; the HUD mic button
  interrupts too. The mic is a **priority list** (`audio.input_device` —
  headset first, webcam fallback), hot-swapped within ~2 s and selectable
  live from the HUD.
- **Brain**: `qwen3:30b` via Ollama tool-calling (24 tools); its `</think>`
  reasoning is stripped before anything is spoken, remembered, or shown.
  Model/provider (any OpenAI-compatible API) switchable at runtime from the
  HUD — the key lives only in git-ignored `secrets.local.yaml`.
- **Fast path**: ~30 regex-matched commands in <1 ms — time, timers, notes,
  volume (real Core Audio), media keys, apps, files, screenshots, window
  management, typing, clipboard, brightness, power.
- **Memory + RAG**: "remember that …" facts are recalled **semantically**
  (local embeddings — "tooth appointment" finds the dentist fact), and
  "what do my documents say about X" searches whitelisted .txt/.md/.pdf by
  meaning, quoting the source file.
- **Pointer mode** (webcam sidecar; always boots OFF): index finger = cursor
  · 🤏 pinch = drag / quick-tap click · ✌ two fingers = scroll · 🤘 = zoom ·
  three fingers = right-click · 👍 / pinky = volume up / down · ✊ held ~1 s
  = exit; unrecognized poses keep moving the cursor.
- **Sight**: "what do you see" (camera) and "read my screen" (display) via a
  local **moondream** vision model.
- **HUD** (<http://localhost:8730>): CORE (diagnostics, activity log, a
  cosmic-web orb carrying **~300 real folder dots** — hover shows the path,
  click opens Explorer), COMMS (chat), CONFIG (model/provider + API key, mic
  picker, color themes, memory manager) and a **search box** covering your
  files and the web. Server-sent events, zero build step.
- **Routines**: proactive scheduled briefings — by default a 08:00 morning
  briefing speaks the weather and headlines unprompted.
- **Plugins**: drop a `.py` into `plugins/` and restart — the skill works by
  voice AND as an LLM tool; a broken plugin is skipped, never fatal
  (`plugins/README.md`).
- **Watch app** (`watch/`): Wear OS companion on the same token-authed API.

## How I built this

- **v1 — `jarvis-web`** (React + Express/Node + nut-js): proved the ideas —
  gesture mouse, wake word, desktop control — but split the logic across two
  runtimes and a build step.
- **v2** — a Python rearchitecture: event bus, one-implementation-two-paths
  skill contract, safety confirmation gate, real tests.
- **This repo is the merge**: v2's architecture as the base, v1's best
  features ported in, the entire Node stack deleted — the HUD is one static
  HTML file served by aiohttp.
- The three hardest problems (full stories in `docs/Bug Log.md`):
  **think-tag leakage** (qwen3's `<think>` reasoning reached TTS, memory, and
  the yes/no safety gate — fixed once at the `chat()` choke point), **sidecar
  isolation** (MediaPipe pins numpy<2, so vision runs in its own venv and
  talks to the app only over HTTP), and **the false "100% local" claim**
  (v1's browser Web-Speech STT quietly sent audio to Google — replaced with
  genuinely local Whisper).
- **AI assistance disclosure** — parts of this codebase were built
  pair-programming with Claude (Anthropic).
  <!-- TODO (Matej): fill in the specifics honestly. Suggested shape:
       "Claude assisted with: <subsystems / migrations / test suites>.
        I independently designed/decided/debugged: <parts>.
        Every change was reviewed by me; the reasoning behind each decision
        is recorded in docs/Decisions.md." -->

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
secrets (online API key, picked mic, the API token) persist in the
git-ignored `secrets.local.yaml`.

## Ports (LAN only — never forward these)

| Port | What |
|------|------|
| 8730 | HUD (127.0.0.1) |
| 8710 | Companion API — `/ask`, `/status`, `/sys`, `/models`, `/model`, `/provider`, `/dirs`, `/open`, `/wake`, `/interrupt`, `/audio/devices`, `/audio/input`, `/search/files`, `/search/web`. **Bearer-token auth** for LAN clients (`Authorization: Bearer <token>`, or `?token=` where headers are impossible); the token is generated on the first `--serve` run into `secrets.local.yaml` → `remote.token`. Requests from 127.0.0.1 (HUD, sidecar) are exempt. `remote.auth_enabled: false` restores the old open API — unsafe. |
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
- Loading moondream temporarily evicts the 30B from VRAM (12 GB GPU) — the
  next chat pays a one-time reload.
