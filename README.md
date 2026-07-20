# MEDO — Local AI Desktop Assistant

The super project: **MEDO v1** (jarvis-web — React/Express/nut-js gesture & voice
assistant) and **MEDO v2** (Python rearchitecture) merged into one codebase.
Local-first: Ollama for the brain, Whisper for ears, Piper for the voice,
MediaPipe for the eyes. Optionally switch the brain to any OpenAI-compatible
cloud API from the HUD — the key is stored in a git-ignored local file and
never committed or echoed back.

## One-click run (Windows)

Double-click **`run.bat`**. First launch creates two virtualenvs, installs
dependencies, downloads the wake-word + Piper voice models, starts Ollama with
the right environment (`OLLAMA_MODELS=D:\OllamaModels`, `GGML_CUDA_NO_PINNED=1`),
launches the vision sidecar, and opens the HUD at <http://localhost:8730>.

Then say **"hey jarvis"** — or type into the HUD. (macOS/Linux: `run.command`.)

## What it does

- **Voice**: openWakeWord → faster-whisper (auto-detects **Macedonian and
  English** per utterance) → Intent Router → Piper TTS. Replies match your
  language (spoken audio uses the English voice — see Limitations).
  **Barge-in**: say the wake word while MEDO is talking to cut it off and be
  heard immediately; the HUD mic button interrupts too. The microphone is a
  **priority list** (`audio.input_device`) — e.g. Bluetooth headset first,
  webcam fallback — hot-swapped within ~2 s of a device (dis)connecting, and
  also selectable live from the HUD CONFIG tab.
- **Brain**: `qwen3:30b` by default via Ollama tool-calling with 24 tools.
  It's a thinking model — its `</think>` reasoning is stripped before anything
  is spoken, remembered, or shown. `keep_alive: 30m` prevents reload stalls.
  Runtime-switchable model/provider (any OpenAI-compatible API) from the HUD.
- **Fast path**: ~30 regex-matched commands run deterministically in <1 ms —
  time, timers, notes, volume (real Core Audio on Windows), media keys, apps,
  files, screenshots, window management, typing, clipboard, brightness, power.
- **Long-term memory**: "remember that …", "what do you remember about me",
  "forget …" — stored in sqlite and recalled **semantically** (local
  embeddings: "when is my tooth appointment" finds the dentist fact).
- **Ask your documents (RAG)**: "what do my documents say about the lease?" —
  .txt/.md/.pdf files in the whitelisted folders are chunked, embedded
  locally, and searched by meaning; answers quote the source file.
- **Pointer mode** (webcam sidecar; pointer-only build — say "pointer on" or
  toggle in the HUD): your index finger drives the mouse cursor ·
  **🤏 pinch = drag / quick-tap click** · **✌ two fingers = scroll** ·
  **🤘 index+pinky = zoom (Ctrl+wheel)** · **three fingers = right click** ·
  **👍 = volume up** · **pinky = volume down** · **✊ held ~1 s = exit**
  (brief fist misreads while pointing don't kick you out; unrecognized poses
  keep moving the cursor). Always boots OFF.
- **Sight**: "what do you see" (camera) and "read my screen" (display) go to a
  local **moondream** vision model.
- **HUD** (<http://localhost:8730>): the *Medo AI Assistant Interface* design
  (see `docs/design/`) — CORE (live diagnostics, environment, activity log,
  cosmic-web orb, routing, subsystems, optical feed), COMMS (secure-channel
  chat), CONFIG (toggles, model/provider + API key, microphone picker, accent
  color themes). Server-sent events, zero build step. The orb carries **~300
  real folder dots** — hover shows the path, click opens it in Explorer — and
  the top **search box** searches your files and the web side by side
  (results open in Explorer / the browser, or hand the query to MEDO).
- **Plugins**: drop a `.py` file in `plugins/` and restart — your skill works
  by voice AND as an LLM tool, no core changes (see `plugins/README.md`; a
  broken plugin is skipped, never fatal).
- **Watch app** (`watch/`): Wear OS companion that talks to the same API.

## Ports (LAN only — never forward these)

| Port | What |
|------|------|
| 8730 | HUD (127.0.0.1) |
| 8710 | Companion API — `/ask`, `/status`, `/sys`, `/models`, `/model`, `/provider`, `/dirs`, `/open`, `/wake`, `/interrupt`, `/audio/devices`, `/audio/input`, `/search/files`, `/search/web`. **Bearer-token auth** for LAN clients (`Authorization: Bearer <token>`, or `?token=` where headers are impossible); the token is generated on the first `--serve` run into `secrets.local.yaml` → `remote.token`. Requests from 127.0.0.1 (HUD, sidecar) are exempt. `remote.auth_enabled: false` restores the old open API — unsafe. |
| 8731 | Vision sidecar — MJPEG `/video`, `/frame.jpg`, `POST /pointer` |
| 11434 | Ollama |

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

## Configuration

Everything lives in **`config.yaml`** (env-overridable, prefix `MEDO_`,
nested with `__`, e.g. `MEDO_LLM__DEFAULT_MODEL=qwen3:14b`). Highlights:
`llm.default_model`, `stt.language` (null = auto-detect),
`audio.input_device` (mic priority list — first available wins, hot-swapped),
`vision.pointer.*` (sensitivity, smoothing, scroll/zoom gains, fist exit hold),
`hud.max_dir_dots`, `memory.max_facts`, `weather.default_city`. Per-machine
secrets (online API key, picked mic) persist in the git-ignored
`secrets.local.yaml`.

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
