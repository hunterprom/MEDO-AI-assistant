# MEDO — Local AI Desktop Assistant

The super project: **MEDO v1** (jarvis-web — React/Express/nut-js gesture & voice
assistant) and **MEDO v2** (Python rearchitecture) merged into one codebase.
Fully local: Ollama for the brain, Whisper for ears, Piper for the voice,
MediaPipe for the eyes. No cloud APIs, no keys, nothing leaves the machine.

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
- **Brain**: `qwen3:30b` by default via Ollama tool-calling with 24 tools.
  It's a thinking model — its `</think>` reasoning is stripped before anything
  is spoken, remembered, or shown. `keep_alive: 30m` prevents reload stalls.
  Runtime-switchable model/provider (any OpenAI-compatible API) from the HUD.
- **Fast path**: ~30 regex-matched commands run deterministically in <1 ms —
  time, timers, notes, volume (real Core Audio on Windows), media keys, apps,
  files, screenshots, window management, typing, clipboard, brightness, power.
- **Long-term memory**: "remember that …", "what do you remember about me",
  "forget …" — stored in sqlite, injected into the LLM's system prompt.
- **Gestures** (webcam sidecar): 👍 yes · ✋ no · ✌ screenshot · ☝ volume up ·
  three volume down · ✊ mute · 🤏 lock · 🤘 play/pause.
- **Pointer mode** — say "pointer on" (or toggle in the HUD): your index
  finger drives the mouse cursor, **pinch = left click**, **three fingers =
  right click**, **fist = exit**. Discrete gestures pause while it's on.
- **Sight**: "what do you see" (camera) and "read my screen" (display) go to a
  local **moondream** vision model.
- **HUD** (<http://localhost:8730>): the *Medo AI Assistant Interface* design
  (see `docs/design/`) — CORE (live diagnostics, environment, activity log,
  cosmic-web orb, routing, subsystems, optical feed), COMMS (secure-channel
  chat), CONFIG (toggles + model picker). Server-sent events, zero build step.
- **Watch app** (`watch/`): Wear OS companion that talks to the same API.

## Ports (LAN only — never forward these)

| Port | What |
|------|------|
| 8730 | HUD (127.0.0.1) |
| 8710 | Companion API — `/ask`, `/status`, `/sys`, `/models`, `/model`, `/provider` (no auth **by design**; LAN only) |
| 8731 | Vision sidecar — MJPEG `/video`, `/frame.jpg`, `POST /pointer` |
| 11434 | Ollama |

## Configuration

Everything lives in **`config.yaml`** (env-overridable, prefix `MEDO_`,
nested with `__`, e.g. `MEDO_LLM__DEFAULT_MODEL=qwen3:14b`). Highlights:
`llm.default_model`, `stt.language` (null = auto-detect), `vision.pointer.*`
(sensitivity, smoothing, debounce), `vision.gestures` (gesture → utterance),
`memory.max_facts`, `weather.default_city`.

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

- Piper has **no Macedonian voice**: Macedonian replies are correct as text in
  the HUD but are spoken with the English voice. Roadmap: optional edge-tts.
- The wake word is openWakeWord's pretrained **"hey jarvis"** — a custom
  "hey MEDO" model is on the roadmap.
- Brightness control needs a laptop-class display (external monitors usually
  don't support WMI brightness).
- Loading moondream temporarily evicts the 30B from VRAM (12 GB GPU) — the
  next chat pays a one-time reload.
