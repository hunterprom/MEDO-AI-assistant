# MEDO v2 — Local Voice Assistant

A fully offline, voice-controlled AI assistant. 100% local (Ollama for the LLM),
no cloud APIs, no keys. Ground-up Python rearchitecture of MEDO v1 (Java).

> **Status: M0–M7 complete.** 142-test pytest suite. One-click launchers
> (`run.command` / `run.bat`) bring up Ollama, the gesture sidecar, and the app
> with voice + HUD + API.

## New in M7

- **Any AI model, local or cloud.** `llm.provider: ollama | openai` — drop an API
  key into the HUD's settings drawer (or `config.yaml` / `MEDO_LLM__API_KEY`) and
  any OpenAI-compatible endpoint becomes MEDO's brain, tool calling included. The
  key is write-only: never logged, never echoed, never rendered.
- **Live model & provider switching** — from the HUD settings drawer or the API
  (`GET /status`, `GET /models`, `POST /model`, `POST /provider`).
- **HUD themes** — 5 presets (Arc Cyan, Mark III Red, Emerald, Royal Purple,
  Amber) + a custom color picker; recolors panels *and* the reactor; persisted.
- **HUD command box** — type a command in the browser, MEDO routes and replies.
- **8 gestures, full PC control** — added **three** (index+middle+ring) → volume
  down, **pinch** → lock the screen, **rock** → play/pause music (new `media`
  skill: Music/Spotify on macOS, media keys elsewhere). The camera overlay now
  shows what vision sees (gesture, last fired, utterance).
- **Watch gesture activation** — double wrist-flick (accelerometer) starts
  listening hands-free on the Galaxy Watch8, with a sensitivity slider; true
  finger-pinch isn't exposed to Wear OS apps (OS limitation, documented in
  `watch/README.md`).
- **Sturdier transcription** — Whisper hallucination filtering (`no_speech_prob`/
  `avg_logprob` gates + junk-transcript blacklist), RMS-target loudness
  normalization, adaptive ambient-noise floor, optional hotwords biasing.

## One-click run

- **macOS/Linux:** double-click `run.command` (or `./run.command`)
- **Windows:** double-click `run.bat`

First launch creates both virtualenvs, installs everything, downloads the wake-word
+ voice models, starts Ollama, launches the gesture sidecar, and opens the HUD at
`http://localhost:8730`. Then say the wake phrase **"Hey Jarvis"** — the assistant
is MEDO, but the wake word is a pretrained openWakeWord model, so the trigger phrase
stays "Hey Jarvis" until a custom "Hey MEDO" model is trained (see Limitations).

## Architecture

```
 Wake Word ──▶ STT ──▶ Intent Router ──┬─▶ FAST PATH  (rule-based skills, no LLM)
 (openWake-   (faster-                 └─▶ LLM  PATH  (Ollama + tool calls)
   Word)       whisper)                         │
                                                ▼
                                          TTS (Piper)

 States:  IDLE ─▶ LISTENING ─▶ THINKING ─▶ SPEAKING ─▶ IDLE
```

**The Intent Router is the heart of the project.** Deterministic commands
("what time is it") match regex patterns on the **fast path** and return in
sub-millisecond time with no LLM. Everything else goes to the **LLM path**
(Ollama). Each request logs which path handled it.

**One skill, two routes.** Every capability is written once as a `Skill`
(`skills/base.py`) that declares both regex `patterns` (fast path) and a JSON
`tool_schema` (LLM function calling, M3). The `SkillRegistry` is the single
source of truth for both.

## Layout

```
config.yaml          all settings (swap models/voices/whitelist without code edits)
main.py              entry point: REPL, voice loop, state machine
core/    config.py events.py router.py safety.py memory.py metrics.py platform.py
llm/     client.py tools.py prompts.py       # Ollama client + function-calling glue
skills/  base.py datetime apps system files timers notes  weather news websearch
voice/   audio.py wakeword.py stt.py tts.py  # openWakeWord · faster-whisper · Piper
ui/      console.py                          # rich state/routing/latency rendering
remote/  server.py                           # LAN companion API (watch app)
tests/   57 tests: router, safety, skills, web, memory, metrics
```

## Requirements

- **Python 3.12** (the voice stack depends on `onnxruntime`, which has no 3.14
  wheels yet — 3.12 has full wheel coverage for the whole stack).
- [Ollama](https://ollama.com) running locally with at least one model pulled.
  On CPU-only machines use a small model: `ollama pull llama3.2:3b`. Optional for
  the fast path; only the LLM path needs it.

## Setup

```bash
cd jarvis-v2
python3.12 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# one-time model downloads for voice mode:
python -c "import openwakeword.utils as u; u.download_models()"   # wake word
# Piper voice (default config expects en_US-lessac-medium):
mkdir -p models/piper && cd models/piper
base=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium
curl -LO $base/en_US-lessac-medium.onnx && curl -LO $base/en_US-lessac-medium.onnx.json
cd ../..

python main.py            # text REPL
python main.py --voice    # hands-free (grant mic permission when prompted)
```

## Demo script (the exact commands to run live)

Text mode (`python main.py`) — each renders a panel with the routing path + latency:

```
1.  what time is it                    → [FAST]  deterministic, ~0 ms
2.  set a timer for 10 seconds         → [FAST]  fires and speaks when done
3.  take a note pick up the drone      → [FAST]  saved to SQLite
4.  read my notes                      → [FAST]  reads it back
5.  what's my battery and cpu          → [FAST]  psutil, read-only
6.  shut down the pc                   → [FAST]  "Are you sure?"  → answer: no
7.  what's the weather                 → [FAST]  Open-Meteo (offline-safe)
8.  and tomorrow?                      → [LLM]   resolves from context → tomorrow's forecast
9.  search for the latest NVIDIA Isaac Sim release and summarize it
                                       → [FAST]  DuckDuckGo + LLM summary
10. /latency                           → averaged per-stage latencies vs budgets
```

Voice mode (`python main.py --voice`): say **"Hey Jarvis, what time is it"** (wake
phrase; MEDO answers) — the
state chips (`● listening → ● thinking → ● speaking`) light up and it replies aloud.
Works with the network unplugged for everything except the web skills.

Commands: `/models`, `/model <name>` (runtime model selection), `/stats`,
`/latency`, `/help`, `/quit`. One-shot: `python main.py --once "what time is it"`.

## Tests

```bash
pytest        # 59 tests: router precedence, safety gate, whitelist, tool
              # dispatch, offline degradation, memory/context, latency
```

## Design decisions

- **The Intent Router (fast path vs LLM path).** Deterministic requests match
  regex and execute with no model in the loop (sub-millisecond, works offline).
  Only open-ended requests pay for the LLM. This is what keeps a CPU-only machine
  usable — most commands never touch the 25 s model path.
- **One skill, two routes.** Each capability is a single `Skill` declaring both
  `patterns` (fast path) and a `tool_schema` (LLM function calling). The registry
  is the one source of truth, so a skill can't drift between the two routes.
- **Structured tool results are spoken directly.** For weather/news/time the
  skill already returns a clean sentence, so the router skips a second LLM
  "summarization" hop — faster, and it stops a small model from second-guessing a
  correct tool result. Only web search (raw results) needs the model to summarize.
- **Runtime model selection.** No hardcoded model; the app lists what Ollama has
  installed and you pick (`/model`), adapting to the hardware.
- **Cross-platform from day one.** OS-specific actions come from a per-platform
  table in `config.yaml` / `core/platform.py`, not scattered `sys.platform` checks.
- **Safety is structural, not advisory.** Destructive actions return
  `needs_confirmation` and the router physically won't run them without a spoken
  "yes"; file skills can only reach whitelisted directories.
- **Graceful degradation over crashes.** No Ollama, no network, a leaked
  tool-call blob from a weak model — each becomes a clear spoken message, never a
  traceback.

## Measured latency

Dev box: 2016 MacBook Pro, Intel i7-6700HQ, 16 GB RAM, **CPU-only** (Ollama does
not use the old AMD GPU). Warm (models resident). See live numbers with `/latency`.

| Stage | Budget | Measured |
|---|---|---|
| Wake word → listening | < 0.5 s | ~0 s (state transition) |
| Fast-path command → action | < 1.0 s | **~0.001 s** ✓ |
| STT (`small`, short command) | — | ~2.7 s (`base` ≈ 0.9 s, less accurate) |
| Piper TTS (one sentence) | — | ~0.14–1.0 s |
| LLM chat, warm (`llama3.2:3b`) | < 4 s first word | ~1.0 s ✓ |
| LLM **+ tool call** (decide tool → run) | < 4 s | **~25 s ✗** (see limitations) |

First LLM call after boot pays a one-time ~15 s model load; keep Ollama running to
stay warm. `qwen2.5:7b` is selectable via `/model` but is slower still on this CPU.

## Honest limitations

- **Tool-calling latency misses the 4 s budget on this hardware (~25 s).** The
  bottleneck is one llama3.2:3b inference to *choose* the tool on a 2016 CPU, not
  the architecture. On a GPU box, or with a smaller/faster tool-calling model, or
  by streaming the first sentence, this drops sharply. Fast-path commands (most of
  them) are unaffected and stay instant.
- **Windows support.** Developed on macOS but wired for Windows: app-launch,
  lock/sleep/shutdown/restart, screenshots, and media keys all have Windows
  branches, and `run.bat` fetches both the wake-word and Piper voice models on
  first launch. **Volume** now has precise get/set/mute on Windows via `pycaw`
  (Core Audio) — installed automatically from `requirements.txt` on Windows; if
  it's missing, volume degrades to a media-key nudge. Camera gestures (the vision
  sidecar) still need a first real-hardware run to confirm MediaPipe wheels.
- **Wake phrase is "Hey Jarvis", not "Hey MEDO".** openWakeWord ships pretrained
  models ("hey_jarvis" is built in); there's no "hey_medo". Renaming the assistant
  to MEDO doesn't change the acoustic trigger. A true "Hey MEDO" needs a custom
  model trained with openWakeWord's training notebook — a self-contained follow-up.
- **Wake-word / silence thresholds are defaults** (`wakeword.threshold`,
  `audio.silence_threshold`) and may need a one-line tune to your mic and room.
- **Web-search summary quality is bounded by the local 3B model** — occasionally
  generic. It's a summarizer, not a researcher.
- **Conversation memory is per-session** (in RAM); notes and reminders persist,
  chat context does not survive a restart by design.

## Skills (M2)

| Skill | Says | Safety |
|---|---|---|
| datetime | "what time is it", "what's the date" | — |
| timers | "set a timer for 5 minutes", "remind me in 10 seconds to stretch" | — |
| notes | "take a note …", "read my notes", "delete note 3" | SQLite |
| apps | "open chrome", "close spotify" | per-OS table in config |
| files | "find file report", "open the file notes.txt" | **whitelist only** |
| volume | "volume up", "set volume to 30", "mute" | — |
| system_info | "battery", "cpu usage", "system status" | read-only |
| screenshot | "take a screenshot" | — |
| power | "lock the screen", "shut down", "restart" | **confirmation** |
| weather | "what's the weather", "weather in Tokyo", "will it rain tomorrow" | offline-safe (Open-Meteo) |
| news | "give me the news", "headlines" | offline-safe (RSS) |
| web_search | "search for X and summarize it" | offline-safe (DuckDuckGo) |

**LLM path & tools (M3).** When no fast pattern matches, the router hands the
utterance to Ollama with every skill's `tool_schema` as a callable tool. The model
either answers directly or calls tools; results are fed back and it responds. The
same skill code serves both routes (`llm/tools.py`). Web skills never crash
offline — they return a spoken "I appear to be offline". A destructive tool the
model tries to call still requires your spoken "yes".

**Safety layer** (`core/safety.py`): destructive actions (shutdown/restart) return
`needs_confirmation`; the router holds the action and executes only on a spoken
"yes" (a "no" or anything ambiguous cancels). File skills route every path through
a `PathWhitelist`, so nothing outside `~/Documents`, `~/Downloads`, `~/Desktop`
(configurable) can be searched or opened — directory-traversal escapes included.

**Memory & personality (M4)** (`core/memory.py`, `llm/prompts.py`): the router
keeps a rolling window of recent turns (`memory.max_turns`) and prepends them to
the LLM prompt, so follow-ups resolve ("what's the weather" → "and tomorrow?" →
*tomorrow's* forecast). Notes and reminders persist in SQLite; reminders are
re-armed on startup. The MEDO prompt is concise and dryly witty, using "sir"
sparingly. Structured tool results (weather/news/time) are spoken directly rather
than re-summarized — faster, and immune to a small model second-guessing itself.

## Hand gestures & the M.E.D.O. HUD (M6)

**Gestures — a third input modality.** MediaPipe Hands watches the webcam; a
rotation-tolerant classifier (`vision/gestures.py`) maps 21 landmarks to a
gesture, and each confirmed gesture becomes an **utterance routed through the same
Intent Router as voice and text**. Defaults (config-editable):

| Gesture | Action |
|---|---|
| 👍 thumbs up | "yes" — confirm a pending action (drives the safety gate) |
| ✋ open palm | "no" — cancel |
| ✌ victory | take a screenshot |
| ☝ point up | volume up |
| ✊ fist | mute |

MediaPipe's mac/Windows wheels pin `numpy<2`, which conflicts with the voice
stack's numpy 2. Rather than compromise either, **vision runs as a separate
process in its own venv** (`.venv-vision`) and POSTs gestures to the companion API
— the same interface the watch app uses. Clean isolation, zero risk to the voice
pipeline. Run it manually with `.venv-vision/bin/python -m vision.run`.

**The HUD** (`ui/hud.py` + `ui/web/`) is an arc-reactor front end served over the
LAN. It subscribes to the event bus and streams state, transcript, routing path,
and replies to the browser via Server-Sent Events; the reactor core changes colour
with each state (`idle → listening → thinking → speaking`) and it embeds the
sidecar's annotated camera feed. Start with `python main.py --hud`.

```
Wake Word ┐
Text/REPL ├─▶ Intent Router ─┬─ FAST ─┐        EventBus ─▶ HUD (browser, SSE)
Gestures ─┘ (via API)        └─ LLM ──┴─▶ TTS  ─────────▶ ConsoleUI (terminal)
```

## Roadmap

- **M1** ✅ voice pipeline (wake word → STT → TTS)
- **M2** ✅ core PC skills + safety confirmation gate
- **M3** ✅ LLM tool calling + web skills (search/weather/news, offline-safe)
- **M4** ✅ memory + personality (rolling context, persistent reminders)
- **M5** ✅ terminal UI, latency instrumentation, full test suite, demo script
- **M6** ✅ camera hand-gestures (MediaPipe sidecar) + M.E.D.O. web HUD
- **v3** hardware skills (robotic dog, Arduino) — the registry is ready for them
