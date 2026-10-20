<p align="center">
  <img src="docs/img/medo-logo.png" alt="MEDO" width="128" height="111">
</p>

# MEDO — a local-first AI desktop assistant

**MEDO is a two-language (English/Macedonian) voice + gesture assistant that runs
entirely on your own hardware** — a custom "medo" wake word trained on my own
voice, Whisper ears, a swappable LLM brain, on-device neural voices,
MediaPipe hand-tracking, a sci-fi HUD, and a bench of expert "council" agents it
can put a question to — or turn loose on its *own* answers.

> One codebase, no cloud required. → **[Full guide: README-detailed.md](README-detailed.md)**

---

## Highlights

- 🎙 **Voice, your voice** — custom "medo"/"hey medo" wake word → faster-whisper →
  neural TTS, spoken back in the language you asked in. Barge-in to interrupt.
  Fifteen of the sixteen voices run **on your CPU** (sherpa-onnx, ~150–300 ms,
  zero VRAM); Macedonian is the one declared cloud voice, until a permissively
  licensed local one exists. → **[Voice](docs/voice.md)**
- 🧠 **Five swappable brains** — Ollama (local, default), GPT / any
  OpenAI-compatible API, the Claude API, and the Claude Code & Codex CLI agents —
  switched live from the HUD.
- ⚡ **Three routing tiers** — a **FAST** regex path (<1 ms, ~30 commands), a
  **SEMANTIC** tier that reaches a skill by *meaning* (local embeddings, and it
  *learns* from your usage), then the **LLM** in a bounded tool loop.
- 👥 **A council of 18 experts** — "ask the electrical engineer about grounding"
  or "convene the council on battery sizing"; each discipline is a system prompt,
  so convening costs no extra processes on Ollama.
- 🔎 **Second opinion** — say *"are you sure?"* and the council red-teams MEDO's
  **own** last answer, tagging each claim and returning a **deterministic,
  fail-safe verdict** that retracts anything unsupported. It never talks itself
  into false confidence.
- 🖐 **Gestures & sight** — drive the cursor by hand (pinch-click, fist to exit);
  "read my screen" / "what do you see" via a local vision model.
- 🌐 **Does things** — searches sites where you actually search, *drives* Chrome
  through the DOM, remembers facts semantically, and makes documents,
  presentations, spreadsheets and small apps.
- 🎮 **Controls other apps** — "play/pause", "minimize", "new tab", "close
  Notepad" by **direct local control** (not MCP), each gated by the policy engine.
- 🔧 **Builds real engineering artifacts** — "design a circuit for an ESP32 and an
  LED", "3D print a phone stand": the brain writes **library code** (SchemDraw,
  CadQuery) and MEDO **runs it in a sandbox**, so what comes back is a valid
  schematic and a printable STL — not a picture of one. Versioned, and edited by
  saying what to change. → **[Maker Studio](docs/maker-studio.md)**
- 🔁 **Routines** — chain skills in a YAML file, fired by a phrase or at 07:30, with
  no Python. A step that needs confirmation is refused when the file loads:
  composition never becomes privilege.
- 🧭 **Cowork** *(off by default)* — hand over a whole multi-step task and MEDO
  plans it **out loud**, lets you edit the plan by voice, confirms the risky
  steps as they come, runs one at a time — and **stops** if one fails instead of
  improvising. Bounded, not autonomous: six steps, real capabilities only, and
  what it reads can never become what it does.
  → **[Cowork](docs/cowork.md)**
- 🛡 **Safe by default** — one **deny-by-default policy engine** gates every
  action, behind a user-instruction-vs-content **trust boundary** that stops a
  document or web page from turning injected text into a command; secrets stay
  local, and a cloud brain gets your question, not your files.
  → **[Security](docs/security.md)**
- 🧩 **Extensible** — drop a `.py` in `plugins/` (capability-reviewed), plug in
  any **MCP** server, pair the Wear OS **watch app**, or POST a manifest from an
  ESP32/Pi and **[MEDO Link](docs/medo-link.md)** turns each capability into a
  voice command *and* an LLM tool — anything that moves confirms first. A HUD at
  `localhost:8730` renders it all live.

## Install

**Prerequisites:** Python 3.12, and a package manager for the automatic app
installs — **winget** (built into Windows 11) or **Homebrew** (macOS/Linux).
Leave ~30 GB free for the local models; an NVIDIA GPU is nice but not required.

```
git clone https://github.com/hunterprom/MEDO.git
cd MEDO
```

Then double-click **`run.bat`** (Windows) or **`run.command`** (macOS/Linux). The
**first** launch is a one-time setup (~24 GB, be patient): it installs Ollama +
Obsidian, pulls the four local models (`nomic-embed-text`, `llama3.2:3b`,
`qwen2.5vl:3b`, `qwen3:30b`), builds the two virtualenvs and the voice models,
then opens the HUD at <http://localhost:8730>. Every later launch skips setup and
boots straight to the app. Say **"medo"**, or type into the HUD.

No winget/Homebrew? Install [Ollama](https://ollama.com/download) and
[Obsidian](https://obsidian.md) by hand and re-run. Full walkthrough:
**[README-detailed.md → Install & first run](README-detailed.md#install--first-run)**.

**Packaging it for someone non-technical?** An app layer (`app/` + `winsetup/`)
builds a one-click Windows installer — a tray launcher that supervises every
process, hardware-fit model selection, and a first-run wizard with progress bars,
no terminal. See **[docs/packaging.md](docs/packaging.md)**.

```
.venv\Scripts\python -m pytest      # run the tests
```

## Local-first & LAN-only

Everything that thinks runs here: speech recognition, the default brain,
embeddings, vision, and the voice in fifteen of sixteen languages. The two
exceptions are ones you switch on — a cloud brain (`llm.provider`), and the
Macedonian voice (`tts.allow_cloud: false` shows those replies instead of
speaking them).

Never forward these ports beyond your LAN. The companion API requires a bearer
token for LAN clients, rate-limits each caller, and refuses cross-origin browser
requests; the HUD serves that token to loopback only; the vision sidecar binds
`127.0.0.1`. **TLS is available** (`remote.tls_enabled`, self-signed on first
run) and off by default — worth turning on for any shared network, since
without it the token crosses the wire in the clear.

| Port | What |
|------|------|
| 8730 | HUD |
| 8710 | Companion API (`/ask`, `/status`, provider/model switch, …) |
| 8731 | Vision sidecar (webcam MJPEG, pointer toggle) |
| 11434 | Ollama |

## Learn more

- **[README-detailed.md](README-detailed.md)** — every feature, the security
  model, configuration, and troubleshooting.
- **[`docs/security.md`](docs/security.md)** — the deny-by-default policy engine,
  trust boundary, and honest threat model ·
  **[`docs/software-connectors.md`](docs/software-connectors.md)** ·
  **[`docs/cowork.md`](docs/cowork.md)** ·
  **[`docs/voice.md`](docs/voice.md)** ·
  **[`docs/medo-link.md`](docs/medo-link.md)** ·
  **[`docs/maker-studio.md`](docs/maker-studio.md)** ·
  **[`docs/packaging.md`](docs/packaging.md)**.
- **[`docs/`](docs/)** — an Obsidian vault: architecture, runbook, bug log,
  decisions, roadmap.
- **[`config.yaml`](config.yaml)** — one file configures everything
  (env-overridable with the `MEDO_` prefix).
