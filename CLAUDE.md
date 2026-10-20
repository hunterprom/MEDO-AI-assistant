# MEDO — project context for Claude Code

Local-first, two-language (English/Macedonian) AI desktop assistant. Voice +
gesture + LLM, running entirely on the user's own hardware. Python 3.12.

Repo: `github.com/hunterprom/MEDO` · working branch `medo-main-work`.

---

## Orientation

Start with `README.md` (overview) → `README-detailed.md` (every feature, ~25 KB)
→ `docs/architecture.md`. The `docs/` folder is an Obsidian vault and is the
authoritative design record — prefer it over inferring intent from code.

| Doc | What's in it |
|---|---|
| `docs/architecture.md` | Pipeline, processes/venvs, HUD contract (frozen) |
| `docs/decisions.md` | Why things are the way they are — read before refactoring |
| `docs/security.md` | Policy engine, trust boundary, threat model |
| `docs/runbook.md` | Operational procedures |
| `docs/bug-log.md`, `docs/roadmap.md` | Known issues, planned work |
| `docs/software-connectors.md`, `docs/packaging.md`, `docs/maker-studio.md`, `docs/medo-link.md` | Subsystem docs |
| `OPUS-FIXLIST.md` | Live review findings + open items |

---

## Runtime pipeline

```
mic ─► openWakeWord ─► record until silence ─► faster-whisper (auto mk/en)
                                                     │
          HUD :8730 (SSE observer) ◄── EventBus ◄────┤
                                                     ▼
watch app ─► companion API :8710 /ask ────────► Intent Router
vision sidecar gestures ─► POST /ask ─────────►   │
                                                  ├─ FAST: regex skill (<1 ms)
                                                  ├─ SEMANTIC: local embeddings, learns from usage
                                                  └─ LLM: bounded tool loop (≤4 rounds, confirmation gate)
                                                     │
                                            Piper TTS ─► speakers
```

Three routing tiers, in order: FAST regex (~30 commands) → SEMANTIC (embedding
match, self-improving via `core/route_memory.py`) → LLM tool loop.

### Processes & virtualenvs

| Process | Venv | Why separate |
|---|---|---|
| `main.py --voice --hud --serve` | `.venv` | voice stack, aiohttp servers |
| `vision/run.py` sidecar | `.venv-vision` | MediaPipe pins numpy<2 |
| `ollama serve` | — | needs `OLLAMA_MODELS`, `GGML_CUDA_NO_PINNED=1` |

The sidecar is dependency-light and **never imports the main app** — it talks
only via `POST :8710/ask`; the app talks back only via `:8731/pointer` and
`/frame.jpg`. Keep that boundary intact.

### Ports (LAN-only; TLS available on 8710, off by default — never forward beyond LAN)

`8730` HUD · `8710` companion API · `8731` vision sidecar · `11434` Ollama

---

## Layout

| Path | Role |
|---|---|
| `main.py` (52 KB) | Entry point. `build_registry()` wires every skill; `async_main()` starts voice/HUD/serve. Flags: `--once`, `--voice`, `--hud`, `--serve` |
| `core/` (32 files, ~8.4k lines) | Engine. `router.py` (1518 L) is the heart; `config.py` (1724 L) is the settings model |
| `skills/` (42 files) | One skill per file, subclassing `skills/base.py`. Each carries regex `patterns` **and** a `tool_schema` — one implementation serves both routing paths |
| `llm/` | `client.py` (813 L) — provider abstraction: Ollama (default), OpenAI-compatible, Claude API, Claude Code CLI, Codex CLI |
| `security/` | `policy.py` deny-by-default engine, `trust.py` user-instruction-vs-content boundary, `capabilities.py`, `audit.py`, `secrets.py` |
| `voice/` | Wake word, STT, TTS, audio loop, wake-word training |
| `vision/` | Sidecar: camera, gestures, pointer, per-OS mouse backends |
| `ui/` | `hud.py` + `web/` static HUD, `overlay.py`, `console.py` |
| `remote/server.py` (1803 L) | Companion API — ~40 aiohttp routes |
| `studio/` | Maker Studio — schematics, 3D models, code building, sandbox |
| `software/` | Direct local app control (not MCP): win32 backend, accessibility, UI scan, learned connectors |
| `app/`, `winsetup/` | Packaging: tray launcher, supervisor, first-run wizard, Inno Setup installer |
| `link/`, `packages/medo_link` | MEDO Link |
| `mobile/`, `phone/`, `watch/` | Flutter apps (Wear OS watch, phone) |
| `plugins/` | Drop-in `.py` plugins, capability-reviewed at load |
| `tests/` | 131 files, ~24k lines |

---

## Key subsystems

- **Council** (`core/council.py`, `skills/council.py`) — 18 experts, each a
  system prompt, so convening costs no extra Ollama processes.
- **Second opinion** (`skills/second_opinion.py`) — "are you sure?" makes the
  council red-team MEDO's *own* last answer, returning a deterministic
  fail-safe verdict that retracts unsupported claims.
- **Policy engine** (`security/policy.py`) — `PolicyEngine.check()` /
  `gate_skill()` gate **every** action. `Effect`/`Actor`/`Provenance` enums;
  `Decision` exposes `allowed` / `denied` / `needs_confirmation`.
- **Trust boundary** (`security/trust.py`) — content (documents, web pages)
  can never be promoted to instructions. Preserve this in any change touching
  document or web ingestion.
- **Bilingual** (`core/languages.py`, `core/reply_language.py`, `lang/`) —
  replies go back in the language asked. Macedonian regressions have bitten
  this repo repeatedly (see git log "round 5"); test both languages.

---

## Configuration

`config.yaml` (48 KB) configures everything — one file, ~30 top-level sections
(`llm`, `router`, `conversation`, `personality`, `vision`, `hud`, `languages`,
`stt`, `tts`, `wakeword`, `audio`, `safety`, `skills`, `browser`, `council`,
`mcp`, `memory`, `routines`, …). Every key is env-overridable with the `MEDO_`
prefix. The schema lives in `core/config.py`.

`secrets.local.yaml` holds API keys + the companion auth token. Gitignored
(with a trailing `*` so `.bak` can't slip past). **Never read, echo, or commit
its values.**

---

## Commands

```bash
.venv\Scripts\python -m pytest        # tests (Windows)
.venv/bin/python -m pytest            # tests (macOS/Linux)
python main.py --once "what time is it"   # route one request and exit
```

Launch: `run.bat` (Windows) / `run.command` (macOS/Linux). First run is a
~24 GB one-time setup (Ollama, Obsidian, four local models, both venvs, voice
models); `.medo-setup-done` marks it complete. `medo-lite.py` + `run-lite.command`
are the minimal path.

`pytest.ini`: `asyncio_mode = auto`, `testpaths = tests`.

---

## Conventions & gotchas

- **Skills are dual-path.** Adding a skill means regex `patterns` *and*
  `tool_schema`, registered in `main.py::build_registry()`. Don't add one path
  only.
- **The HUD contract is frozen** (`docs/architecture.md`). Chat bubbles render
  **only** from SSE, so replies never appear twice. Don't add a second render path.
- **Never commit** `*.db` (`medo.db`, `jarvis.db` hold personal data),
  `secrets.local.yaml*`, `models/*`, `screenshots/`, logs.
- **Don't import the main app from `vision/`** — the sidecar boundary is
  deliberate.
- `MEDO-main/` at the repo root is an empty nested leftover, not a module.
- Legacy naming: the project was renamed jarvis → medo; `jarvis.db` and a
  `_migrate_db_name()` path in `main.py` remain.
- Optional deps degrade gracefully rather than crash (e.g. no Playwright →
  browser skills speak a message; no Piper for a language → Edge TTS → English).
