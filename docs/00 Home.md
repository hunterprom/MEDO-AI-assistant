# MEDO — Vault Home

The knowledge base for the **MEDO super project** (v1 jarvis-web + v2 Python,
merged 2026-07-03). Open this folder as a vault in Obsidian.

## Map

- [[Architecture]] — how the pieces fit: pipeline, ports, venvs, skills
- [[Runbook]] — start/stop, health checks, common failures and their fixes
- [[Bug Log]] — every bug found and fixed during the merge, plus known limits
- [[Roadmap]] — what's next (streaming TTS, "hey MEDO" wake word, mk voice…)
- [[Decisions]] — why things are the way they are

## Quick start

1. Double-click `run.bat` in the project root.
2. Wait for the HUD at `http://localhost:8730`.
3. Say **"hey MEDO"** (or just **"medo"**) or type a command in COMMS.

## Vitals

| | |
|---|---|
| Brain | qwen3:30b (Ollama, D:\OllamaModels) |
| Vision LLM | moondream |
| STT | faster-whisper `small`, auto language (mk + en) |
| TTS | Piper `en_US-lessac-medium` |
| Wake word | **"hey MEDO"** / "medo" — custom openWakeWord model, trained on the owner's voice (`models/wakeword/hey_medo.onnx`) |
| Ports | HUD 8730 · API 8710 · camera 8731 · Ollama 11434 |
