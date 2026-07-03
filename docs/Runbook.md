# Runbook

## Start

`run.bat` (double-click). It sets the Ollama env vars, probes/starts
`ollama serve`, warns if `qwen3:30b`/`moondream` are missing (never
auto-pulls the 18 GB model), creates venvs on first run, downloads the Piper
voice (with `--ssl-no-revoke` — schannel curl fails CRL checks on this
network), starts the sidecar, and opens the HUD.

## Health checks

```powershell
curl http://127.0.0.1:8730/                 # HUD page
curl http://127.0.0.1:8710/status           # provider, model, state
curl http://127.0.0.1:8710/sys              # cpu / ram / gpu telemetry
curl http://127.0.0.1:8731/pointer          # sidecar alive + pointer state
curl -X POST http://127.0.0.1:8710/ask -H "Content-Type: application/json" -d "{\"text\":\"what time is it\"}"
```

## Stop

Close the `run.bat` console (main app), the "MEDO Vision" window (sidecar).
Leftovers: `taskkill /IM python.exe /F` (careful) or by PID from
`netstat -ano | findstr :8710`.

## Common failures

| Symptom | Cause → fix |
|---|---|
| HUD `ERR_CONNECTION_REFUSED` | main app died at startup — read the console; most often a missing model file |
| No speech, replies text-only | Piper voice missing → rerun `run.bat` (downloads it), check the WARNING line |
| LLM replies "can't reach my language model" | Ollama down or empty `ollama list` → env var `OLLAMA_MODELS=D:\OllamaModels` not set in that shell; use `run.bat` |
| qwen3:30b OOM on load ("CUDA_Host buffer") | `GGML_CUDA_NO_PINNED=1` missing; close VRAM-hungry apps (animated wallpaper) |
| First LLM reply after "what do you see" is slow | moondream evicted the 30B from VRAM — expected, one reload |
| Camera feed black / sidecar exits | camera in use by another app, or camera index wrong (`vision.camera_index`) |
| Pointer won't turn on | sidecar not running, or `vision.pointer.enabled: false` |
| "volume"/"mute" does nothing | pycaw/comtypes missing from `.venv` → `pip install -r requirements.txt` |
| Whisper mishears Macedonian | it's auto-detect; pin `stt.language: mk` (or `en`) if you speak one language |

## Network TLS quirk (this machine)

Something on this network intercepts TLS: python `requests` fails with
`CERTIFICATE_VERIFY_FAILED` (breaks openwakeword's model download on a fresh
venv) and schannel curl needs `--ssl-no-revoke` (run.bat already passes it).
If the wake-word download fails, copy the models from any previous install:

    robocopy <old-venv>\Lib\site-packages\openwakeword\resources ^
             .venv\Lib\site-packages\openwakeword\resources /E

## Ollama with the wrong model store

If Ollama was auto-started (tray icon) without `OLLAMA_MODELS=D:\OllamaModels`
it sees zero models — `/status` shows `"models": []` and vision replies claim
moondream is missing. `run.bat` detects this (API up, qwen3 invisible, D:
store present) and restarts the daemon with the right environment.

## Changing things

- **Model**: HUD → CONFIG → Language model, or `POST /model`, or `config.yaml`.
- **Wake sensitivity**: `wakeword.threshold` (0–1, higher = stricter).
- **Gesture map**: `vision.gestures` in `config.yaml`.
- **Pointer feel**: `vision.pointer.sensitivity` (reach), `ema_alpha`
  (steadiness — lower is smoother), `click_debounce_ms`.
- **Facts cap**: `memory.max_facts` (how many go into the system prompt).
