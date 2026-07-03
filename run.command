#!/bin/bash
# ============================================================================
# Jarvis v2 — one-click launcher (macOS/Linux). Double-click in Finder, or:
#   ./run.command
# Starts Ollama, the vision sidecar, and the app with voice + HUD + companion API.
# First run sets up both virtualenvs and installs all dependencies.
# ============================================================================
cd "$(dirname "$0")" || exit 1
set -u

say() { printf "\033[36m[jarvis]\033[0m %s\n" "$1"; }

PYBIN="$(command -v python3.12 || command -v python3)"
if [ -z "$PYBIN" ]; then echo "Python 3.12 not found. Install it first."; exit 1; fi

# 1. Ollama server + a model ------------------------------------------------
if command -v ollama >/dev/null 2>&1; then
  pgrep -x ollama >/dev/null 2>&1 || { say "starting Ollama…"; (ollama serve >/tmp/jarvis-ollama.log 2>&1 &); sleep 2; }
  ollama list 2>/dev/null | grep -q . || { say "pulling llama3.2:3b (one-time)…"; ollama pull llama3.2:3b; }
else
  say "Ollama not installed — the LLM path will be offline (fast path still works)."
fi

# 2. Main venv (voice + app) ------------------------------------------------
if [ ! -d .venv ]; then
  say "creating main venv + installing deps (one-time)…"
  "$PYBIN" -m venv .venv && ./.venv/bin/pip install -q -r requirements.txt
  ./.venv/bin/python -c "import openwakeword.utils as u; u.download_models()" 2>/dev/null
fi

# 2b. Piper voice model (spoken replies; --voice needs it) ------------------
if [ ! -f models/piper/en_US-lessac-medium.onnx ]; then
  say "downloading Piper voice (one-time)…"
  mkdir -p models/piper
  base="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
  curl -fL -o models/piper/en_US-lessac-medium.onnx "$base/en_US-lessac-medium.onnx"
  curl -fL -o models/piper/en_US-lessac-medium.onnx.json "$base/en_US-lessac-medium.onnx.json"
fi

# 3. Vision venv (hand gestures — separate because MediaPipe pins numpy<2) ---
if [ ! -d .venv-vision ]; then
  say "creating vision venv + installing deps (one-time)…"
  "$PYBIN" -m venv .venv-vision
  ./.venv-vision/bin/pip install -q --only-binary=:all: -r requirements-vision.txt
fi

# 4. Launch: vision sidecar (background) + open HUD + run the app ------------
say "starting vision sidecar…"
( ./.venv-vision/bin/python -m vision.run ) & VIS=$!
( sleep 6; open "http://localhost:8730" >/dev/null 2>&1 ) &
say "opening HUD at http://localhost:8730 — say “Hey Jarvis”. Ctrl-C to quit."
trap 'kill $VIS 2>/dev/null' EXIT
./.venv/bin/python main.py --voice --hud --serve
