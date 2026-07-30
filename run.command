#!/bin/bash
# ============================================================================
# MEDO — one-click launcher (macOS/Linux). Double-click in Finder, or:
#   ./run.command
# Starts Ollama, the vision sidecar, and the app with voice + HUD + companion API.
# FIRST RUN installs everything: Ollama, Obsidian, the local AI models (~24 GB),
# both virtualenvs and the voice models. Later launches start straight up.
# Delete .medo-setup-done to force the one-time setup again.
# ============================================================================
cd "$(dirname "$0")" || exit 1
set -u

say() { printf "\033[36m[medo]\033[0m %s\n" "$1"; }

# The MEDO logo (the presence sphere), rendered for the terminal on every launch.
banner() {
  printf '\033]0;MEDO\007'                 # set the window/tab title
  printf '\033[36m'                        # MEDO blue
  cat <<'EOF'
        .  *  .
      *   (O)   *
        .  *  .

   __  __ ___ ___   ___
  |  \/  | __|   \ / _ \
  | |\/| | _|| |) | (_) |
  |_|  |_|___|___/ \___/

        local-first AI, on your hardware
EOF
  printf '\033[0m\n'
}
banner

PYBIN="$(command -v python3.12 || command -v python3)"
if [ -z "$PYBIN" ]; then echo "Python 3.12 not found. Install it first."; exit 1; fi

# Pull an Ollama model unless it's already present.
pull_model() {
  if ollama list 2>/dev/null | grep -qi "$1"; then
    say "  $1 already present — skipping."
  else
    say "  pulling $1 ($2)…"
    ollama pull "$1"
  fi
}

# 0. First-run provisioning: Ollama + Obsidian (one-time) --------------------
# Guarded by the .medo-setup-done marker; MEDO_FIRSTRUN also gates the model
# pulls in section 1. The marker is only written once qwen3:30b has landed, so
# an interrupted first run resumes next launch instead of being skipped.
FIRSTRUN=""
[ -f .medo-setup-done ] || FIRSTRUN=1
if [ -n "$FIRSTRUN" ]; then
  say "FIRST-RUN SETUP — installing Ollama, Obsidian and the local AI models (~24 GB, one-time)…"
  if ! command -v ollama >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1; then
      say "installing Ollama via Homebrew…"; brew install ollama
    else
      say "Ollama not installed and Homebrew is missing — get it from https://ollama.com/download, then re-run."
    fi
  fi
  if command -v brew >/dev/null 2>&1; then
    say "installing Obsidian (skipped if already installed)…"; brew install --cask obsidian 2>/dev/null || true
  else
    say "install Obsidian from https://obsidian.md (Homebrew missing)."
  fi
fi

# 1. Ollama server + models --------------------------------------------------
if command -v ollama >/dev/null 2>&1; then
  pgrep -x ollama >/dev/null 2>&1 || { say "starting Ollama…"; (ollama serve >/tmp/medo-ollama.log 2>&1 &); sleep 2; }
  if [ -n "$FIRSTRUN" ]; then
    say "pulling the local models MEDO uses (smallest first, so it is usable quickly)…"
    pull_model nomic-embed-text "semantic routing + memory"
    pull_model llama3.2:3b      "fast fallback brain"
    pull_model qwen2.5vl:3b     "vision — what do you see"
    pull_model qwen3:30b        "main brain, 18 GB — be patient"
  else
    ollama list 2>/dev/null | grep -qi "qwen3" || say "WARNING: no qwen3 model found — pull it: ollama pull qwen3:30b"
  fi
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

# 3b. Mark first-run setup complete — only once qwen3:30b has actually landed,
#     so an interrupted 18 GB download resumes next launch. Don't want the big
#     model? Run 'touch .medo-setup-done' yourself to skip it.
if [ -n "$FIRSTRUN" ]; then
  if ollama list 2>/dev/null | grep -qi "qwen3:30b"; then
    touch .medo-setup-done; say "first-run setup complete — later launches start straight up."
  else
    say "setup will finish downloading the 18 GB brain on the next launch (or: touch .medo-setup-done to skip it)."
  fi
fi

# 4. Launch: vision sidecar (background) + open HUD + run the app ------------
say "starting vision sidecar…"
( ./.venv-vision/bin/python -m vision.run ) & VIS=$!
( sleep 6; open "http://localhost:8730" >/dev/null 2>&1 ) &
say "opening HUD at http://localhost:8730 — say “Hey MEDO”. Ctrl-C to quit."
trap 'kill $VIS 2>/dev/null' EXIT
./.venv/bin/python main.py --voice --hud --serve
