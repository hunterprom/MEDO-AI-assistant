#!/bin/bash
# MEDO lite launcher (macOS/Linux) — double-click in Finder, or ./run-lite.command
# Same setup as run.command but starts the low-resource lite profile:
# vision off, whisper tiny/int8, smallest local model. Pass --hud to add the HUD.
cd "$(dirname "$0")" || exit 1
set -u

say() { printf "\033[36m[medo-lite]\033[0m %s\n" "$1"; }

PYBIN="$(command -v python3.12 || command -v python3)"
[ -z "$PYBIN" ] && { echo "Python 3.12 not found. Install it first."; exit 1; }

# Ollama (optional): start it if present; lite auto-picks the smallest model.
if command -v ollama >/dev/null 2>&1; then
  pgrep -x ollama >/dev/null 2>&1 || { say "starting Ollama…"; (ollama serve >/tmp/medo-ollama.log 2>&1 &); sleep 2; }
else
  say "Ollama not installed — LLM path offline (fast-path still works)."
fi

# Main venv only (lite doesn't run the vision sidecar).
if [ ! -d .venv ]; then
  say "creating venv + installing deps (one-time)…"
  "$PYBIN" -m venv .venv && ./.venv/bin/pip install -q -r requirements.txt
  ./.venv/bin/python -c "import openwakeword.utils as u; u.download_models()" 2>/dev/null
fi

say "starting lite — vision off, whisper tiny/int8. Ctrl-C to quit."
exec ./.venv/bin/python medo-lite.py --voice "$@"
