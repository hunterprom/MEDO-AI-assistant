"""The end-user application layer: one launcher that owns every MEDO process.

`app.launcher` is the single entry point a non-technical user runs (via the
installed shortcut / tray). It supervises three OS processes — Ollama, the MEDO
engine (which itself hosts the HUD + companion API and reaps the overlay), and
the vision sidecar — restarts a dead child with backoff, surfaces a friendly
error instead of a traceback, and tears everything down cleanly on Quit.

The supervision LOGIC lives in `app.supervisor` (pure, injectable, unit-tested);
`app.launcher` is the thin wiring (real subprocesses, the pystray tray, opening
the browser). The dev workflow (run.bat + config.yaml) is untouched — this layer
is additive.
"""
