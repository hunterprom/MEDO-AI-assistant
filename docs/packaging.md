# Packaging MEDO into a one-click Windows app

Goal: a non-technical user double-clicks an installer, follows a plain-language
wizard, and ends up with a working, hardware-fitted MEDO — no terminal, no
`config.yaml`, no manual Ollama setup. The dev workflow (`run.bat` + `config.yaml`)
is untouched; this is an additive **app layer** (`app/`) plus **winsetup/**.

## Architecture (what gets built)

Three frozen executables (see `winsetup/manifest.py`):

| Exe | From | Role |
|---|---|---|
| **`MEDO.exe`** | `app/launcher.py` | The user-facing app: tray icon + **supervisor**. Starts/stops the other two, ensures Ollama, opens the HUD in the browser. |
| `medo-engine.exe` | `main.py` | The engine — hosts the HUD + companion API, owns voice/STT/TTS. |
| `medo-vision.exe` | `vision/run.py` | The hand-gesture sidecar. **Separate bundle on purpose.** |

Installed layout: `C:\Program Files\MEDO\MEDO.exe`, `…\engine\`, `…\vision\`.

## The dependency fights (and how they're handled)

- **MediaPipe pins `numpy<2`** while the engine uses newer numpy — they *cannot*
  share one environment. This is why there are two venvs today and why the vision
  sidecar is frozen **separately** (`medo-vision.spec`, built with the
  `.venv-vision` PyInstaller). Never try to merge vision into the engine bundle.
- **MediaPipe hides its `.binarypb` graphs + native modules** from PyInstaller's
  analyzer → the frozen sidecar dies with "graph not found". Fixed by
  `collect_data_files("mediapipe", include_py_files=True)` in the vision spec.
- **faster-whisper / onnxruntime / ctranslate2** hide their backends → collected
  via `collect_submodules` + `collect_data_files("onnxruntime")` in the engine
  spec.
- **sounddevice / PortAudio**: the PortAudio DLL ships inside the `sounddevice`
  wheel; `hiddenimports=["sounddevice","_sounddevice_data"]` pulls it in. Test
  audio on a clean machine (no Python) — this is the most common "works on my
  box" gap.
- **HUD assets**: the aiohttp HUD serves files from `ui/`; bundled via
  `datas=[("ui","ui")]`. The confirm-word banks (`lang/`) and the wakeword model
  (`models/wakeword/`) are bundled too.
- **The LLM + whisper models are NOT bundled** — the first-run wizard downloads
  them per hardware profile (keeps the installer to a sane size). The piper voice
  (~60 MB) may be bundled or made a first-run download; default is bundled.

## Build

```bat
:: from the repo root, both venvs present, PyInstaller installed in each
python winsetup\build.py
```

This freezes engine + launcher (main venv), vision (vision venv), then runs Inno
Setup (`iscc winsetup\installer.iss`) → `dist\MEDO-Setup.exe`.

## Code signing / SmartScreen

Unsigned, Windows **SmartScreen** shows "Windows protected your PC" on first run —
users must click *More info → Run anyway*. To avoid it, sign both
`dist\MEDO\MEDO.exe` and `MEDO-Setup.exe` with `signtool` and an OV/EV
certificate, then set `SignTool=` in `installer.iss`. **Placeholder left on
purpose** — no cert is assumed. Until signed, tell testers to expect the warning
(it is not a virus flag, just unsigned-publisher).

## Release checklist

1. `python -m pytest` green (incl. `tests/test_packaging_manifest.py`).
2. Verify model tags/sizes against ollama.com/library (they change) — update
   `app/profiles.py` if needed.
3. Confirm the Ollama Windows installer URL + `/VERYSILENT` flag and the
   `/api/pull` stream shape (`app/setup_ops.py`) still hold.
4. `python winsetup\build.py`.
5. Sign the exes (or note SmartScreen for testers).
6. **Manual install test (below) on a clean Windows VM with no Python/Ollama.**

## Manual install-test checklist (clean machine, no Python/Ollama)

- [ ] Installer runs; creates Start-menu **MEDO** + desktop **MEDO** shortcuts.
- [ ] First launch opens the **wizard** in the browser (no terminal, no traceback).
- [ ] Wizard installs Ollama (with progress), downloads the profile's model (progress + size), tests the mic, ends on the "try your wake word" screen.
- [ ] Hardware profile chosen matches the machine (weak laptop → `lite`).
- [ ] Say the wake word → MEDO responds. Gestures work (vision sidecar up).
- [ ] Tray: **Open / Settings / Quit**. Quit leaves **no** orphaned `MEDO*`/`ollama`/`python` processes (Task Manager).
- [ ] Kill `medo-engine.exe` in Task Manager → supervisor restarts it; kill it repeatedly → a friendly error, not a crash.
- [ ] Re-run setup from Settings works.
- [ ] Uninstaller removes the app and *asks* before deleting models/settings.

## This is direct packaging, not a container

Everything runs as native Windows processes the supervisor owns — no Docker, no
service. Ollama is the one external dependency, installed by the wizard and left
running only if MEDO started it.
