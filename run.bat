@echo off
REM ============================================================================
REM  MEDO - one-click launcher (Windows). Double-click run.bat.
REM  Starts Ollama (with the right env), the vision sidecar (gestures, pointer
REM  mode, camera stream), and the app with voice + HUD + companion API.
REM  FIRST RUN installs everything: Ollama, Obsidian, the local AI models
REM  (~24 GB), both virtualenvs and the voice models. Later launches start
REM  straight up. Delete .medo-setup-done to force the one-time setup again.
REM ============================================================================
cd /d "%~dp0"
setlocal
title MEDO

REM --- MEDO logo (the presence sphere, rendered for the console) ---------------
echo(
echo         .  *  .
echo       *   (O)   *
echo         .  *  .
echo(
echo    __  __ ___ ___   ___
echo   ^|  \/  ^| __^|   \ / _ \
echo   ^| ^|\/^| ^| _^|^| ^|) ^| (_) ^|
echo   ^|_^|  ^|_^|___^|___/ \___/
echo(
echo         local-first AI, on your hardware
echo(

REM 0. Ollama environment ------------------------------------------------------
REM    Model store lives on D: and pinned CUDA host buffers must be off, or a
REM    partially-offloaded 30B intermittently OOMs on load ("failed to allocate
REM    CUDA_Host buffer"). Both are persistent user env vars too; this is the
REM    in-session backup so the launcher works in any shell.
set "OLLAMA_MODELS=D:\OllamaModels"
set "GGML_CUDA_NO_PINNED=1"

REM 0b. First-run provisioning: Ollama + Obsidian (one-time) --------------------
REM    Guarded by the .medo-setup-done marker so we don't re-run winget or
REM    re-download 24 GB of models on every launch. The marker is only written
REM    once the big brain (qwen3:30b) has landed, so an interrupted first run
REM    simply resumes next time. MEDO_FIRSTRUN also gates the model pulls below.
set "MEDO_FIRSTRUN="
if not exist ".medo-setup-done" set "MEDO_FIRSTRUN=1"
if not defined MEDO_FIRSTRUN goto firstrun_done
echo [medo] ============================================================
echo [medo]  FIRST-RUN SETUP - installing Ollama, Obsidian and the local
echo [medo]  AI models. This downloads ~24 GB and can take a while. It
echo [medo]  runs only ONCE; later launches skip straight to the app.
echo [medo] ============================================================
where winget >nul 2>&1
if not %errorlevel%==0 (
  echo [medo] winget not found - install Ollama from https://ollama.com/download
  echo [medo]   and Obsidian from https://obsidian.md, then re-run. Continuing...
  goto firstrun_done
)
where ollama >nul 2>&1
if not %errorlevel%==0 (
  echo [medo] installing Ollama via winget...
  winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements --silent
  REM winget only updates PATH for NEW shells; add it for THIS session so the
  REM model pulls below can reach ollama right away.
  set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Ollama"
)
echo [medo] installing Obsidian via winget ^(skipped if already installed^)...
winget install -e --id Obsidian.Obsidian --accept-package-agreements --accept-source-agreements --silent
:firstrun_done

REM 1. Ollama server + model checks ---------------------------------------------
REM Flat goto flow: %errorlevel% inside parenthesized blocks expands at parse
REM time in batch, which silently breaks nested checks.
where ollama >nul 2>&1
if not %errorlevel%==0 (
  echo [medo] Ollama not installed - LLM path offline; fast-path skills still work.
  goto ollama_done
)
curl -s --max-time 2 http://127.0.0.1:11434/api/tags >nul 2>&1
if not %errorlevel%==0 goto ollama_start
REM A daemon is up. If it can't see qwen3 while the D: store exists, it was
REM started without OLLAMA_MODELS (e.g. tray autostart) - restart it with env.
ollama list 2>nul | findstr /i /c:"qwen3" >nul
if %errorlevel%==0 goto ollama_ready
if not exist "D:\OllamaModels\manifests" goto ollama_ready
echo [medo] Ollama is running without the D: model store - restarting it...
REM Kill the tray app FIRST: it silently respawns ollama.exe without
REM OLLAMA_MODELS, which is exactly the zero-models state we're fixing.
taskkill /im "ollama app.exe" /f >nul 2>&1
taskkill /im ollama.exe /f >nul 2>&1
timeout /t 2 >nul
:ollama_start
echo [medo] starting Ollama...
start "" /b ollama serve
timeout /t 3 >nul
:ollama_ready
if defined MEDO_FIRSTRUN (
  echo [medo] pulling the local models MEDO uses ^(smallest first, so it is usable quickly^)...
  call :pull_model nomic-embed-text "semantic routing + memory"
  call :pull_model llama3.2:3b "fast fallback brain"
  call :pull_model qwen2.5vl:3b "vision - what do you see"
  call :pull_model qwen3:30b "main brain, 18 GB - be patient"
) else (
  ollama list 2>nul | findstr /i /c:"qwen3:30b" >nul || echo [medo] WARNING: qwen3:30b not found - LLM path uses whatever is installed ^(pull it: ollama pull qwen3:30b^).
  ollama list 2>nul | findstr /i /c:"qwen2.5vl" >nul || echo [medo] WARNING: qwen2.5vl:3b missing - "what do you see" is offline. Install: ollama pull qwen2.5vl:3b
)
:ollama_done

REM 2. Main venv -----------------------------------------------------------------
if not exist ".venv\" (
  echo [medo] creating main venv + installing deps ^(one-time^)...
  py -3.12 -m venv .venv
  call .venv\Scripts\pip install -q -r requirements.txt
  call .venv\Scripts\python -c "import openwakeword.utils as u; u.download_models()" || (
    echo [medo] WARNING: wake-word model download failed ^(this network's TLS
    echo [medo]          interception breaks python requests^). Voice wake won't
    echo [medo]          work until openwakeword's models are in the venv.
  )
)

REM 2b. Piper voice model ^(spoken replies; --voice needs it^) --------------------
if not exist "models\piper\en_US-lessac-medium.onnx" (
  echo [medo] downloading Piper voice ^(one-time^)...
  if not exist "models\piper" mkdir "models\piper"
  REM --ssl-no-revoke: Windows' schannel curl can't reach CRL/OCSP endpoints on
  REM many networks (CRYPT_E_NO_REVOCATION_CHECK) and would fail the download.
  curl -fL --ssl-no-revoke -o "models\piper\en_US-lessac-medium.onnx" "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
  curl -fL --ssl-no-revoke -o "models\piper\en_US-lessac-medium.onnx.json" "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
  if not exist "models\piper\en_US-lessac-medium.onnx" echo [medo] WARNING: Piper voice download failed - MEDO runs but won't speak until it's present.
)

REM 3. Vision venv ^(MediaPipe pins numpy^<2, so it lives apart^) -----------------
if not exist ".venv-vision\" (
  echo [medo] creating vision venv + installing deps ^(one-time^)...
  py -3.12 -m venv .venv-vision
  call .venv-vision\Scripts\pip install -q --only-binary=:all: -r requirements-vision.txt
)

REM 3b. Mark first-run setup complete - only once the 18 GB brain has actually
REM     landed, so an interrupted download resumes next launch instead of being
REM     skipped forever. Don't want qwen3:30b? Create .medo-setup-done yourself.
if defined MEDO_FIRSTRUN call :finish_setup

REM 4. Launch ---------------------------------------------------------------------
REM A previous MEDO still holding the ports makes the new one crash at startup
REM (bind error 10048) - and you end up talking to the OLD build. Replace it.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8710 :8730 :8731" ^| findstr LISTENING') do taskkill /pid %%p /f >nul 2>&1
echo [medo] starting vision sidecar ^(gestures, pointer mode, camera stream^)...
start "MEDO Vision" .venv-vision\Scripts\python -m vision.run
REM Open the HUD once the app has had a moment to bind the port.
start "MEDO HUD" /min cmd /c "timeout /t 6 >nul & explorer http://localhost:8730"
echo [medo] HUD: http://localhost:8730  -  say "hey medo". Close this window to quit.
call .venv\Scripts\python main.py --voice --hud --serve

endlocal
exit /b 0

REM ---------------------------------------------------------------------------
REM  Subroutines
REM ---------------------------------------------------------------------------

REM :pull_model <name> <human description> - pull an Ollama model, skip if present
:pull_model
ollama list 2>nul | findstr /i /c:"%~1" >nul
if %errorlevel%==0 (
  echo [medo]   %~1 already present - skipping.
  goto :eof
)
echo [medo]   pulling %~1 ^(%~2^)...
ollama pull %~1
goto :eof

REM :finish_setup - write the first-run marker only when the big model is present
:finish_setup
ollama list 2>nul | findstr /i /c:"qwen3:30b" >nul
if %errorlevel%==0 (
  echo done> ".medo-setup-done"
  echo [medo] first-run setup complete - later launches start straight up.
) else (
  echo [medo] setup incomplete: qwen3:30b still needed - it resumes on the next launch.
)
goto :eof
