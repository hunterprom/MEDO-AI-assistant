@echo off
REM ============================================================================
REM  Jarvis v2 - one-click launcher (Windows). Double-click run.bat.
REM  Starts Ollama, the vision sidecar, and the app with voice + HUD + API.
REM  First run sets up both virtualenvs and installs all dependencies.
REM ============================================================================
cd /d "%~dp0"
setlocal

REM 1. Ollama server + a model ------------------------------------------------
where ollama >nul 2>&1
if %errorlevel%==0 (
  tasklist /fi "imagename eq ollama.exe" | find /i "ollama.exe" >nul || (
    echo [jarvis] starting Ollama...
    start "" /b ollama serve
    timeout /t 3 >nul
  )
  ollama list | findstr /r "." >nul || (
    echo [jarvis] pulling llama3.2:3b ^(one-time^)...
    ollama pull llama3.2:3b
  )
) else (
  echo [jarvis] Ollama not installed - LLM path offline; fast path still works.
)

REM 2. Main venv --------------------------------------------------------------
if not exist ".venv\" (
  echo [jarvis] creating main venv + installing deps ^(one-time^)...
  py -3.12 -m venv .venv
  call .venv\Scripts\pip install -q -r requirements.txt
  call .venv\Scripts\python -c "import openwakeword.utils as u; u.download_models()"
)

REM 2b. Piper voice model ^(spoken replies; --voice needs it^) ----------------
if not exist "models\piper\en_US-lessac-medium.onnx" (
  echo [jarvis] downloading Piper voice ^(one-time^)...
  if not exist "models\piper" mkdir "models\piper"
  REM --ssl-no-revoke: Windows' schannel curl can't reach CRL/OCSP behind many
  REM firewalls (CRYPT_E_NO_REVOCATION_CHECK) and would otherwise fail the download.
  curl -fL --ssl-no-revoke -o "models\piper\en_US-lessac-medium.onnx" "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
  curl -fL --ssl-no-revoke -o "models\piper\en_US-lessac-medium.onnx.json" "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
  if not exist "models\piper\en_US-lessac-medium.onnx" echo [jarvis] WARNING: Piper voice download failed - MEDO will run but won't speak until it's present.
)

REM 3. Vision venv ^(MediaPipe pins numpy^<2, so it lives apart^) --------------
if not exist ".venv-vision\" (
  echo [jarvis] creating vision venv + installing deps ^(one-time^)...
  py -3.12 -m venv .venv-vision
  call .venv-vision\Scripts\pip install -q --only-binary=:all: -r requirements-vision.txt
)

REM 4. Launch -----------------------------------------------------------------
echo [jarvis] starting vision sidecar...
start "Jarvis Vision" .venv-vision\Scripts\python -m vision.run
REM Open the HUD once the app has had a moment to bind the port.
start "MEDO HUD" /min cmd /c "timeout /t 6 >nul & explorer http://localhost:8730"
echo [jarvis] HUD at http://localhost:8730 - say "Hey Jarvis". Close window to quit.
call .venv\Scripts\python main.py --voice --hud --serve

endlocal
