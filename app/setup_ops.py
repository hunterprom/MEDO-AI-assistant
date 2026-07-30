"""Concrete first-run operations the wizard drives (install Ollama, pull models,
test the mic). Integration code — it touches the network, subprocesses, and audio
devices, so it isn't unit-tested; the wizard STATE MACHINE that calls it is
(app/wizard.py, with fakes). Each method is defensive and returns a bool /
plain value, never raises into the wizard.

VERIFY ON A BUILD MACHINE: the Ollama Windows installer URL + silent flag, and
the /api/pull streaming shape, can change — confirm against ollama.com before a
release (per the "don't invent, verify live" rule).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List

logger = logging.getLogger(__name__)

Progress = Callable[[float], None]

OLLAMA_HOST = "http://127.0.0.1:11434"
# Ollama ships a Windows installer built with Inno Setup (supports /VERYSILENT).
OLLAMA_WINDOWS_INSTALLER = "https://ollama.com/download/OllamaSetup.exe"


class SetupOps:
    def __init__(self, host: str = OLLAMA_HOST) -> None:
        self._host = host.rstrip("/")

    # -- ollama ---------------------------------------------------------------

    def ollama_installed(self) -> bool:
        if shutil.which("ollama"):
            return True
        return self._api_up()

    def _api_up(self) -> bool:
        try:
            import httpx
            return httpx.get(f"{self._host}/api/tags", timeout=2.0).status_code < 500
        except Exception:
            return False

    def install_ollama(self, progress: Progress) -> bool:
        try:
            import httpx
            dest = Path(tempfile.gettempdir()) / "OllamaSetup.exe"
            with httpx.stream("GET", OLLAMA_WINDOWS_INSTALLER, timeout=60.0,
                              follow_redirects=True) as r:
                if r.status_code != 200:
                    return False
                total = int(r.headers.get("content-length", 0)) or 0
                got = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_bytes(1 << 16):
                        f.write(chunk)
                        got += len(chunk)
                        if total:
                            progress(min(0.9, got / total * 0.9))
            # Inno Setup silent install; VERIFY the flag on a build machine.
            proc = subprocess.run([str(dest), "/VERYSILENT", "/NORESTART"],
                                  timeout=600)
            progress(1.0)
            return proc.returncode == 0
        except Exception:
            logger.warning("Ollama install failed", exc_info=True)
            return False

    # -- models ---------------------------------------------------------------

    def model_present(self, tag: str) -> bool:
        try:
            import httpx
            r = httpx.get(f"{self._host}/api/tags", timeout=5.0)
            names = {m.get("name", "") for m in r.json().get("models", [])}
            # ollama reports "llama3.2:3b"; a bare "llama3.2" implies ":latest"
            return tag in names or f"{tag}:latest" in names
        except Exception:
            return False

    def pull_model(self, tag: str, progress: Progress) -> bool:
        try:
            import httpx
            import json
            with httpx.stream("POST", f"{self._host}/api/pull",
                              json={"model": tag}, timeout=None) as r:
                if r.status_code != 200:
                    return False
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except ValueError:
                        continue
                    total, done = msg.get("total"), msg.get("completed")
                    if total:
                        progress(min(0.99, (done or 0) / total))
                    if msg.get("error"):
                        return False
                    if msg.get("status") == "success":
                        progress(1.0)
                        return True
            return self.model_present(tag)
        except Exception:
            logger.warning("pull %s failed", tag, exc_info=True)
            return False

    # -- microphone -----------------------------------------------------------

    def _input_devices(self) -> List[tuple]:
        """(real sounddevice id, name) for every INPUT device — the id, not the
        filtered position, is what sd.rec needs."""
        import sounddevice as sd
        return [(i, d["name"]) for i, d in enumerate(sd.query_devices())
                if d.get("max_input_channels", 0) > 0]

    def list_microphones(self) -> List[str]:
        try:
            return [name for _id, name in self._input_devices()]
        except Exception:
            return []

    def test_microphone(self, index: int) -> bool:
        try:
            import numpy as np
            import sounddevice as sd
            devices = self._input_devices()
            if not (0 <= index < len(devices)):
                return False
            device_id = devices[index][0]           # map list position -> real id
            rec = sd.rec(int(0.6 * 16000), samplerate=16000, channels=1,
                         dtype="float32", device=device_id)   # the SELECTED mic
            sd.wait()
            rms = float(np.sqrt(np.mean(np.square(rec))))
            return rms > 0.005          # heard *something* above the noise floor
        except Exception:
            logger.warning("mic test failed", exc_info=True)
            return False
