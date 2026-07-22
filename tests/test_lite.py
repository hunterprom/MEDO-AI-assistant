"""Lite launcher: env-override profile + smallest-model selection (offline)."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _no_medo_env_leak():
    """medo-lite sets MEDO_* via os.environ.setdefault (not monkeypatch), so
    snapshot and restore the environment around every test — otherwise a lite
    override (e.g. MEDO_STT__MODEL=tiny) would bleed into the rest of the suite."""
    before = {k: v for k, v in os.environ.items() if k.startswith("MEDO_")}
    yield
    for k in [k for k in os.environ if k.startswith("MEDO_")]:
        if k not in before:
            del os.environ[k]
    os.environ.update(before)


def _load_lite():
    spec = importlib.util.spec_from_file_location("medolite", _ROOT / "medo-lite.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smallest_model_prefers_fewest_params(monkeypatch):
    ml = _load_lite()

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            import json
            return json.dumps({"models": [
                {"name": "qwen3:30b", "details": {"parameter_size": "30B"}},
                {"name": "llama3.2:1b", "details": {"parameter_size": "1B"}},
                {"name": "llama3.2:3b", "details": {"parameter_size": "3B"}},
            ]}).encode()

    monkeypatch.setattr(ml.urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert ml._smallest_installed_ollama_model("http://x", cap_b=4.0) == "llama3.2:1b"


def test_smallest_model_none_when_ollama_down(monkeypatch):
    ml = _load_lite()

    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(ml.urllib.request, "urlopen", boom)
    assert ml._smallest_installed_ollama_model("http://x") is None


def test_lite_applies_overrides(monkeypatch):
    ml = _load_lite()
    for k in list(__import__("os").environ):
        if k.startswith("MEDO_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(ml, "_smallest_installed_ollama_model", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["medo-lite.py", "--once", "hi"])
    fake_main = types.ModuleType("main")
    captured = {}
    fake_main.main = lambda: captured.setdefault("ran", True)
    monkeypatch.setitem(sys.modules, "main", fake_main)

    ml.main()

    import os
    assert captured.get("ran") is True
    assert os.environ["MEDO_STT__MODEL"] == "tiny"
    assert os.environ["MEDO_STT__COMPUTE_TYPE"] == "int8"
    assert os.environ["MEDO_STT__BEAM_SIZE"] == "1"
    assert os.environ["MEDO_VISION__ENABLED"] == "false"
    assert os.environ["MEDO_MEMORY__EMBED_MODEL"] == ""


def test_with_vision_flag_reenables_vision(monkeypatch):
    ml = _load_lite()
    for k in list(__import__("os").environ):
        if k.startswith("MEDO_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(ml, "_smallest_installed_ollama_model", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["medo-lite.py", "--with-vision", "--once", "hi"])
    fake_main = types.ModuleType("main")
    fake_main.main = lambda: None
    monkeypatch.setitem(sys.modules, "main", fake_main)

    ml.main()

    import os
    assert os.environ["MEDO_VISION__ENABLED"] == "true"
    # --with-vision is consumed, not passed through to main's argv
    assert "--with-vision" not in sys.argv
