"""Backward compat: a pre-merge (v2) config.yaml must still validate, and the
shipped config.yaml must load with the new fields."""

from __future__ import annotations

import dataclasses

from core.config import load_settings

OLD_V2_CONFIG = """
llm:
  provider: "ollama"
  host: "http://localhost:11434"
  api_key: ""
  default_model: null
  fallback_model: "llama3.2:3b"
  temperature: 0.6
  num_ctx: 4096
  request_timeout_s: 120

stt:
  engine: "faster-whisper"
  model: "small"
  language: "en"
  initial_prompt: >-
    Commands for a voice assistant: what time is it, open chrome.

vision:
  enabled: true
  camera_index: 0
  stream_port: 8731
  gestures:
    thumbs_up: "yes"

memory:
  db_path: "jarvis.db"
  max_turns: 10
"""


def test_old_v2_config_still_validates(tmp_path):
    path = tmp_path / "old.yaml"
    path.write_text(OLD_V2_CONFIG, encoding="utf-8")
    s = load_settings(path)
    # old values preserved
    assert s.stt.language == "en"
    assert s.llm.default_model is None
    assert s.vision.gestures["thumbs_up"] == "yes"
    # new fields fall back to safe defaults
    assert s.llm.keep_alive == "30m"
    assert s.vision.pointer.enabled is True
    assert s.vision.pointer.sensitivity == 2.5
    assert s.vision_llm.model == "moondream"
    assert s.memory.max_facts == 20


def test_shipped_config_loads_with_new_fields():
    s = load_settings()  # the repo's config.yaml
    assert s.llm.default_model == "qwen3:30b"
    assert s.stt.language is None            # whisper auto-detect (mk + en)
    assert s.stt.initial_prompt in (None, "")
    assert s.vision.pointer.click_debounce_ms == 600
    assert s.memory.max_facts == 20


def test_sidecar_pointer_config_mirrors_the_main_one():
    """The vision sidecar has its own dataclass, built field-by-field.

    It runs in a separate venv and cannot import core.config, so every field
    added to PointerConfig must be mirrored in PointerRunConfig AND in the
    loader that fills it. Forgetting either kills the sidecar at startup with
    an AttributeError the moment engine.py reads the missing knob — which is
    exactly what happened when the gesture-strictness knobs were added.
    """
    import inspect

    from core.config import PointerConfig
    from vision.run import PointerRunConfig, load_config

    expected = set(PointerConfig.model_fields)
    actual = {f.name for f in dataclasses.fields(PointerRunConfig)}
    missing = expected - actual
    assert not missing, f"PointerRunConfig is missing: {sorted(missing)}"

    # …and the loader must actually pass them, not just declare them.
    source = inspect.getsource(load_config)
    unset = [name for name in expected if f"{name}=" not in source]
    assert not unset, f"load_config never sets: {sorted(unset)}"
