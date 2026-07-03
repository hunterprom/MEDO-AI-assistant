"""Backward compat: a pre-merge (v2) config.yaml must still validate, and the
shipped config.yaml must load with the new fields."""

from __future__ import annotations

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
