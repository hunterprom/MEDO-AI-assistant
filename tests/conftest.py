"""Shared test fixtures.

Tests that build a Router via ``load_settings()`` would otherwise talk to the
developer's real ``jarvis.db`` — and, since routing metrics persist (M3 of the
evaluator pass), every "what time is it" test run would pollute the README's
Performance numbers. Point the whole suite at a throwaway database instead.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_assistant_db(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDO_MEMORY__DB_PATH", str(tmp_path / "test-medo.db"))
