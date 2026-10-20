"""P1-2: the setup marker knows WHICH spec was installed, not just that one was.

The old marker was the word "done". Adding a package to requirements.txt
changed nothing on a machine that already had a venv — the failure surfaced
later, as an ImportError that reads like a code bug. These tests pin the
behaviour that replaces it, including the upgrade path for the millions (two)
of existing installs that still have a legacy marker on disk.
"""

from __future__ import annotations

import json

import pytest

from scripts import setup_marker as sm


@pytest.fixture
def spec(tmp_path):
    """A throwaway project root with requirements files and a marker path."""
    (tmp_path / "requirements.txt").write_text("aiohttp\nnumpy\n", encoding="utf-8")
    (tmp_path / "requirements-vision.txt").write_text("mediapipe\n", encoding="utf-8")
    (tmp_path / "requirements-studio.txt").write_text("cadquery\n", encoding="utf-8")
    return tmp_path, tmp_path / ".medo-setup-done"


# --- the digest ---------------------------------------------------------------


def test_the_same_requirements_hash_the_same(spec):
    root, _ = spec
    assert sm.spec_digest("main-deps", root) == sm.spec_digest("main-deps", root)


def test_changing_requirements_changes_the_hash(spec):
    root, _ = spec
    before = sm.spec_digest("main-deps", root)
    (root / "requirements.txt").write_text("aiohttp\nnumpy\nhttpx\n", encoding="utf-8")
    assert sm.spec_digest("main-deps", root) != before


def test_components_do_not_share_a_hash(spec):
    root, _ = spec
    assert sm.spec_digest("main-deps", root) != sm.spec_digest("vision-deps", root)


def test_line_endings_do_not_change_the_hash(spec):
    """A Windows checkout with autocrlf must read as the same spec as macOS."""
    root, _ = spec
    unix = sm.spec_digest("main-deps", root)
    (root / "requirements.txt").write_text("aiohttp\r\nnumpy\r\n", encoding="utf-8",
                                           newline="")
    assert sm.spec_digest("main-deps", root) == unix


def test_a_missing_requirements_file_is_stable_not_fatal(spec):
    root, _ = spec
    (root / "requirements-studio.txt").unlink()
    assert sm.spec_digest("studio-deps", root) == sm.spec_digest("studio-deps", root)


# --- staleness ----------------------------------------------------------------


def test_a_fresh_install_is_stale_everywhere(spec):
    root, marker = spec
    assert set(sm.stale_components(marker, root)) == set(sm.COMPONENTS)


def test_nothing_is_stale_right_after_a_write(spec):
    root, marker = spec
    sm.write_marker(path=marker, root=root)
    assert sm.stale_components(marker, root) == []


def test_editing_one_requirements_file_stales_only_that_component(spec):
    root, marker = spec
    sm.write_marker(path=marker, root=root)
    (root / "requirements-vision.txt").write_text("mediapipe==1.0\n", encoding="utf-8")
    assert sm.stale_components(marker, root) == ["vision-deps"]


def test_writing_one_component_does_not_erase_the_others(spec):
    root, marker = spec
    sm.write_marker(path=marker, root=root)
    (root / "requirements.txt").write_text("aiohttp\n", encoding="utf-8")
    sm.write_marker(["main-deps"], path=marker, root=root)
    assert sm.stale_components(marker, root) == []


def test_bumping_the_setup_revision_stales_everything(spec, monkeypatch):
    """The escape hatch for step changes a requirements file can't express."""
    root, marker = spec
    sm.write_marker(path=marker, root=root)
    assert sm.stale_components(marker, root) == []
    monkeypatch.setattr(sm, "SETUP_REVISION", sm.SETUP_REVISION + 1)
    assert set(sm.stale_components(marker, root)) == set(sm.COMPONENTS)


# --- provisioning is a separate question --------------------------------------


def test_provisioning_is_not_recorded_by_a_plain_write(spec):
    root, marker = spec
    sm.write_marker(path=marker, root=root)
    assert sm.is_provisioned(marker) is False


def test_provisioning_is_recorded_when_asked(spec):
    root, marker = spec
    sm.write_marker(path=marker, root=root, provisioned=True)
    assert sm.is_provisioned(marker) is True


def test_a_later_dependency_write_keeps_provisioning(spec):
    """Reinstalling a package must never re-trigger the 24 GB download."""
    root, marker = spec
    sm.write_marker(path=marker, root=root, provisioned=True)
    sm.write_marker(["main-deps"], path=marker, root=root)
    assert sm.is_provisioned(marker) is True


# --- the upgrade path ---------------------------------------------------------


def test_a_legacy_marker_counts_as_provisioned(spec):
    """Existing installs must not be shown the first-run banner again."""
    root, marker = spec
    marker.write_text("done\n", encoding="utf-8")
    assert sm.is_provisioned(marker) is True


def test_a_legacy_marker_leaves_dependencies_stale(spec):
    """What it installed is unknown, so re-check once (a fast no-op pip run)."""
    root, marker = spec
    marker.write_text("done\n", encoding="utf-8")
    assert set(sm.stale_components(marker, root)) == set(sm.COMPONENTS)


def test_upgrading_from_a_legacy_marker_settles(spec):
    root, marker = spec
    marker.write_text("done\n", encoding="utf-8")
    sm.write_marker(path=marker, root=root)
    assert sm.stale_components(marker, root) == []
    assert sm.is_provisioned(marker) is True
    assert json.loads(marker.read_text(encoding="utf-8"))["version"] == sm.MARKER_VERSION


def test_a_corrupt_marker_does_not_crash_the_launcher(spec):
    root, marker = spec
    marker.write_text("{ truncated", encoding="utf-8")
    assert set(sm.stale_components(marker, root)) == set(sm.COMPONENTS)
    assert sm.is_provisioned(marker) is True      # don't re-download 24 GB


def test_a_marker_holding_a_json_scalar_is_handled(spec):
    root, marker = spec
    marker.write_text("42", encoding="utf-8")
    assert set(sm.stale_components(marker, root)) == set(sm.COMPONENTS)


def test_an_empty_marker_reads_as_a_first_run(spec):
    root, marker = spec
    marker.write_text("", encoding="utf-8")
    assert sm.is_provisioned(marker) is False


# --- the report ---------------------------------------------------------------


def test_explain_says_what_will_happen(spec):
    root, marker = spec
    sm.write_marker(path=marker, root=root, provisioned=True)
    (root / "requirements.txt").write_text("aiohttp\nhttpx\n", encoding="utf-8")
    text = sm.explain(marker, root)
    assert "main-deps" in text and "STALE" in text
    assert "vision-deps" in text and "current" in text


def test_explain_is_ascii_for_a_cp1252_console(spec):
    """It prints during setup on a Windows console that cannot take an em dash."""
    root, marker = spec
    sm.write_marker(path=marker, root=root)
    text = sm.explain(marker, root)
    text.encode("ascii")            # raises if a non-ASCII character crept in


def test_explain_handles_a_first_run(spec):
    root, marker = spec
    assert "first run" in sm.explain(marker, root)
