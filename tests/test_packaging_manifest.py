"""Packaging consistency (S5) — a CI-ish check without a real build.

We can't run PyInstaller/Inno here, but we CAN enforce that the installer script,
the PyInstaller specs, and the manifest agree — so a rename never silently
desyncs the installer from the built exes — and that the known dependency traps
are actually documented + handled.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from winsetup import manifest

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "winsetup"


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_installer_creates_the_named_shortcuts():
    iss = _read("winsetup/installer.iss")
    assert "#define AppName" in iss and f'"{manifest.APP_NAME}"' in iss
    assert manifest.EXES["launcher"] in iss                    # MEDO.exe
    # Start-menu + desktop shortcut named via {#AppName}
    assert "{group}\\{#AppName}" in iss
    assert "{commondesktop}\\{#AppName}" in iss
    assert "{uninstallexe}" in iss                             # registers uninstaller


def test_installer_launches_the_wizard_and_asks_before_deleting_data():
    iss = _read("winsetup/installer.iss")
    assert "postinstall" in iss                               # runs after install
    assert "usPostUninstall" in iss and "MB_YESNO" in iss     # asks, doesn't assume
    assert "userappdata" in iss.lower()                       # removes user data on yes


def test_installer_lays_out_engine_and_vision_in_subfolders():
    iss = _read("winsetup/installer.iss")
    assert "{app}\\engine" in iss and "{app}\\vision" in iss


def test_specs_target_the_right_sources():
    assert "../main.py" in _read("winsetup/medo-engine.spec")
    assert "../vision/run.py" in _read("winsetup/medo-vision.spec")
    assert "../app/launcher.py" in _read("winsetup/medo-launcher.spec")


def test_vision_is_a_separate_bundle_not_merged_into_the_engine():
    engine = _read("winsetup/medo-engine.spec")
    vision = _read("winsetup/medo-vision.spec")
    # mediapipe belongs ONLY to the vision bundle; the engine excludes it.
    assert "mediapipe" in vision
    assert 'excludes=["mediapipe"' in engine or '"mediapipe"' in \
        engine.split("excludes")[1].split("]")[0]


def test_engine_bundles_hud_and_language_data():
    engine = _read("winsetup/medo-engine.spec")
    assert '("ui", "ui")' in engine and '("lang", "lang")' in engine


def test_docs_flag_the_mediapipe_numpy_fight_and_signing():
    doc = _read("docs/Packaging.md").lower()
    assert "numpy<2" in doc and "mediapipe" in doc
    assert "smartscreen" in doc and "sign" in doc
    assert "manual install-test checklist" in doc
    assert "orphan" in doc                                    # the no-orphans check


def test_build_script_uses_both_venvs():
    build = _read("winsetup/build.py")
    assert ".venv-vision" in build and ".venv" in build
    assert "installer.iss" in build


def test_launcher_frozen_paths_match_the_installer_layout():
    # engine_command()/vision_command() frozen branches must point at the
    # subfolders the installer creates.
    from app import launcher
    src = Path(launcher.__file__).read_text(encoding="utf-8")
    assert '"engine"' in src and '"vision"' in src


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
