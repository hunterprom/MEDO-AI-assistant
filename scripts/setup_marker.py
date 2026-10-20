"""Does this install still match what MEDO needs? — the versioned setup marker.

``.medo-setup-done`` used to be a file containing the word "done". It answered
one question ("has setup ever run?") and the launchers used it to answer a
different one ("are the dependencies current?"), which it cannot. The visible
consequence: adding a package to ``requirements.txt`` changed nothing on anyone
else's machine, because the venv already existed and the marker already said
done. The invisible consequence is worse — MEDO would then fail at import time,
in a way that looks like a code bug rather than a stale environment.

So the marker now records a **hash of the setup spec**, per component:

    {"version": 2, "setup_revision": 1, "provisioned": true,
     "components": {"main-deps": "<sha256>", "vision-deps": "<sha256>"}}

A component is stale when its requirements file has changed since the hash was
written, or when :data:`SETUP_REVISION` has been bumped because the *steps*
changed (a new model to pull, a new system package). The launchers ask this
script what is stale and reinstall only that.

Two things are deliberately kept apart, because conflating them is what broke:

* **provisioning** — the one-time 24 GB of system packages and models. Gated on
  "has this ever completed", which is genuinely a yes/no.
* **dependencies** — gated on the spec hash, which changes over a project's life
  and must be allowed to.

Stdlib only, and it runs under the SYSTEM python: it is consulted before the
venv it might be about to rebuild exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKER_PATH = PROJECT_ROOT / ".medo-setup-done"

#: Bump when the setup STEPS change in a way that requirements files don't
#: capture — a new Ollama model to pull, a new system package, a voice file that
#: has to be downloaded. Every install then re-runs setup once. Do NOT bump for
#: a package version change: the requirements hash already covers that.
SETUP_REVISION = 1

#: component -> the requirements file that defines it.
COMPONENTS = {
    "main-deps": "requirements.txt",
    "vision-deps": "requirements-vision.txt",
    "studio-deps": "requirements-studio.txt",
}

MARKER_VERSION = 2


def spec_digest(component: str, root: Path = PROJECT_ROOT) -> str:
    """SHA-256 over the setup revision and one component's requirements file.

    A missing requirements file hashes to a stable "absent" value rather than
    raising, so an install that doesn't ship the optional studio requirements
    isn't permanently stale.
    """
    path = root / COMPONENTS[component]
    h = hashlib.sha256()
    h.update(str(SETUP_REVISION).encode())
    h.update(b"\x00")
    h.update(component.encode())
    h.update(b"\x00")
    try:
        # Normalise line endings: a git checkout with autocrlf must not read as
        # a different spec from the same file on macOS.
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    except OSError:
        text = "\x00absent"
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def read_marker(path: Path = MARKER_PATH) -> dict:
    """The marker as a dict. Legacy and corrupt markers degrade, never raise.

    A legacy marker (the literal word "done") means provisioning finished on an
    older MEDO: keep that — nobody should be shown the 24 GB first-run banner
    again — but report every dependency component as stale, because what was
    installed then is unknown. The reinstall that follows is a no-op pip run if
    the environment is in fact current.
    """
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return {}
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"version": 1, "provisioned": True, "components": {}}
    if not isinstance(data, dict):
        return {"version": 1, "provisioned": True, "components": {}}
    return data


def is_provisioned(path: Path = MARKER_PATH) -> bool:
    """Has the one-time system provisioning ever completed?"""
    return bool(read_marker(path).get("provisioned"))


def stale_components(path: Path = MARKER_PATH, root: Path = PROJECT_ROOT) -> list[str]:
    """Components whose installed spec no longer matches the files on disk."""
    recorded = read_marker(path).get("components") or {}
    if not isinstance(recorded, dict):
        recorded = {}
    return [name for name in COMPONENTS
            if recorded.get(name) != spec_digest(name, root)]


def write_marker(components: list[str] | None = None, *, provisioned: bool | None = None,
                 path: Path = MARKER_PATH, root: Path = PROJECT_ROOT) -> dict:
    """Record that ``components`` are now installed at the current spec.

    Merges rather than replaces: a run that reinstalled only the vision deps
    must not erase the record of the main ones.
    """
    marker = read_marker(path)
    recorded = dict(marker.get("components") or {})
    for name in (components if components is not None else list(COMPONENTS)):
        if name in COMPONENTS:
            recorded[name] = spec_digest(name, root)
    payload = {
        "version": MARKER_VERSION,
        "setup_revision": SETUP_REVISION,
        "provisioned": (bool(marker.get("provisioned")) if provisioned is None
                        else bool(provisioned)),
        "components": recorded,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def explain(path: Path = MARKER_PATH, root: Path = PROJECT_ROOT) -> str:
    """Human-readable account of what setup would do and why."""
    marker = read_marker(path)
    if not marker:
        return "No setup marker — this is a first run; full setup will run."
    lines = []
    if marker.get("version", 1) < MARKER_VERSION:
        lines.append("The marker predates versioned setup: dependencies will be "
                     "re-checked once, then recorded.")
    lines.append(f"Provisioned (system packages + models): "
                 f"{'yes' if marker.get('provisioned') else 'no'}")
    stale = stale_components(path, root)
    for name, req in COMPONENTS.items():
        # ASCII only: this prints to a cp1252 Windows console during setup,
        # where a stray em dash is either mojibake or a UnicodeEncodeError.
        state = "STALE - will reinstall" if name in stale else "current"
        lines.append(f"  {name:<12} ({req}): {state}")
    if not stale:
        lines.append("Everything matches; setup will be skipped.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MEDO setup marker — is this install current?")
    parser.add_argument("--stale", action="store_true",
                        help="print stale component names, space-separated")
    parser.add_argument("--provisioned", action="store_true",
                        help="exit 0 if one-time provisioning has completed")
    parser.add_argument("--write", nargs="*", metavar="COMPONENT", default=None,
                        help="record components as installed (default: all)")
    parser.add_argument("--mark-provisioned", action="store_true",
                        help="with --write: also record provisioning as done")
    parser.add_argument("--explain", action="store_true",
                        help="explain what setup would do, and why")
    args = parser.parse_args(argv)

    if args.explain:
        print(explain())
        return 0
    if args.provisioned:
        return 0 if is_provisioned() else 1
    if args.write is not None:
        payload = write_marker(args.write or None,
                               provisioned=True if args.mark_provisioned else None)
        print(f"recorded {len(payload['components'])} component(s) "
              f"at setup revision {SETUP_REVISION}")
        return 0
    if args.stale:
        print(" ".join(stale_components()))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
