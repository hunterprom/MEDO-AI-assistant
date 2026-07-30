"""Plugin install-review + approval gate (S4).

The plugin loader hot-imports ``plugins/*.py`` — arbitrary local code, the biggest
in-process risk surface. This module gates it: a plugin's declared capabilities
are read STATICALLY (ast, WITHOUT executing it), and an un-approved plugin is
NOT imported at all — so untrusted code never runs until the user approves it,
having seen exactly what it wants ("this plugin wants: network, write_files —
allow?").

Honest limit (documented in docs/Security.md): once approved and imported, an
in-process plugin cannot be perfectly confined — Python can't stop `import httpx`
inside already-running code. The controls that DO hold: (1) nothing runs until
approved, (2) the plugin's SKILLS are policy-gated at runtime to only their
DECLARED capabilities (the engine denies an undeclared one), and (3) generated
code goes through quarantine + tests + explicit apply (core/self_dev.py). For
strong confinement of untrusted third-party code, a subprocess sandbox is future
work.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, List, Optional, Tuple

from security.capabilities import Capability

logger = logging.getLogger(__name__)

_VALUES = {c.value for c in Capability}


def file_digest(path: Path) -> str:
    """sha256 of the file's bytes — the identity the approval is bound to, so any
    EDIT to a plugin re-triggers review (approve-then-swap can't sneak through)."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


@dataclass(frozen=True)
class PluginReview:
    stem: str
    digest: str
    capabilities: FrozenSet[Capability]
    display_name: str = ""
    unknown_caps: Tuple[str, ...] = field(default_factory=tuple)
    declared: bool = True            # False when the file names no CAPABILITIES

    def summary(self) -> str:
        caps = ", ".join(sorted(c.value for c in self.capabilities)) or "none declared"
        extra = (f" (also names unknown capabilities: "
                 f"{', '.join(self.unknown_caps)})" if self.unknown_caps else "")
        return f"{self.display_name or self.stem} wants: {caps}{extra}"


def static_review(path: Path) -> PluginReview:
    """Read a plugin's declared capabilities WITHOUT executing it (ast only).

    A plugin declares, at module level:  ``CAPABILITIES = ["network", ...]`` and
    optionally ``PLUGIN_NAME = "..."``. A file that declares nothing is reviewed
    as ``declared=False`` (unknown surface → the user is told it's unvetted)."""
    path = Path(path)
    digest = file_digest(path)
    caps: set[Capability] = set()
    unknown: list[str] = []
    name = ""
    declared = False
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return PluginReview(path.stem, digest, frozenset(), declared=False)

    for node in tree.body:                       # module level only
        if not isinstance(node, ast.Assign):
            continue
        targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if "CAPABILITIES" in targets and isinstance(node.value, (ast.List, ast.Tuple)):
            declared = True
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    if elt.value in _VALUES:
                        caps.add(Capability(elt.value))
                    else:
                        unknown.append(elt.value)
        elif "PLUGIN_NAME" in targets and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                name = node.value.value
    return PluginReview(path.stem, digest, frozenset(caps), display_name=name,
                        unknown_caps=tuple(unknown), declared=declared)


def _default_store_path() -> Path:
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"),
                                                     ".config")
    return Path(base) / "MEDO" / "plugin_approvals.json"


class PluginApprovalStore:
    """Records which plugin (by stem) is approved at which content digest."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path is not None else _default_store_path()

    def _load(self) -> dict:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def is_approved(self, stem: str, digest: str) -> bool:
        entry = self._load().get(stem)
        return bool(entry) and entry.get("digest") == digest and bool(digest)

    def approve(self, review: PluginReview) -> None:
        data = self._load()
        data[review.stem] = {"digest": review.digest,
                             "capabilities": sorted(c.value
                                                    for c in review.capabilities)}
        self._save(data)

    def revoke(self, stem: str) -> None:
        data = self._load()
        if data.pop(stem, None) is not None:
            self._save(data)


# -- CLI: review + approve plugins --------------------------------------------

def _main(argv: List[str]) -> int:  # pragma: no cover - thin CLI wrapper
    from core.plugins import PLUGINS_DIR

    store = PluginApprovalStore()
    cmd = argv[0] if argv else "list"
    reviews = [static_review(p) for p in sorted(PLUGINS_DIR.glob("*.py"))
               if not p.name.startswith("_")]
    if cmd == "list":
        for r in reviews:
            ok = store.is_approved(r.stem, r.digest)
            print(f"[{'approved' if ok else 'HELD'}] {r.summary()}")
        return 0
    if cmd == "approve" and len(argv) > 1:
        target = argv[1]
        for r in reviews:
            if r.stem == target:
                store.approve(r)
                print(f"approved {r.stem} — {r.summary()}")
                return 0
        print(f"no plugin named {target!r} in {PLUGINS_DIR}")
        return 1
    print("usage: python -m security.plugins [list | approve <name>]")
    return 1


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(_main(sys.argv[1:]))
