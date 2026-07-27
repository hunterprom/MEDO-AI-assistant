"""MEDO building standalone apps — the coding agent, pointed at a fresh folder.

Where self-dev (core/self_dev.py) edits MEDO's OWN code under review, this is the
outward version: "make an app that …" scaffolds a NEW, self-contained project in
its own directory under ``apps_dir``, using the same installed coding agent. The
blast radius is that one folder — nothing here touches MEDO's code or the repo.

Off by default (:class:`~core.config.AppBuilderConfig` ``enabled=False``); the
agent executes on this machine, so turning it on is opting into that.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from core.config import AppBuilderConfig

logger = logging.getLogger(__name__)

_PROMPT = (
    "Build a working app in the CURRENT directory (it's empty and yours). Create "
    "every file it needs — code, and a short README.md that says what it is and "
    "how to run it. Keep it self-contained and prefer the standard library / no "
    "heavy dependencies unless the task truly needs them. Do NOT run shell "
    "commands, git, or a package manager — just write the files.\n\n"
    "THE APP:\n{spec}\n"
)
#: Noise we never count as a produced file.
_IGNORE = {".git", "__pycache__", "node_modules", ".venv"}


class AppBuildError(Exception):
    """An app build that failed in a way worth reporting verbatim."""


def _slug(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:limit].rstrip("-") or "app"


@dataclass
class AppBuild:
    name: str
    spec: str
    path: Path
    files: list[str] = field(default_factory=list)
    agent_output: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.files) and not self.error


class AppBuilder:
    """Scaffold a standalone app in its own folder with the coding agent.

    ``agent`` is injectable (``async (app_dir: Path, spec: str) -> str``) so the
    whole flow is testable without the real CLI: the default shells out to it.
    """

    def __init__(self, config: AppBuilderConfig, *, agent=None) -> None:
        self._config = config
        self._root = Path(config.apps_dir).expanduser()
        self._agent = agent or self._run_cli_agent

    async def build(self, name: str, spec: str) -> AppBuild:
        spec = (spec or "").strip()
        if not spec:
            raise AppBuildError("an app build needs a description of what to make")
        if not self._config.enabled:
            raise AppBuildError(
                "app building is disabled — set app_builder.enabled to allow it")

        name = (name or "").strip() or spec
        folder = _slug(name)
        app_dir = self._root / folder
        if app_dir.exists() and any(app_dir.iterdir()):
            app_dir = self._root / f"{folder}-{uuid.uuid4().hex[:6]}"
        app_dir.mkdir(parents=True, exist_ok=True)

        try:
            agent_output = await self._agent(app_dir, spec)
        except AppBuildError:
            raise
        except Exception as exc:
            raise AppBuildError(f"the coding agent failed: {exc}") from exc

        files = self._list_files(app_dir)
        build = AppBuild(name=folder, spec=spec, path=app_dir, files=files,
                         agent_output=agent_output,
                         error="" if files else "the agent produced no files")
        logger.info("app_builder: %s -> %s (%d files)",
                    "built" if build.ok else "empty", app_dir, len(files))
        return build

    # -- internals ----------------------------------------------------------

    def _list_files(self, app_dir: Path) -> list[str]:
        out = []
        for p in sorted(app_dir.rglob("*")):
            if p.is_file() and not any(part in _IGNORE for part in p.parts):
                out.append(str(p.relative_to(app_dir)).replace("\\", "/"))
        return out

    async def _run_cli_agent(self, app_dir: Path, spec: str) -> str:
        prompt = _PROMPT.format(spec=spec)
        engine = self._config.engine
        if engine == "claude-code":
            argv = ["claude", "-p", prompt,
                    "--permission-mode", self._config.permission_mode,
                    "--output-format", "text"]
        elif engine == "codex":
            argv = ["codex", "exec", "--skip-git-repo-check", prompt]
        else:
            raise AppBuildError(f"unknown app-builder engine {engine!r}")
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(app_dir),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(), timeout=self._config.agent_timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            raise AppBuildError(
                f"the coding agent timed out after "
                f"{self._config.agent_timeout_s:.0f}s") from None
        if proc.returncode != 0:
            detail = (err or out).decode(errors="replace")[:300]
            raise AppBuildError(f"the coding agent exited {proc.returncode}: {detail}")
        return out.decode(errors="replace").strip()


def _amain() -> None:  # pragma: no cover - manual, real-agent trial
    from core.config import load_settings

    spec = " ".join(sys.argv[1:]).strip()
    if not spec:
        print('usage: python -m core.app_builder "an app that ..."')
        raise SystemExit(2)
    settings = load_settings()
    settings.app_builder.enabled = True
    builder = AppBuilder(settings.app_builder)

    async def _go() -> None:
        build = await builder.build("", spec)
        print("path :", build.path)
        print("files:", ", ".join(build.files) or "(none)")
        print("ok   :", build.ok)
        print("\n--- agent ---\n" + build.agent_output[:2000])

    asyncio.run(_go())


if __name__ == "__main__":  # pragma: no cover
    _amain()
