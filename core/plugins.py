"""Plugin SDK: drop-in skills from the ``plugins/`` folder.

Any ``plugins/*.py`` file is imported at startup and its skills are registered
alongside the built-ins — they match on the fast path AND become LLM tools,
with zero changes to core code. Two ways to expose skills:

1. Define :class:`~skills.base.Skill` subclasses with no-arg constructors —
   they're auto-instantiated.
2. Define ``def setup(services) -> list[Skill]`` for skills that need app
   services. ``services`` is a dict with ``settings``, ``announcer``,
   ``summarize``, ``reminders``, and ``doc_index`` (any may be None).

A broken plugin is logged and skipped — it can never take the assistant down.
Files starting with ``_`` are ignored. See ``plugins/README.md`` for the
template and ``plugins/example_dice.py`` for a working sample.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from core.config import PROJECT_ROOT
from skills.base import Skill, SkillRegistry

logger = logging.getLogger(__name__)

PLUGINS_DIR = PROJECT_ROOT / "plugins"


def load_plugins(
    registry: SkillRegistry,
    services: dict,
    plugins_dir: Path = PLUGINS_DIR,
) -> list[str]:
    """Import every plugin and register its skills. Returns 'file:skill' names."""
    loaded: list[str] = []
    if not plugins_dir.exists():
        return loaded
    for path in sorted(plugins_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            module_name = f"medo_plugin_{path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            skills: list[Skill] = []
            if hasattr(module, "setup"):
                skills = list(module.setup(services) or [])
            else:
                for obj in vars(module).values():
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, Skill)
                        and obj is not Skill
                        and obj.__module__ == module_name
                    ):
                        skills.append(obj())

            for skill in skills:
                registry.register(skill)
                loaded.append(f"{path.stem}:{skill.name}")
        except Exception:  # one bad plugin must never take the assistant down
            logger.exception("plugin %s failed to load — skipped", path.name)
    if loaded:
        logger.info("plugins loaded: %s", ", ".join(loaded))
    return loaded
