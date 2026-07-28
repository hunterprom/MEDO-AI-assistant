# MEDO plugins

Drop a `.py` file in this folder and restart MEDO — its skills are discovered
automatically and work **both** as spoken/typed commands (fast path) and as
tools the LLM can call. No core changes needed.

## Minimal plugin

```python
import re
from skills.base import Skill, SkillRequest, SkillResult

class HelloSkill(Skill):
    name = "hello_plugin"
    description = "Says hello. Use when the user asks the plugin to greet."

    patterns = [re.compile(r"\bsay hello plugin\b", re.IGNORECASE)]

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("Hello from a plugin!")
```

That's it: `patterns` make it a voice command, `description` (plus an optional
`tool_schema()` for arguments) exposes it to the LLM.

## Skills that need app services

Define `setup(services)` instead — it receives a dict with `settings`,
`announcer`, `summarize`, `reminders`, `doc_index` (any may be None) and
returns the skill instances:

```python
def setup(services):
    return [MySkill(services["settings"])]
```

## Rules

- Files starting with `_` are ignored.
- A plugin that raises at import/registration is logged and **skipped** —
  it can't crash MEDO.
- Keep replies short: they're spoken aloud.
- See `example_dice.py` for a complete working sample with tool arguments.
