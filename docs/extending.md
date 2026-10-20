# Extending MEDO

Four ways to add capability, in increasing order of how much you have to write:

| You want to | Do this | Python? |
|---|---|---|
| Chain existing skills | a routine (`routines/*.yaml`) | no |
| Add a new expert to the council | `council.extra` in `config.yaml` | no |
| Connect another app's tools | `mcp.servers` in `config.yaml` | no |
| Add a genuinely new capability | a skill, or a plugin | yes |

Start at the top. Most "MEDO should be able to…" requests are a routine.

---

## A routine — chaining skills, no code

Drop a `.yaml` into `routines/`. It is registered as an ordinary skill on the
next start, so it routes on the fast path, appears as an LLM tool, and passes
the same policy gate as everything else.

```yaml
# routines/evening.yaml
name: evening wrap
description: What's left today, and tomorrow's weather.
trigger:
  phrase: ["evening wrap", "вечерен преглед"]   # matched literally, not as regex
  at: "18:00"
  days: [mon, tue, wed, thu, fri]
params:
  city: Skopje
steps:
  - say: "Here's where things stand."
  - skill: weather
    args: {city: "{city}"}
    into: forecast
  - say: "Tomorrow: {forecast}"
```

Check it without starting MEDO:

```bash
python -m core.composer --check
```

A routine **may not call a skill that needs confirmation** — MEDO refuses to
load it and says which step. See `routines/README.md` for why.

---

## A specialist — `council.extra`, no code

A specialist is a system prompt, not a process, so adding one is a config edit.
It is reachable by name immediately, and by *topic* as soon as the council index
rebuilds — the role description is embedded, so no keyword list is required.

```yaml
council:
  extra:
    veterinarian:
      title: "the vet"
      prompt: >-
        You are a small-animal veterinarian. You reason about symptoms, dosages
        by weight, and when something needs a clinic today rather than tomorrow.
        You always say this is general information, not a diagnosis.
      triggers: ["dog", "cat", "puppy", "куче", "мачка"]   # optional
      wants_tools: false
```

`triggers` is optional and only breaks near-ties now that selection is semantic
(`core/council_index.py`). `title` is what MEDO says out loud — "I asked the
vet".

---

## An MCP server — borrowing another app's tools

Any application that speaks the Model Context Protocol becomes callable by the
LLM path. Its tools are registered as ordinary skills.

```yaml
mcp:
  enabled: true
  servers:
    filesystem:
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "C:/Users/me/Notes"]
    home-assistant:
      url: "http://homeassistant.local:8123/mcp"
      headers: {Authorization: "Bearer ..."}
```

A server that fails to start is logged and skipped; it never blocks startup.

### The other direction — MEDO as an MCP server

Other local AI tools can call MEDO's skills:

```json
{"mcpServers": {"medo": {
  "command": "C:/Users/me/Desktop/MEDO-main/.venv/Scripts/python.exe",
  "args": ["-m", "mcp_servers.skills_server"]
}}}
```

Read-only skills only, by default. See what would be offered:

```bash
python -m mcp_servers.skills_server --list
```

---

## A skill — the real extension point

A skill is written **once** and exposed twice: regex `patterns` for the
deterministic fast path, and a `tool_schema` for the LLM. Add one path only and
half of MEDO can't reach it.

```python
# skills/tide.py
import re

from skills.base import Skill, SkillRequest, SkillResult


class TideSkill(Skill):
    name = "tide"                     # also the LLM tool name
    description = "Report the next high or low tide for a coastal town."
    controls_pc = False               # it only answers

    patterns = [
        re.compile(r"\bwhen(?:'s| is) (?:the )?(?:next )?(?P<kind>high|low) tide\b",
                   re.IGNORECASE),
        re.compile(r"\bкога е (?P<kind2>плима|осека)\b", re.IGNORECASE),
    ]

    #: Plain phrases for the SEMANTIC tier. NOT synonyms of the regexes above —
    #: those belong in `patterns`. This is how someone might ask sideways.
    routing_phrases = [
        "is the tide coming in",
        "how long until the water goes out",
    ]

    async def execute(self, request: SkillRequest) -> SkillResult:
        groups = request.match.groupdict() if request.match else {}
        kind = (request.args.get("kind") or groups.get("kind")
                or groups.get("kind2") or "high")
        return SkillResult(f"The next {kind} tide is at half past four.")

    def tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["high", "low"]},
                    },
                },
            },
        }
```

Register it in `main.py::build_registry()`. **Order sets fast-path precedence**
on overlapping patterns — a narrow skill goes before a broad one:

```python
from skills.tide import TideSkill

registry.register(TideSkill())
```

### If the skill ACTS on the machine

Declare it, and MEDO gates it for you — the PC-control switch, the policy
engine, and the confirmation prompt all key off these:

```python
class UnlockDoorSkill(Skill):
    name = "unlock_door"
    description = "Unlock the front door."
    controls_pc = True                      # the PC-control switch gates it
    requires_confirmation = True            # never runs without a spoken yes
    capabilities = frozenset({Capability.CONTROL_DEVICE})

    async def preflight(self, request):
        """Checks that must happen BEFORE the user is asked to confirm."""
        if not self._lock.paired():
            return SkillResult("The lock isn't paired yet.", success=False)
        return None

    def confirmation_prompt(self, request):
        return "That unlocks the front door. Are you sure?"

    async def execute(self, request):
        # The router holds this skill at the confirmation gate, so reaching
        # here means the user said yes.
        return SkillResult("Unlocked.")
```

Do not write the confirmation check inside `execute` and rely on it: the router
gates a `requires_confirmation` skill *before* calling you, and
`tests/test_destructive_routes.py` proves no tier can get past it.

---

## A plugin — a skill you can drop in without touching the repo

Same `Skill` contract, discovered from `plugins/`. Use one when the capability
is yours rather than MEDO's — a house-specific integration, something with a
private API key.

```python
# plugins/bins.py
"""When do the bins go out?"""
import re

from skills.base import Skill, SkillRequest, SkillResult


class BinsSkill(Skill):
    name = "bins"
    description = "Say which bin goes out this week."
    patterns = [re.compile(r"\bwhich bin\b", re.IGNORECASE)]

    async def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult("Green bin this week.")


#: Discovered by name. Return every skill this plugin provides.
def register():
    return [BinsSkill()]
```

Plugins are **capability-reviewed at load** when `security.plugin_approval` is
on: MEDO reads the file statically, reports what it would be able to do, and
does not import it until approved.

```bash
python -m security.plugins            # review and approve
```

A plugin gets no privileges a built-in skill doesn't have. One that declares
`requires_confirmation` is gated exactly the same way.

---

## Testing what you added

```bash
.venv\Scripts\python -m pytest tests/test_your_skill.py -q     # Windows
python main.py --once "when's the next high tide"              # route one request
```

Test **both languages**. The fast path is regex, and a pattern written in
English is deaf to the Macedonian phrasing — the request then falls through to
the LLM, which narrates or invents. `core/mk.py` holds the shared Macedonian
verb vocabulary (`OPEN`, `CLOSE`, `MAKE`, `SEARCH`, `NOUN_ENDING`); see
`docs/decisions.md` for the three recurring traps (declension, conjugation,
reply language).
