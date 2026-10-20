# Routines — chaining skills without writing Python

Drop a `.yaml` file in here and MEDO registers it as a skill on the next start.
It is then routable like any other: say the trigger phrase, let it fire on a
schedule, or let the model call it as a tool.

```yaml
name: morning briefing          # required; becomes the skill `routine_morning_briefing`
description: What today looks like.
trigger:
  phrase: ["good morning", "добро утро"]   # matched LITERALLY, not as regex
  at: "07:30"                              # HH:MM, 24-hour
  days: [mon, tue, wed, thu, fri]          # omit for every day
params:
  city: Skopje                  # defaults; the LLM may override them as tool args
steps:
  - say: "Good morning."        # a literal line
  - skill: weather              # a skill, by its registry name
    args: {city: "{city}"}      # {placeholders} come from params + earlier steps
  - skill: news
    into: headlines             # store this step's speech as {headlines}
```

**Templating** substitutes `{name}` from `params`, from `{previous}` (the last
step's speech), and from any `into:` name. It is a dictionary lookup — there is
no expression language, deliberately.

**What a routine may not do:** call a skill that needs a spoken confirmation.
MEDO refuses to load such a routine and says which step. A confirmation prompt
in the middle of a chain has nobody to answer it when the routine fired on a
timer, and when it fired on a phrase it would turn one "yes" into permission for
whatever the rest of the file said. Routines compose capability; they never
grant it.

**When a step fails** the rest still runs, and MEDO says what it could not do —
a briefing that loses the news should still tell you the weather, and should not
pretend it was complete.

Check your file without starting MEDO:

    python -m core.composer --check
