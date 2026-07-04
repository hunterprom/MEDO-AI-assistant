"""Example plugin: coin flips and dice rolls.

A complete working sample of the plugin SDK — auto-discovered from this
folder, callable by voice ("flip a coin", "roll a d20") and by the LLM as a
tool (with a ``sides`` argument). Copy it as the starting point for your own.
"""

from __future__ import annotations

import random
import re
from typing import Any

from skills.base import Skill, SkillRequest, SkillResult


class DiceSkill(Skill):
    name = "dice"
    description = (
        "Flip a coin or roll dice. Use when the user wants a random outcome, "
        "a coin flip, or a dice/d20 roll."
    )

    patterns = [
        re.compile(r"\bflip\s+a\s+coin\b", re.IGNORECASE),
        re.compile(r"\broll\s+(?:a\s+|the\s+)?(?:die|dice|d(?P<sides>\d+))\b", re.IGNORECASE),
    ]

    async def execute(self, request: SkillRequest) -> SkillResult:
        text = request.text.lower()
        gd = request.match.groupdict() if request.match else {}
        if "coin" in text or str(request.args.get("sides")) == "2":
            return SkillResult(f"It's {random.choice(['heads', 'tails'])}.")
        try:
            sides = int(request.args.get("sides") or gd.get("sides") or 6)
        except (TypeError, ValueError):
            sides = 6
        sides = max(2, min(sides, 1000))
        return SkillResult(f"You rolled a {random.randint(1, sides)} on the d{sides}.")

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sides": {
                            "type": "integer",
                            "description": "number of sides (2 = coin flip, default 6)",
                        }
                    },
                },
            },
        }
