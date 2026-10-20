"""MEDO Cowork — bounded, visible orchestration of MEDO's OWN capabilities.

Hand MEDO a multi-step task by voice; it makes a plan you can hear, confirms the
risky steps, runs them one at a time, and STOPS when one fails instead of
improvising a new plan.

Four rules, and the whole package exists to keep them:

1. **Plan first, show it.** Nothing runs before the plan has been said out loud
   and approved (:mod:`cowork.planner`, :mod:`cowork.plan`).
2. **Confirm the risky.** Approving a plan is not a blank cheque: every
   side-effectful step is re-checked at the policy gate at execution time, and a
   high-impact one asks again (:mod:`cowork.executor`).
3. **One step at a time, narrated.**
4. **Stop on failure.** Never auto-replan, never take an unplanned action to
   "recover" — report where it stopped and offer retry / skip / abort.

This is NOT a general autonomous agent. It orchestrates the skills, software
connectors and MEDO Link devices that are already registered — a plan can only
ever contain something MEDO can genuinely do, resolved against the LIVE
registry. See docs/cowork.md.
"""
