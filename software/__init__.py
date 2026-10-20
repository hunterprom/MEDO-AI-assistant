"""Control OTHER local software by command — direct local control, NOT MCP.

Connectors declare an app + a list of actions; the registry exposes each action
as a routable skill (fast path + LLM tool) gated by the central policy engine.
See docs/software-connectors.md and software/connector_base.py.
"""
