"""Skills: one implementation per capability, reachable from both routes.

Each skill declares regex ``patterns`` (fast path) *and* a ``tool_schema`` (LLM
function calling). Register once; the router and the Ollama client both use it.
"""
