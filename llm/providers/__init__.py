"""Pluggable LLM brains behind one `BrainProvider` interface.

Local (Ollama) is the default and fallback; OpenAI-compatible clouds are opt-in.
The registry that turns config entries into providers is wired in S2.
"""

from llm.providers.base import BrainProvider
from llm.providers.ollama import OllamaProvider
from llm.providers.openai_compat import KNOWN_OPENAI_COMPAT, OpenAICompatProvider

__all__ = [
    "BrainProvider",
    "OllamaProvider",
    "OpenAICompatProvider",
    "KNOWN_OPENAI_COMPAT",
]
