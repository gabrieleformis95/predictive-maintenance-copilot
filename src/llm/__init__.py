"""LLM provider abstraction.

The rest of the codebase imports `get_llm_client()` and calls
`client.complete(...)`. The actual backend (Ollama, Groq, HuggingFace, OpenAI)
is selected at runtime via the LLM_PROVIDER env var.

Add a new provider by:
  1. creating src/llm/<provider>.py with a class implementing LLMClient
  2. adding a branch to `get_llm_client()` in src/llm/client.py
  3. adding the provider name to the Settings.llm_provider type in config.py
"""

from src.llm.client import LLMClient, LLMMessage, get_llm_client

__all__ = ["LLMClient", "LLMMessage", "get_llm_client"]
