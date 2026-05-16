"""Common interface for LLM providers + a small factory.

Why bother with the indirection? Two reasons:

  1. Swap providers without touching call sites. Today we use Ollama for local
     dev; tomorrow we deploy to Hugging Face Spaces and switch to Groq via env
     var alone.
  2. Keep tests free of network calls. `tests/test_llm.py` monkeypatches the
     factory to return a FakeLLMClient.

Usage
-----
>>> from src.llm import get_llm_client
>>> client = get_llm_client()
>>> reply = client.complete(
...     system="You are an industrial maintenance assistant.",
...     user="Summarise the alert in one sentence.",
...     response_format="json",
... )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from src.config import settings

ResponseFormat = Literal["text", "json"]


@dataclass
class LLMMessage:
    """One chat message exchanged with the model."""

    role: Literal["system", "user", "assistant"]
    content: str


class LLMClient(ABC):
    """Provider-agnostic completion interface."""

    provider_name: str

    @abstractmethod
    def complete(
        self,
        *,
        system: str | None = None,
        user: str,
        response_format: ResponseFormat = "text",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Return the LLM's reply as a string.

        When `response_format="json"`, the implementation should ask the
        underlying API to constrain output to valid JSON when supported.
        Callers are still expected to defensively parse the result.
        """


def get_llm_client(provider: str | None = None) -> LLMClient:
    """Factory. Returns the configured LLM client.

    Parameters
    ----------
    provider: optional override. When None, reads from settings.llm_provider.
    """
    name = (provider or settings.llm_provider).lower().strip()

    if name == "ollama":
        from src.llm.ollama import OllamaClient

        return OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
        )

    if name == "groq":
        from src.llm.groq_backend import GroqClient

        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set; cannot use 'groq' provider.")
        model = settings.llm_model if settings.llm_model != "qwen2.5:7b" else settings.groq_model
        return GroqClient(api_key=settings.groq_api_key, model=model)

    if name == "huggingface":
        from src.llm.huggingface import HuggingFaceClient

        if not settings.hf_token:
            raise RuntimeError("HF_TOKEN is not set; cannot use 'huggingface' provider.")
        return HuggingFaceClient(token=settings.hf_token, model=settings.llm_model)

    if name == "openai":
        from src.llm.openai_backend import OpenAIClient

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot use 'openai' provider.")
        return OpenAIClient(api_key=settings.openai_api_key, model=settings.llm_model)

    raise ValueError(
        f"Unknown LLM provider: {name!r}. Supported: ollama | groq | huggingface | openai."
    )
