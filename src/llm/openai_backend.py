"""OpenAI backend (kept for completeness / when budget allows).

Module is `openai_backend` to avoid shadowing the official `openai` package
that any caller may have installed.
"""

from __future__ import annotations

import contextlib

import httpx

from src.llm.client import LLMClient, ResponseFormat


class OpenAIClient(LLMClient):
    provider_name = "openai"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def complete(
        self,
        *,
        system: str | None = None,
        user: str,
        response_format: ResponseFormat = "text",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        r = self._client.post("/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        return str(data["choices"][0]["message"]["content"])

    def __repr__(self) -> str:
        return f"OpenAIClient(model={self.model!r})"

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._client.close()
