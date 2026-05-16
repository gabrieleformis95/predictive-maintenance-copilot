"""Ollama backend — calls a local Ollama server via its native /api/chat endpoint.

Requires `ollama serve` running locally. Native API (not the OpenAI shim)
because it surfaces the `format=json` field that constrains decoding to
valid JSON, which we want for structured alerts.
"""

from __future__ import annotations

import contextlib
import json

import httpx

from src.llm.client import LLMClient, ResponseFormat


class OllamaClient(LLMClient):
    provider_name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_s)

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
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if response_format == "json":
            payload["format"] = "json"

        try:
            r = self._client.post("/api/chat", json=payload)
            r.raise_for_status()
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running?"
            ) from e

        data = r.json()
        # /api/chat returns {"message": {"role": "assistant", "content": "..."}}
        return str(data["message"]["content"])

    def __repr__(self) -> str:
        return f"OllamaClient(base_url={self.base_url!r}, model={self.model!r})"

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._client.close()


# Convenience: a tiny helper to extract JSON when the model occasionally
# wraps it in markdown fences despite format="json".
def parse_json_lenient(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        # strip ```json ... ```
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return dict(json.loads(text))
