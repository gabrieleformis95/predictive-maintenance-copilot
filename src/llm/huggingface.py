"""Hugging Face Inference API backend.

Two routes are supported:
  - Serverless Inference API   (free tier, can have cold starts)
  - Dedicated Inference Endpoint (your own URL, paid)

We use the chat-completions-compatible router endpoint where available.
"""

from __future__ import annotations

import contextlib

import httpx

from src.llm.client import LLMClient, ResponseFormat


class HuggingFaceClient(LLMClient):
    provider_name = "huggingface"
    DEFAULT_BASE_URL = "https://api-inference.huggingface.co/v1"

    def __init__(
        self,
        token: str,
        model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        base_url: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.token = token
        self.model = model
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {token}"},
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
        return f"HuggingFaceClient(model={self.model!r})"

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._client.close()
