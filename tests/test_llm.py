"""Tests for src/llm backends and factory."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from src.llm.client import get_llm_client
from src.llm.groq_backend import GroqClient
from src.llm.huggingface import HuggingFaceClient
from src.llm.ollama import OllamaClient, parse_json_lenient
from src.llm.openai_backend import OpenAIClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(body: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = body
    resp.raise_for_status.return_value = None
    return resp


def _ollama_body(content: str = "reply") -> dict:
    return {"message": {"content": content}}


def _openai_body(content: str = "reply") -> dict:
    return {"choices": [{"message": {"content": content}}]}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_get_llm_client_ollama():
    """GIVEN provider name 'ollama'.
    WHEN get_llm_client is called.
    THEN an OllamaClient instance is returned.
    """
    assert isinstance(get_llm_client("ollama"), OllamaClient)


def test_get_llm_client_groq_with_key(monkeypatch):
    """GIVEN provider 'groq' and GROQ_API_KEY is set.
    WHEN get_llm_client is called.
    THEN a GroqClient instance is returned.
    """
    import src.llm.client as mod

    monkeypatch.setattr(mod.settings, "groq_api_key", "fake-key")
    assert isinstance(get_llm_client("groq"), GroqClient)


def test_get_llm_client_groq_without_key(monkeypatch):
    """GIVEN provider 'groq' and GROQ_API_KEY is not set.
    WHEN get_llm_client is called.
    THEN RuntimeError is raised mentioning GROQ_API_KEY.
    """
    import src.llm.client as mod

    monkeypatch.setattr(mod.settings, "groq_api_key", None)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        get_llm_client("groq")


def test_get_llm_client_huggingface_with_token(monkeypatch):
    """GIVEN provider 'huggingface' and HF_TOKEN is set.
    WHEN get_llm_client is called.
    THEN a HuggingFaceClient instance is returned.
    """
    import src.llm.client as mod

    monkeypatch.setattr(mod.settings, "hf_token", "fake-token")
    assert isinstance(get_llm_client("huggingface"), HuggingFaceClient)


def test_get_llm_client_huggingface_without_token(monkeypatch):
    """GIVEN provider 'huggingface' and HF_TOKEN is not set.
    WHEN get_llm_client is called.
    THEN RuntimeError is raised mentioning HF_TOKEN.
    """
    import src.llm.client as mod

    monkeypatch.setattr(mod.settings, "hf_token", None)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        get_llm_client("huggingface")


def test_get_llm_client_openai_with_key(monkeypatch):
    """GIVEN provider 'openai' and OPENAI_API_KEY is set.
    WHEN get_llm_client is called.
    THEN an OpenAIClient instance is returned.
    """
    import src.llm.client as mod

    monkeypatch.setattr(mod.settings, "openai_api_key", "fake-key")
    assert isinstance(get_llm_client("openai"), OpenAIClient)


def test_get_llm_client_openai_without_key(monkeypatch):
    """GIVEN provider 'openai' and OPENAI_API_KEY is not set.
    WHEN get_llm_client is called.
    THEN RuntimeError is raised mentioning OPENAI_API_KEY.
    """
    import src.llm.client as mod

    monkeypatch.setattr(mod.settings, "openai_api_key", None)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_llm_client("openai")


def test_get_llm_client_unknown_provider():
    """GIVEN an unrecognised provider name.
    WHEN get_llm_client is called.
    THEN ValueError is raised.
    """
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm_client("gpt42")


# ---------------------------------------------------------------------------
# OllamaClient
# ---------------------------------------------------------------------------


def test_ollama_complete_returns_content():
    """GIVEN a mocked HTTP response with content 'hello'.
    WHEN OllamaClient.complete() is called.
    THEN the returned string equals 'hello'.
    """
    client = OllamaClient()
    client._client = MagicMock()
    client._client.post.return_value = _mock_response(_ollama_body("hello"))
    assert client.complete(user="test") == "hello"


def test_ollama_complete_includes_system_message():
    """GIVEN both a system prompt and a user message.
    WHEN OllamaClient.complete() is called.
    THEN the request payload contains messages with roles ['system', 'user'].
    """
    client = OllamaClient()
    client._client = MagicMock()
    client._client.post.return_value = _mock_response(_ollama_body("ok"))
    client.complete(system="sys", user="usr")
    payload = client._client.post.call_args[1]["json"]
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]


def test_ollama_complete_json_format_sets_payload_field():
    """GIVEN response_format='json'.
    WHEN OllamaClient.complete() is called.
    THEN the request payload includes format='json'.
    """
    client = OllamaClient()
    client._client = MagicMock()
    client._client.post.return_value = _mock_response(_ollama_body("{}"))
    client.complete(user="test", response_format="json")
    payload = client._client.post.call_args[1]["json"]
    assert payload.get("format") == "json"


def test_ollama_connect_error_raises_runtime():
    """GIVEN the Ollama server is unreachable (ConnectError).
    WHEN OllamaClient.complete() is called.
    THEN RuntimeError is raised with a message mentioning Ollama.
    """
    client = OllamaClient()
    client._client = MagicMock()
    client._client.post.side_effect = httpx.ConnectError("refused")
    with pytest.raises(RuntimeError, match="Ollama"):
        client.complete(user="test")


def test_ollama_repr_contains_url_and_model():
    """GIVEN an OllamaClient with known base_url and model.
    WHEN repr() is called.
    THEN the string includes both the URL and model name.
    """
    client = OllamaClient(base_url="http://localhost:11434", model="qwen")
    r = repr(client)
    assert "http://localhost:11434" in r
    assert "qwen" in r


# ---------------------------------------------------------------------------
# GroqClient
# ---------------------------------------------------------------------------


def test_groq_complete_returns_content():
    """GIVEN a mocked Groq response with content 'groq-reply'.
    WHEN GroqClient.complete() is called.
    THEN the returned string equals 'groq-reply'.
    """
    client = GroqClient(api_key="fake")
    client._client = MagicMock()
    client._client.post.return_value = _mock_response(_openai_body("groq-reply"))
    assert client.complete(user="q") == "groq-reply"


def test_groq_complete_json_format_sets_response_format():
    """GIVEN response_format='json'.
    WHEN GroqClient.complete() is called.
    THEN the payload includes response_format={'type': 'json_object'}.
    """
    client = GroqClient(api_key="fake")
    client._client = MagicMock()
    client._client.post.return_value = _mock_response(_openai_body("{}"))
    client.complete(user="q", response_format="json")
    payload = client._client.post.call_args[1]["json"]
    assert payload.get("response_format") == {"type": "json_object"}


def test_groq_complete_no_system_sends_only_user():
    """GIVEN no system prompt.
    WHEN GroqClient.complete() is called.
    THEN the payload messages contain only the user role.
    """
    client = GroqClient(api_key="fake")
    client._client = MagicMock()
    client._client.post.return_value = _mock_response(_openai_body("ok"))
    client.complete(user="only-user")
    payload = client._client.post.call_args[1]["json"]
    assert [m["role"] for m in payload["messages"]] == ["user"]


# ---------------------------------------------------------------------------
# OpenAIClient
# ---------------------------------------------------------------------------


def test_openai_complete_returns_content():
    """GIVEN a mocked OpenAI response with content 'openai-reply'.
    WHEN OpenAIClient.complete() is called.
    THEN the returned string equals 'openai-reply'.
    """
    client = OpenAIClient(api_key="fake")
    client._client = MagicMock()
    client._client.post.return_value = _mock_response(_openai_body("openai-reply"))
    assert client.complete(user="q") == "openai-reply"


def test_openai_complete_json_format():
    """GIVEN response_format='json'.
    WHEN OpenAIClient.complete() is called.
    THEN the payload includes response_format={'type': 'json_object'}.
    """
    client = OpenAIClient(api_key="fake")
    client._client = MagicMock()
    client._client.post.return_value = _mock_response(_openai_body("{}"))
    client.complete(user="q", response_format="json")
    payload = client._client.post.call_args[1]["json"]
    assert payload.get("response_format") == {"type": "json_object"}


# ---------------------------------------------------------------------------
# HuggingFaceClient
# ---------------------------------------------------------------------------


def test_huggingface_complete_returns_content():
    """GIVEN a mocked HuggingFace inference response with content 'hf-reply'.
    WHEN HuggingFaceClient.complete() is called.
    THEN the returned string equals 'hf-reply'.
    """
    client = HuggingFaceClient(token="fake")
    client._client = MagicMock()
    client._client.post.return_value = _mock_response(_openai_body("hf-reply"))
    assert client.complete(user="q") == "hf-reply"


def test_huggingface_complete_with_system():
    """GIVEN a system prompt and user message.
    WHEN HuggingFaceClient.complete() is called.
    THEN the payload messages contain both system and user roles.
    """
    client = HuggingFaceClient(token="fake")
    client._client = MagicMock()
    client._client.post.return_value = _mock_response(_openai_body("ok"))
    client.complete(system="sys", user="usr")
    payload = client._client.post.call_args[1]["json"]
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]


# ---------------------------------------------------------------------------
# parse_json_lenient
# ---------------------------------------------------------------------------


def test_parse_json_lenient_plain_json():
    """GIVEN a plain JSON string.
    WHEN parse_json_lenient is called.
    THEN the parsed dict is returned unchanged.
    """
    assert parse_json_lenient('{"key": 1}') == {"key": 1}


def test_parse_json_lenient_strips_markdown_fences():
    """GIVEN a JSON string wrapped in ```json ... ``` markdown fences.
    WHEN parse_json_lenient is called.
    THEN the fences are stripped and the correct dict is returned.
    """
    assert parse_json_lenient('```json\n{"key": 2}\n```') == {"key": 2}


def test_parse_json_lenient_strips_plain_fences():
    """GIVEN a JSON string wrapped in plain ``` fences (no language tag).
    WHEN parse_json_lenient is called.
    THEN the fences are stripped and the correct dict is returned.
    """
    assert parse_json_lenient('```\n{"key": 3}\n```') == {"key": 3}
