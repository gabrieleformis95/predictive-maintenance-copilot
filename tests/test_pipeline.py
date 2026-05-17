"""Tests for src/pipeline.py - pure helpers, model loading, LLM calls, and explain_anomaly."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.pipeline import _build_retrieval_query, _sensor_deviations, _severity

# ---------------------------------------------------------------------------
# _sensor_deviations
# ---------------------------------------------------------------------------


def test_sensor_deviations_sorted_by_magnitude():
    """GIVEN a window where feature 'b' has the largest absolute mean deviation.
    WHEN _sensor_deviations is called.
    THEN results are sorted descending by absolute z-score and 'b' is first.
    """
    window = np.array([[1.0, -3.0, 0.5]] * 10)
    result = _sensor_deviations(window, ["a", "b", "c"])
    magnitudes = [abs(z) for _, z in result]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert result[0][0] == "b"


def test_sensor_deviations_returns_all_features():
    """GIVEN a window with 4 features.
    WHEN _sensor_deviations is called.
    THEN all 4 feature names are present in the result.
    """
    window = np.ones((5, 4))
    result = _sensor_deviations(window, ["s1", "s2", "s3", "s4"])
    assert {name for name, _ in result} == {"s1", "s2", "s3", "s4"}


# ---------------------------------------------------------------------------
# _build_retrieval_query
# ---------------------------------------------------------------------------


def test_build_retrieval_query_includes_top_n_sensors():
    """GIVEN 4 sensor deviations and top_n=3.
    WHEN _build_retrieval_query is called.
    THEN the query includes the top 3 sensor names and excludes the 4th.
    """
    devs = [("sensor_4", 2.5), ("sensor_7", -1.8), ("sensor_2", 0.9), ("sensor_1", 0.1)]
    query = _build_retrieval_query(devs, top_n=3)
    assert "sensor_4" in query
    assert "sensor_7" in query
    assert "sensor_2" in query
    assert "sensor_1" not in query


# ---------------------------------------------------------------------------
# _severity
# ---------------------------------------------------------------------------


def test_severity_critical():
    """GIVEN a score/threshold ratio >= 1.5.
    WHEN _severity is called.
    THEN 'critical' is returned.
    """
    assert _severity(1.5, 1.0) == "critical"


def test_severity_warning():
    """GIVEN a score/threshold ratio >= 1.0 and < 1.5.
    WHEN _severity is called.
    THEN 'warning' is returned.
    """
    assert _severity(1.0, 1.0) == "warning"


def test_severity_info():
    """GIVEN a score/threshold ratio < 1.0.
    WHEN _severity is called.
    THEN 'info' is returned.
    """
    assert _severity(0.5, 1.0) == "info"


# ---------------------------------------------------------------------------
# _default_checkpoint
# ---------------------------------------------------------------------------


def test_default_checkpoint_path():
    """GIVEN no arguments.
    WHEN _default_checkpoint is called.
    THEN the path contains 'autoencoder_FD001.pt'.
    """
    from src.pipeline import _default_checkpoint

    assert "autoencoder_FD001.pt" in _default_checkpoint()


# ---------------------------------------------------------------------------
# _load_model
# ---------------------------------------------------------------------------


def test_load_model_returns_eval_model(fake_checkpoint):
    """GIVEN a valid checkpoint file with 3 features.
    WHEN _load_model is called.
    THEN it returns a model in eval mode and the correct checkpoint metadata.
    """
    from src.pipeline import _load_model

    _load_model.cache_clear()
    model, ckpt = _load_model(fake_checkpoint)
    assert not model.training
    assert ckpt["feature_columns"] == ["s1", "s2", "s3"]
    _load_model.cache_clear()


def test_load_model_caches_result(fake_checkpoint):
    """GIVEN a valid checkpoint file.
    WHEN _load_model is called twice with the same path.
    THEN torch.load is called only once (lru_cache hit on the second call).
    """
    from src.pipeline import _load_model

    _load_model.cache_clear()
    with patch("torch.load", wraps=torch.load) as mock_load:
        _load_model(fake_checkpoint)
        _load_model(fake_checkpoint)
    assert mock_load.call_count == 1
    _load_model.cache_clear()


# ---------------------------------------------------------------------------
# _call_ollama
# ---------------------------------------------------------------------------


def test_call_ollama_returns_content():
    """GIVEN a mocked httpx.post returning a valid Ollama JSON response.
    WHEN _call_ollama is called.
    THEN the message content string is returned.
    """
    from src.pipeline import _call_ollama

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "anomaly detected"}}
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.post", return_value=mock_resp):
        assert _call_ollama("sys", "usr") == "anomaly detected"


# ---------------------------------------------------------------------------
# _call_openai_compat
# ---------------------------------------------------------------------------


def test_call_openai_compat_returns_content():
    """GIVEN a mocked httpx.post returning a valid OpenAI-compatible JSON response.
    WHEN _call_openai_compat is called.
    THEN the choices[0].message.content string is returned.
    """
    from src.pipeline import _call_openai_compat

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "maintenance needed"}}]}
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.post", return_value=mock_resp):
        result = _call_openai_compat("sys", "usr", "http://base", "key", "model")
    assert result == "maintenance needed"


# ---------------------------------------------------------------------------
# _call_llm routing
# ---------------------------------------------------------------------------


def _mock_cfg(**kwargs):
    m = MagicMock()
    m.llm_provider = kwargs.get("llm_provider", "ollama")
    m.groq_api_key = kwargs.get("groq_api_key")
    m.groq_base_url = kwargs.get("groq_base_url", "https://api.groq.com/openai/v1")
    m.groq_model = kwargs.get("groq_model", "llama-3.3-70b")
    m.openai_api_key = kwargs.get("openai_api_key")
    return m


def test_call_llm_routes_to_ollama():
    """GIVEN llm_provider='ollama'.
    WHEN _call_llm is called.
    THEN _call_ollama is invoked and its result is returned.
    """
    from src.pipeline import _call_llm

    with (
        patch("src.config.settings", _mock_cfg(llm_provider="ollama")),
        patch("src.pipeline._call_ollama", return_value="ollama-reply") as mock,
    ):
        result = _call_llm("sys", "usr")
    assert result == "ollama-reply"
    mock.assert_called_once_with("sys", "usr")


def test_call_llm_routes_to_groq():
    """GIVEN llm_provider='groq' and a valid GROQ_API_KEY.
    WHEN _call_llm is called.
    THEN _call_openai_compat is invoked and its result is returned.
    """
    from src.pipeline import _call_llm

    with (
        patch("src.config.settings", _mock_cfg(llm_provider="groq", groq_api_key="k")),
        patch("src.pipeline._call_openai_compat", return_value="groq-reply") as mock,
    ):
        result = _call_llm("sys", "usr")
    assert result == "groq-reply"
    mock.assert_called_once()


def test_call_llm_groq_raises_without_key():
    """GIVEN llm_provider='groq' and no GROQ_API_KEY.
    WHEN _call_llm is called.
    THEN RuntimeError is raised mentioning GROQ_API_KEY.
    """
    from src.pipeline import _call_llm

    with (
        patch("src.config.settings", _mock_cfg(llm_provider="groq", groq_api_key=None)),
        pytest.raises(RuntimeError, match="GROQ_API_KEY"),
    ):
        _call_llm("sys", "usr")


def test_call_llm_routes_to_openai():
    """GIVEN llm_provider='openai' and a valid OPENAI_API_KEY.
    WHEN _call_llm is called.
    THEN _call_openai_compat is invoked with the OpenAI base_url.
    """
    from src.pipeline import _call_llm

    with (
        patch(
            "src.config.settings",
            _mock_cfg(llm_provider="openai", openai_api_key="k"),
        ),
        patch("src.pipeline._call_openai_compat", return_value="openai-reply") as mock,
    ):
        result = _call_llm("sys", "usr")
    assert result == "openai-reply"
    assert "openai.com" in mock.call_args.kwargs["base_url"]


def test_call_llm_openai_raises_without_key():
    """GIVEN llm_provider='openai' and no OPENAI_API_KEY.
    WHEN _call_llm is called.
    THEN RuntimeError is raised mentioning OPENAI_API_KEY.
    """
    from src.pipeline import _call_llm

    with (
        patch("src.config.settings", _mock_cfg(llm_provider="openai", openai_api_key=None)),
        pytest.raises(RuntimeError, match="OPENAI_API_KEY"),
    ):
        _call_llm("sys", "usr")


def test_call_llm_ollama_falls_back_to_groq():
    """GIVEN llm_provider='ollama', Ollama raises, and GROQ_API_KEY is set.
    WHEN _call_llm is called.
    THEN _call_openai_compat is invoked as fallback and returns its result.
    """
    from src.pipeline import _call_llm

    with (
        patch("src.config.settings", _mock_cfg(llm_provider="ollama", groq_api_key="k")),
        patch("src.pipeline._call_ollama", side_effect=RuntimeError("offline")),
        patch("src.pipeline._call_openai_compat", return_value="fallback") as mock,
    ):
        result = _call_llm("sys", "usr")
    assert result == "fallback"
    mock.assert_called_once()


def test_call_llm_ollama_raises_when_no_fallback():
    """GIVEN llm_provider='ollama', Ollama raises, and no GROQ_API_KEY.
    WHEN _call_llm is called.
    THEN RuntimeError is raised.
    """
    from src.pipeline import _call_llm

    with (
        patch("src.config.settings", _mock_cfg(llm_provider="ollama", groq_api_key=None)),
        patch("src.pipeline._call_ollama", side_effect=RuntimeError("offline")),
        pytest.raises(RuntimeError),
    ):
        _call_llm("sys", "usr")


# ---------------------------------------------------------------------------
# explain_anomaly
# ---------------------------------------------------------------------------


def test_explain_anomaly_healthy_returns_none_alert():
    """GIVEN a zero-valued window and a threshold of 1.0 (score will be ~0).
    WHEN explain_anomaly is called.
    THEN score is ~0 and alert is None.
    """
    window = np.zeros((30, 3), dtype=np.float32)
    mock_model = MagicMock()
    mock_model.return_value = torch.zeros(1, 30, 3)

    with patch("src.pipeline._load_model", return_value=(mock_model, {})):
        from src.pipeline import explain_anomaly

        score, alert = explain_anomaly(
            window, ["s1", "s2", "s3"], threshold=1.0, checkpoint_path="dummy.pt"
        )
    assert score == pytest.approx(0.0, abs=1e-6)
    assert alert is None


def test_explain_anomaly_anomalous_returns_alert():
    """GIVEN a ones-valued window and threshold=0.5 (model returns zeros, so score=1.0).
    WHEN explain_anomaly is called.
    THEN score ~1.0 and alert has severity='warning'.
    """
    window = np.ones((30, 3), dtype=np.float32)
    mock_model = MagicMock()
    mock_model.return_value = torch.zeros(1, 30, 3)
    raw = json.dumps({
        "severity": "warning",
        "probable_cause": "High reconstruction error.",
        "recommended_action": "Inspect equipment.",
        "affected_sensors": ["s1"],
        "citations": [],
    })
    with (
        patch("src.pipeline._load_model", return_value=(mock_model, {})),
        patch("src.pipeline.retrieve", return_value=[]),
        patch("src.pipeline._call_llm", return_value=raw),
    ):
        from src.pipeline import explain_anomaly

        score, alert = explain_anomaly(
            window, ["s1", "s2", "s3"], threshold=0.5, checkpoint_path="dummy.pt"
        )
    assert score == pytest.approx(1.0, abs=1e-5)
    assert alert is not None
    assert alert.severity == "warning"


def test_explain_anomaly_llm_failure_returns_rule_based_alert():
    """GIVEN an anomalous window and an LLM that raises RuntimeError.
    WHEN explain_anomaly is called.
    THEN a rule-based fallback alert is returned instead of propagating the error.
    """
    window = np.ones((30, 3), dtype=np.float32)
    mock_model = MagicMock()
    mock_model.return_value = torch.zeros(1, 30, 3)
    with (
        patch("src.pipeline._load_model", return_value=(mock_model, {})),
        patch("src.pipeline.retrieve", return_value=[]),
        patch("src.pipeline._call_llm", side_effect=RuntimeError("LLM down")),
    ):
        from src.pipeline import explain_anomaly

        _, alert = explain_anomaly(
            window, ["s1", "s2", "s3"], threshold=0.5, checkpoint_path="dummy.pt"
        )
    assert alert is not None
    assert alert.recommended_action == "Inspect equipment manually."
