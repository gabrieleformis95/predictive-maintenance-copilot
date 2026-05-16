"""Tests for src/pipeline.py - pure helpers and mocked explain_anomaly."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.pipeline import _build_retrieval_query, _sensor_deviations, _severity


def test_sensor_deviations_sorted_by_magnitude():
    window = np.array([[1.0, -3.0, 0.5]] * 10)
    result = _sensor_deviations(window, ["a", "b", "c"])
    magnitudes = [abs(z) for _, z in result]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert result[0][0] == "b"


def test_sensor_deviations_returns_all_features():
    window = np.ones((5, 4))
    result = _sensor_deviations(window, ["s1", "s2", "s3", "s4"])
    assert {name for name, _ in result} == {"s1", "s2", "s3", "s4"}


def test_build_retrieval_query_uses_top_n():
    deviations = [("sensor_4", 2.5), ("sensor_7", -1.8), ("sensor_2", 0.9), ("sensor_1", 0.1)]
    query = _build_retrieval_query(deviations, top_n=3)
    assert "sensor_4" in query
    assert "sensor_7" in query
    assert "sensor_2" in query
    assert "sensor_1" not in query


def test_severity_thresholds():
    assert _severity(1.5, 1.0) == "critical"
    assert _severity(1.0, 1.0) == "warning"
    assert _severity(0.5, 1.0) == "info"


def test_explain_anomaly_healthy_returns_none_alert():
    n_features, window_size = 3, 30
    window = np.zeros((window_size, n_features), dtype=np.float32)

    mock_model = MagicMock()
    mock_model.return_value = torch.zeros(1, window_size, n_features)

    with patch("src.pipeline._load_model", return_value=(mock_model, {})):
        from src.pipeline import explain_anomaly

        score, alert = explain_anomaly(
            window, ["s1", "s2", "s3"], threshold=1.0, checkpoint_path="dummy.pt"
        )

    assert score == pytest.approx(0.0, abs=1e-6)
    assert alert is None


def test_explain_anomaly_anomalous_returns_alert():
    n_features, window_size = 3, 30
    # ones input, model returns zeros → score = 1.0
    window = np.ones((window_size, n_features), dtype=np.float32)

    mock_model = MagicMock()
    mock_model.return_value = torch.zeros(1, window_size, n_features)

    raw_alert = json.dumps({
        "severity": "warning",
        "probable_cause": "High reconstruction error.",
        "recommended_action": "Inspect equipment.",
        "affected_sensors": ["s1"],
        "citations": [],
    })

    with (
        patch("src.pipeline._load_model", return_value=(mock_model, {})),
        patch("src.pipeline.retrieve", return_value=[]),
        patch("src.pipeline._call_llm", return_value=raw_alert),
    ):
        from src.pipeline import explain_anomaly

        score, alert = explain_anomaly(
            window, ["s1", "s2", "s3"], threshold=0.5, checkpoint_path="dummy.pt"
        )

    assert score == pytest.approx(1.0, abs=1e-5)
    assert alert is not None
    assert alert.severity == "warning"


def test_explain_anomaly_llm_failure_returns_rule_based_alert():
    n_features, window_size = 3, 30
    window = np.ones((window_size, n_features), dtype=np.float32)

    mock_model = MagicMock()
    mock_model.return_value = torch.zeros(1, window_size, n_features)

    with (
        patch("src.pipeline._load_model", return_value=(mock_model, {})),
        patch("src.pipeline.retrieve", return_value=[]),
        patch("src.pipeline._call_llm", side_effect=RuntimeError("LLM down")),
    ):
        from src.pipeline import explain_anomaly

        score, alert = explain_anomaly(
            window, ["s1", "s2", "s3"], threshold=0.5, checkpoint_path="dummy.pt"
        )

    assert alert is not None
    assert alert.recommended_action == "Inspect equipment manually."
