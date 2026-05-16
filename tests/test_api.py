"""Tests for src/api/main.py using TestClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.rag.prompts import AnomalyAlert


@pytest.fixture(scope="module")
def client():
    from src.api.main import app

    with (
        patch("src.pipeline._load_model"),
        patch("src.rag.retriever._get_state", return_value=None),
        patch("src.api.main._get_threshold", return_value=0.33),
    ):
        with TestClient(app) as c:
            yield c


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Predictive Maintenance Copilot"


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "llm_provider" in data


def test_predict_healthy(client):
    window = np.zeros((30, 3)).tolist()
    with patch("src.pipeline.explain_anomaly", return_value=(0.1, None)):
        resp = client.post(
            "/predict",
            json={"sensor_window": window, "feature_names": ["s1", "s2", "s3"], "threshold": 0.5},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_anomaly"] is False
    assert data["alert"] is None


def test_predict_anomaly(client):
    window = np.ones((30, 3)).tolist()
    alert = AnomalyAlert(
        severity="warning",
        probable_cause="High vibration.",
        recommended_action="Inspect bearings.",
        affected_sensors=["s1"],
        citations=[],
    )
    with patch("src.pipeline.explain_anomaly", return_value=(0.8, alert)):
        resp = client.post(
            "/predict",
            json={"sensor_window": window, "feature_names": ["s1", "s2", "s3"], "threshold": 0.5},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_anomaly"] is True
    assert data["alert"]["severity"] == "warning"


def test_predict_feature_name_mismatch(client):
    window = np.zeros((30, 3)).tolist()
    resp = client.post(
        "/predict",
        json={"sensor_window": window, "feature_names": ["s1", "s2"], "threshold": 0.5},
    )
    assert resp.status_code == 400


def test_predict_pipeline_error_returns_500(client):
    window = np.zeros((30, 3)).tolist()
    with patch("src.pipeline.explain_anomaly", side_effect=RuntimeError("model fail")):
        resp = client.post(
            "/predict",
            json={"sensor_window": window, "feature_names": ["s1", "s2", "s3"], "threshold": 0.5},
        )
    assert resp.status_code == 500
