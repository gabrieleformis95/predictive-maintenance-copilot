"""Tests for src/api/main.py using FastAPI TestClient."""

from __future__ import annotations

from unittest.mock import patch

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
        TestClient(app) as c,
    ):
        yield c


# ---------------------------------------------------------------------------
# _get_threshold
# ---------------------------------------------------------------------------


def test_get_threshold_reads_from_checkpoint():
    """GIVEN torch.load returns a checkpoint with anomaly_threshold_f1=0.42.
    WHEN _get_threshold is called.
    THEN 0.42 is returned and cached.
    """
    import src.api.main as mod

    mod._threshold_cache = None
    with patch("torch.load", return_value={"anomaly_threshold_f1": 0.42}):
        result = mod._get_threshold()
    assert result == pytest.approx(0.42)
    mod._threshold_cache = None


def test_get_threshold_falls_back_on_error():
    """GIVEN torch.load raises FileNotFoundError.
    WHEN _get_threshold is called.
    THEN the default value 0.33 is returned.
    """
    import src.api.main as mod

    mod._threshold_cache = None
    with patch("torch.load", side_effect=FileNotFoundError("no checkpoint")):
        result = mod._get_threshold()
    assert result == pytest.approx(0.33)
    mod._threshold_cache = None


def test_get_threshold_uses_cache():
    """GIVEN _threshold_cache is already populated with 0.99.
    WHEN _get_threshold is called.
    THEN 0.99 is returned without calling torch.load.
    """
    import src.api.main as mod

    mod._threshold_cache = 0.99
    with patch("torch.load") as mock_load:
        result = mod._get_threshold()
    assert result == pytest.approx(0.99)
    mock_load.assert_not_called()
    mod._threshold_cache = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def test_root(client):
    """GIVEN the API is running.
    WHEN GET / is called.
    THEN 200 is returned with the app name in the body.
    """
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Predictive Maintenance Copilot"


def test_healthz(client):
    """GIVEN the API is running.
    WHEN GET /healthz is called.
    THEN 200 is returned with status, version, and llm_provider fields.
    """
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "llm_provider" in data


def test_predict_healthy(client):
    """GIVEN a zero-valued sensor window and a mocked pipeline returning no anomaly.
    WHEN POST /predict is called.
    THEN 200 is returned with is_anomaly=False and alert=None.
    """
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
    """GIVEN a sensor window and a mocked pipeline returning a warning alert.
    WHEN POST /predict is called.
    THEN 200 is returned with is_anomaly=True and the alert severity.
    """
    alert = AnomalyAlert(
        severity="warning",
        probable_cause="High vibration.",
        recommended_action="Inspect bearings.",
        affected_sensors=["s1"],
        citations=[],
    )
    window = np.ones((30, 3)).tolist()
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
    """GIVEN a (30, 3) window but only 2 feature names.
    WHEN POST /predict is called.
    THEN 400 is returned.
    """
    window = np.zeros((30, 3)).tolist()
    resp = client.post(
        "/predict",
        json={"sensor_window": window, "feature_names": ["s1", "s2"], "threshold": 0.5},
    )
    assert resp.status_code == 400


def test_predict_pipeline_error_returns_500(client):
    """GIVEN the pipeline raises RuntimeError.
    WHEN POST /predict is called.
    THEN 500 is returned.
    """
    window = np.zeros((30, 3)).tolist()
    with patch("src.pipeline.explain_anomaly", side_effect=RuntimeError("model fail")):
        resp = client.post(
            "/predict",
            json={"sensor_window": window, "feature_names": ["s1", "s2", "s3"], "threshold": 0.5},
        )
    assert resp.status_code == 500
