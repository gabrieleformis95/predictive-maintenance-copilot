"""FastAPI app exposing the anomaly + explanation pipeline."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import __version__
from src.config import settings
from src.rag.prompts import AnomalyAlert

logger = logging.getLogger(__name__)


_threshold_cache: float | None = None


def _get_threshold() -> float:
    global _threshold_cache
    if _threshold_cache is not None:
        return _threshold_cache
    try:
        ckpt = torch.load(str(settings.checkpoint_path), map_location="cpu", weights_only=True)
        _threshold_cache = float(
            ckpt.get("anomaly_threshold_f1") or ckpt.get("anomaly_threshold") or 0.33
        )
    except Exception:
        _threshold_cache = 0.33
    return _threshold_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up model and retriever at startup so first request is fast.
    from src.pipeline import _load_model
    from src.rag.retriever import _get_state

    logger.info("Warming up model...")
    _load_model(str(settings.checkpoint_path))
    logger.info("Warming up retriever...")
    _get_state()
    logger.info("Caching threshold...")
    _get_threshold()
    yield


app = FastAPI(
    title="Predictive Maintenance Copilot",
    description=(
        "Detects equipment anomalies from sensor windows and returns "
        "an LLM-generated, RAG-grounded operator alert."
    ),
    version=__version__,
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    llm_provider: str
    embedding_model: str
    anomaly_threshold: float


class PredictRequest(BaseModel):
    sensor_window: list[list[float]] = Field(
        description="2D array shaped (window_size, n_features) of standardized sensor values."
    )
    feature_names: list[str] = Field(description="Names of the input features, in column order.")
    threshold: float | None = Field(
        default=None,
        description="Override anomaly threshold. Defaults to checkpoint value.",
    )


class PredictResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    alert: AnomalyAlert | None = None


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(
        version=__version__,
        llm_provider=settings.llm_provider,
        embedding_model=settings.embedding_model,
        anomaly_threshold=_get_threshold(),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Score a sensor window and return an LLM explanation if anomalous."""
    from src.pipeline import explain_anomaly

    window = np.asarray(request.sensor_window, dtype=np.float32)
    if window.ndim != 2:
        raise HTTPException(status_code=400, detail="sensor_window must be a 2D array")
    if window.shape[1] != len(request.feature_names):
        raise HTTPException(
            status_code=400,
            detail=(
                f"feature_names length ({len(request.feature_names)}) "
                f"does not match sensor_window columns ({window.shape[1]})"
            ),
        )

    threshold = request.threshold or _get_threshold()

    try:
        score, alert = explain_anomaly(
            window,
            request.feature_names,
            threshold=threshold,
            checkpoint_path=str(settings.checkpoint_path),
        )
    except Exception as e:
        logger.error("Pipeline error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

    return PredictResponse(
        is_anomaly=alert is not None,
        anomaly_score=round(score, 6),
        threshold=round(threshold, 6),
        alert=alert,
    )


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "Predictive Maintenance Copilot",
        "docs": "/docs",
        "health": "/healthz",
    }
