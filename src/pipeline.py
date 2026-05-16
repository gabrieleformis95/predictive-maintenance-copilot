"""End-to-end orchestration: sensor window → anomaly score → RAG → LLM alert.

LLM routing:
  1. Ollama (local) — primary
  2. Groq (cloud, free tier) — automatic fallback when Ollama is unreachable
  3. OpenAI — explicit via llm_provider="openai" in config / .env
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from src.rag.prompts import SYSTEM_PROMPT, AnomalyAlert, build_user_prompt
from src.rag.retriever import retrieve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading (cached per checkpoint path)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _load_model(checkpoint_path: str):
    from src.models.autoencoder import LSTMAutoencoder

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = LSTMAutoencoder(
        n_features=len(ckpt["feature_columns"]),
        window_size=ckpt["window_size"],
        hidden_dim=ckpt["hidden_dim"],
        latent_dim=ckpt["latent_dim"],
        num_layers=ckpt.get("num_layers", 1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def _default_checkpoint(subset: str = "FD001") -> str:
    return str(Path("checkpoints") / f"autoencoder_{subset}.pt")


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------


def _sensor_deviations(
    window: np.ndarray,
    feature_names: list[str],
) -> list[tuple[str, float]]:
    """Return per-sensor mean z-score over the window, sorted by magnitude.

    The window is already standardized (mean≈0, std≈1 over the training
    distribution), so the mean value across the window is itself a z-score
    relative to healthy baseline.
    """
    mean_per_sensor = window.mean(axis=0)
    return sorted(
        zip(feature_names, mean_per_sensor.tolist(), strict=False),
        key=lambda x: abs(x[1]),
        reverse=True,
    )


def _build_retrieval_query(deviations: list[tuple[str, float]], top_n: int = 3) -> str:
    top = deviations[:top_n]
    parts = [f"{name} (deviation={z:+.2f})" for name, z in top]
    return (
        "turbofan engine anomaly: "
        + ", ".join(parts)
        + " — maintenance procedure inspection recommendation"
    )


def _severity(score: float, threshold: float) -> str:
    ratio = score / threshold
    if ratio >= 1.5:
        return "critical"
    if ratio >= 1.0:
        return "warning"
    return "info"


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    import httpx

    from src.config import settings

    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
    }
    resp = httpx.post(
        f"{settings.ollama_base_url}/api/chat",
        json=payload,
        timeout=240.0,
    )
    resp.raise_for_status()
    return str(resp.json()["message"]["content"])


def _call_openai_compat(
    system_prompt: str, user_prompt: str, base_url: str, api_key: str, model: str
) -> str:
    """Generic call for any OpenAI-compatible endpoint (Groq, OpenAI)."""
    import httpx

    resp = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return str(resp.json()["choices"][0]["message"]["content"])


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Route to LLM: Ollama primary, Groq fallback, or explicit provider."""
    from src.config import settings

    provider = settings.llm_provider

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("llm_provider=openai but OPENAI_API_KEY is not set.")
        return _call_openai_compat(
            system_prompt,
            user_prompt,
            base_url="https://api.openai.com/v1",
            api_key=settings.openai_api_key,
            model=settings.openai_model if hasattr(settings, "openai_model") else "gpt-4o-mini",
        )

    if provider == "groq":
        if not settings.groq_api_key:
            raise RuntimeError("llm_provider=groq but GROQ_API_KEY is not set.")
        return _call_openai_compat(
            system_prompt,
            user_prompt,
            base_url=settings.groq_base_url,
            api_key=settings.groq_api_key,
            model=settings.groq_model,
        )

    # Default: Ollama with Groq fallback.
    try:
        return _call_ollama(system_prompt, user_prompt)
    except Exception as ollama_err:
        logger.warning("Ollama unavailable (%s) — falling back to Groq.", ollama_err)
        if settings.groq_api_key:
            return _call_openai_compat(
                system_prompt,
                user_prompt,
                base_url=settings.groq_base_url,
                api_key=settings.groq_api_key,
                model=settings.groq_model,
            )
        raise RuntimeError(
            "Ollama is unreachable and GROQ_API_KEY is not configured. "
            "Start Ollama or set groq_api_key in .env."
        ) from ollama_err


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def explain_anomaly(
    sensor_window: np.ndarray,
    feature_names: list[str],
    *,
    threshold: float,
    checkpoint_path: str | None = None,
) -> tuple[float, AnomalyAlert | None]:
    """Score a sensor window and return (score, alert).

    Parameters
    ----------
    sensor_window: shape (window_size, n_features), already standardized.
    feature_names: column names in the same order as window columns.
    threshold: reconstruction-error threshold above which a window is anomalous.
    checkpoint_path: path to .pt checkpoint. Defaults to checkpoints/autoencoder_FD001.pt.

    Returns (score, None) when the window is healthy (score < threshold).
    Returns (score, AnomalyAlert) when anomalous.
    """
    ckpt_path = checkpoint_path or _default_checkpoint()
    model, _ = _load_model(ckpt_path)

    x = torch.from_numpy(sensor_window.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        x_hat = model(x)
        score = float(((x_hat - x) ** 2).mean().item())

    if score < threshold:
        return score, None

    deviations = _sensor_deviations(sensor_window, feature_names)
    query = _build_retrieval_query(deviations)
    chunks = retrieve(query, k=5)

    severity = _severity(score, threshold)
    anomaly_description = (
        f"Reconstruction error {score:.4f} exceeds threshold {threshold:.4f} "
        f"(ratio {score / threshold:.2f}x). Severity assessed as {severity}."
    )
    sensor_dev_text = "\n".join(f"  {name}: {z:+.3f}" for name, z in deviations[:6])
    excerpts_text = (
        "\n\n".join(f"[{c.source}, page {c.page}]\n{c.text[:500]}" for c in chunks)
        if chunks
        else "No manual excerpts available."
    )

    user_prompt = build_user_prompt(anomaly_description, sensor_dev_text, excerpts_text)

    try:
        raw = _call_llm(SYSTEM_PROMPT, user_prompt)
        alert = AnomalyAlert.model_validate(json.loads(raw))
    except Exception as e:
        logger.error("LLM call or parse failed: %s — returning rule-based alert.", e)
        alert = AnomalyAlert(
            severity=severity,
            probable_cause=f"Reconstruction error {score:.4f} exceeds threshold {threshold:.4f}.",
            recommended_action="Inspect equipment manually.",
            affected_sensors=[name for name, _ in deviations[:3]],
            citations=[],
        )

    return score, alert


def train_cli() -> None:
    """Console-script entry point for training."""
    from scripts.train import main

    main()
