"""Drift monitoring with Evidently.

Generates two HTML reports saved to reports/:
  - sensor_drift.html   : feature distributions (train healthy vs test)
  - score_drift.html    : anomaly score distributions (high-RUL vs low-RUL engines)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from rich.logging import RichHandler

from src.config import settings
from src.data.loaders import load_cmapss, make_sliding_windows
from src.data.preprocessing import apply_scaler, fit_scaler
from src.models.autoencoder import LSTMAutoencoder, reconstruction_error

app = typer.Typer(help="Run Evidently drift monitoring.")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("scripts.monitor")


def _load_model(device: torch.device) -> tuple[LSTMAutoencoder, list[str], int, float]:
    ckpt = torch.load(str(settings.checkpoint_path), map_location=device, weights_only=True)
    model = LSTMAutoencoder(
        n_features=len(ckpt["feature_columns"]),
        window_size=ckpt["window_size"],
        hidden_dim=ckpt["hidden_dim"],
        latent_dim=ckpt["latent_dim"],
        num_layers=ckpt.get("num_layers", 1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    threshold = float(ckpt.get("anomaly_threshold_f1") or ckpt.get("anomaly_threshold") or 0.33)
    return model, ckpt["feature_columns"], ckpt["window_size"], threshold


@app.command()
def main(
    subset: str = typer.Option("FD001", "--subset"),
    healthy_early_cycles: int = typer.Option(50, "--healthy-early-cycles"),
    out_dir: Path = typer.Option(Path("reports"), "--out-dir"),
) -> None:
    from evidently import Report
    from evidently.metrics import ValueDrift
    from evidently.presets import DataDriftPreset

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    logger.info("Loading dataset and model...")
    ds = load_cmapss(subset=subset)  # type: ignore[arg-type]
    feats = ds.feature_columns
    scaler = fit_scaler(ds.train, feats)

    # -----------------------------------------------------------------------
    # Report 1: Sensor drift — train healthy vs test
    # -----------------------------------------------------------------------
    logger.info("Building sensor drift report...")

    train_healthy = ds.train[ds.train["cycle"] <= healthy_early_cycles].copy()
    train_scaled = apply_scaler(train_healthy, feats, scaler)
    test_scaled = apply_scaler(ds.test.copy(), feats, scaler)

    ref_sensors = train_scaled[feats].reset_index(drop=True)
    cur_sensors = test_scaled[feats].reset_index(drop=True)

    sensor_snap = Report([DataDriftPreset()]).run(
        reference_data=ref_sensors, current_data=cur_sensors
    )
    sensor_path = out_dir / "sensor_drift.html"
    sensor_snap.save_html(str(sensor_path))
    logger.info("Saved: %s", sensor_path)

    # -----------------------------------------------------------------------
    # Report 2: Anomaly score drift — high-RUL engines vs low-RUL engines
    # -----------------------------------------------------------------------
    logger.info("Computing anomaly scores per test engine...")

    model, feature_cols, window_size, threshold = _load_model(device)
    all_units = sorted(ds.test["unit"].unique())
    mid = len(all_units) // 2
    high_rul_units = all_units[:mid]   # engines with more RUL remaining
    low_rul_units = all_units[mid:]    # engines closer to failure

    def _scores_for_units(units: list) -> np.ndarray:
        scores: list[float] = []
        test_s = apply_scaler(ds.test.copy(), feature_cols, scaler)
        for uid in units:
            udf = test_s[test_s["unit"] == uid]
            vals = udf[feature_cols].values.astype(np.float32)
            if len(vals) < window_size:
                continue
            x = np.stack([vals[i : i + window_size] for i in range(len(vals) - window_size + 1)])
            errs = reconstruction_error(model, torch.from_numpy(x)).numpy()
            scores.extend(errs.tolist())
        return np.array(scores)

    high_scores = _scores_for_units(high_rul_units)
    low_scores = _scores_for_units(low_rul_units)

    n = min(len(high_scores), len(low_scores))
    score_ref = pd.DataFrame({"anomaly_score": high_scores[:n]})
    score_cur = pd.DataFrame({"anomaly_score": low_scores[:n]})

    score_snap = Report([ValueDrift(column="anomaly_score")]).run(
        reference_data=score_ref, current_data=score_cur
    )
    score_path = out_dir / "score_drift.html"
    score_snap.save_html(str(score_path))
    logger.info("Saved: %s", score_path)

    logger.info(
        "Done. Threshold=%.4f | High-RUL mean score=%.4f | Low-RUL mean score=%.4f",
        threshold,
        float(high_scores.mean()),
        float(low_scores.mean()),
    )


if __name__ == "__main__":
    app()
