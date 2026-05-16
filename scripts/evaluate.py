"""Evaluate anomaly detection performance on the C-MAPSS test set.

Per-unit anomaly score = mean reconstruction error over all sliding windows.
Ground truth: is_anomaly = RUL <= anomaly_horizon (default 30 cycles, per
Saxena et al. convention). All metrics are logged to MLflow.

Three threshold strategies are reported:
  - P95 (from checkpoint, val-calibrated)
  - F1-optimal (via sklearn precision_recall_curve)
  - Manual (--threshold override, optional)

Threshold-independent metrics (PR-AUC, ROC-AUC) are the primary summary.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.data.loaders import load_cmapss, make_sliding_windows
from src.data.preprocessing import apply_scaler, fit_scaler
from src.models.autoencoder import LSTMAutoencoder, reconstruction_error

app = typer.Typer(help="Evaluate anomaly detection on the C-MAPSS test set.")
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("scripts.evaluate")


def _threshold_metrics(scores: np.ndarray, labels: np.ndarray, t: float) -> dict[str, float]:
    preds = (scores > t).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


@app.command()
def main(
    subset: str = typer.Option("FD001", "--subset"),
    checkpoint: Path = typer.Option(None, "--checkpoint"),
    anomaly_horizon: int = typer.Option(
        30,
        "--anomaly-horizon",
        help=(
            "Units with RUL <= this at their last cycle are labeled anomalous. "
            "30 = imminent failure (Saxena et al.), 50 = onset of degradation."
        ),
    ),
    threshold: float = typer.Option(
        0.0,
        "--threshold",
        help="Manual threshold override. 0.0 = omit manual row from table.",
    ),
    save_pr_curve: bool = typer.Option(True, "--save-pr-curve/--no-pr-curve"),
) -> None:
    import mlflow
    import matplotlib.pyplot as plt

    from src.config import settings

    ckpt_path = checkpoint or Path("checkpoints") / f"autoencoder_{subset}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found — run `make train` first.")

    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    feature_columns: list[str] = ckpt["feature_columns"]
    window_size: int = ckpt["window_size"]
    p95_threshold: float = ckpt.get("anomaly_threshold", 0.0)

    logger.info("Loading C-MAPSS %s", subset)
    ds = load_cmapss(subset=subset)

    scaler = fit_scaler(ds.train, feature_columns)
    test_scaled = apply_scaler(ds.test, feature_columns, scaler)

    model = LSTMAutoencoder(
        n_features=len(feature_columns),
        window_size=window_size,
        hidden_dim=ckpt["hidden_dim"],
        latent_dim=ckpt["latent_dim"],
        num_layers=ckpt.get("num_layers", 1),
    )
    model.load_state_dict(ckpt["model_state"])
    logger.info("Model loaded (%d parameters)", sum(p.numel() for p in model.parameters()))

    X_test, unit_ids = make_sliding_windows(test_scaled, feature_columns, window_size=window_size)
    logger.info("Test windows: %s", X_test.shape)

    with torch.no_grad():
        errors = reconstruction_error(model, torch.from_numpy(X_test)).numpy()

    unit_scores: dict[int, float] = {
        int(uid): float(errors[unit_ids == uid].mean())
        for uid in np.unique(unit_ids)
    }

    units_sorted = sorted(unit_scores)
    scores = np.array([unit_scores[u] for u in units_sorted])
    ruls = ds.rul_test.loc[units_sorted].to_numpy(dtype=float)
    labels = (ruls <= anomaly_horizon).astype(int)

    logger.info(
        "Anomaly horizon=%d — %d anomalous / %d healthy units",
        anomaly_horizon, int(labels.sum()), int((1 - labels).sum()),
    )

    # Threshold-independent metrics.
    pr_auc = float(average_precision_score(labels, scores))
    roc_auc = float(roc_auc_score(labels, scores))
    correlation = float(np.corrcoef(scores, ruls)[0, 1])

    # F1-optimal threshold via sklearn (O(N log N), no manual loop).
    precisions, recalls, thresholds_pr = precision_recall_curve(labels, scores)
    # precisions/recalls have N+1 elements; thresholds_pr has N.
    f1_arr = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0.0,
    )
    best_idx = int(np.argmax(f1_arr))
    f1_opt_threshold = float(thresholds_pr[best_idx])

    # PR curve plot.
    if save_pr_curve:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(recalls, precisions, color="steelblue",
                label=f"PR curve  AUC={pr_auc:.3f}")

        if p95_threshold > 0.0:
            m = _threshold_metrics(scores, labels, p95_threshold)
            ax.scatter(m["recall"], m["precision"], marker="o", color="crimson", zorder=5,
                       label=f"P95  F1={m['f1']:.3f}")

        m_opt = _threshold_metrics(scores, labels, f1_opt_threshold)
        ax.scatter(m_opt["recall"], m_opt["precision"], marker="*", s=180,
                   color="seagreen", zorder=5, label=f"F1-opt  F1={m_opt['f1']:.3f}")

        if threshold != 0.0:
            m_man = _threshold_metrics(scores, labels, threshold)
            ax.scatter(m_man["recall"], m_man["precision"], marker="^", color="darkorange",
                       zorder=5, label=f"Manual  F1={m_man['f1']:.3f}")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve — {subset}  (anomaly_horizon={anomaly_horizon})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        pr_path = Path("checkpoints") / f"pr_curve_{subset}_h{anomaly_horizon}.png"
        fig.savefig(pr_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("PR curve saved: %s", pr_path)

    # Log to MLflow.
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name=f"eval_{subset}_h{anomaly_horizon}"):
        mlflow.log_params({"subset": subset, "anomaly_horizon": anomaly_horizon})
        mlflow.log_metrics({
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "score_rul_correlation": correlation,
            "f1_optimal": float(f1_arr[best_idx]),
            "f1_opt_threshold": f1_opt_threshold,
        })
        if p95_threshold > 0.0:
            m_p95 = _threshold_metrics(scores, labels, p95_threshold)
            mlflow.log_metrics({
                "p95_precision": m_p95["precision"],
                "p95_recall": m_p95["recall"],
                "p95_f1": m_p95["f1"],
            })
        if save_pr_curve:
            mlflow.log_artifact(str(pr_path))

    # Persist f1-optimal threshold back into checkpoint.
    ckpt["anomaly_threshold_f1"] = f1_opt_threshold
    torch.save(ckpt, ckpt_path)
    logger.info("Checkpoint updated: anomaly_threshold_f1=%.6f", f1_opt_threshold)

    # Results table.
    tbl = Table(
        title=f"Anomaly Detection — {subset}  anomaly_horizon={anomaly_horizon}",
        show_lines=True,
    )
    tbl.add_column("Threshold strategy", style="cyan", no_wrap=True)
    tbl.add_column("Value", style="white")
    tbl.add_column("Precision", style="magenta")
    tbl.add_column("Recall", style="magenta")
    tbl.add_column("F1", style="green")
    tbl.add_column("TP/FP/FN/TN", style="dim")

    def _row(strategy: str, t: float, m: dict[str, float]) -> None:
        tbl.add_row(
            strategy, f"{t:.6f}",
            f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}",
            f"{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}",
        )

    if p95_threshold > 0.0:
        _row("P95 (checkpoint)", p95_threshold, _threshold_metrics(scores, labels, p95_threshold))
    _row("F1-optimal", f1_opt_threshold, _threshold_metrics(scores, labels, f1_opt_threshold))
    if threshold != 0.0:
        _row("Manual (--threshold)", threshold, _threshold_metrics(scores, labels, threshold))

    console.print(tbl)

    summary = Table(title="Threshold-independent metrics", show_lines=True)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="magenta")
    summary.add_row("PR-AUC", f"{pr_auc:.3f}")
    summary.add_row("ROC-AUC", f"{roc_auc:.3f}")
    summary.add_row("Score-RUL correlation", f"{correlation:.3f}  (expected < 0)")
    console.print(summary)


if __name__ == "__main__":
    app()
