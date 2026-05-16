"""Train the LSTM autoencoder on C-MAPSS healthy operation windows.

Logs hyperparameters, metrics, and the resulting model to MLflow so each
training run is fully reproducible.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import typer
from rich.logging import RichHandler
from torch.utils.data import DataLoader, TensorDataset

from src.config import settings
from src.data.loaders import load_cmapss, make_sliding_windows
from src.data.preprocessing import apply_scaler, fit_scaler
from src.models.autoencoder import LSTMAutoencoder

app = typer.Typer(help="Train the LSTM autoencoder.")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("scripts.train")


@app.command()
def main(
    subset: str = typer.Option("FD001", "--subset"),
    epochs: int = typer.Option(50, "--epochs"),
    batch_size: int = typer.Option(128, "--batch-size"),
    hidden_dim: int = typer.Option(64, "--hidden-dim"),
    latent_dim: int = typer.Option(16, "--latent-dim"),
    lr: float = typer.Option(1e-3, "--lr"),
    healthy_early_cycles: int = typer.Option(
        50,
        "--healthy-early-cycles",
        help=(
            "Train the autoencoder only on the first N cycles of each engine, "
            "treating these as 'healthy' baseline operation."
        ),
    ),
    val_fraction: float = typer.Option(
        0.2,
        "--val-fraction",
        help="Fraction of training engines held out to calibrate the anomaly threshold.",
    ),
) -> None:
    """Train and log to MLflow."""
    import mlflow

    logger.info("Loading C-MAPSS %s", subset)
    ds = load_cmapss(subset=subset)  # type: ignore[arg-type]
    feats = ds.feature_columns

    # Split engines into train / validation before scaling.
    all_units = sorted(ds.train["unit"].unique())
    n_val = max(1, int(len(all_units) * val_fraction))
    train_units = all_units[:-n_val]
    val_units = all_units[-n_val:]
    logger.info("Engines — train: %d, val: %d", len(train_units), len(val_units))

    train_df = ds.train[ds.train["unit"].isin(train_units)]
    val_df = ds.train[ds.train["unit"].isin(val_units)]

    # Fit scaler on train engines only.
    scaler = fit_scaler(train_df, feats)
    train_scaled = apply_scaler(train_df, feats, scaler)
    val_scaled = apply_scaler(val_df, feats, scaler)

    # Keep only the first `healthy_early_cycles` cycles per engine.
    healthy = train_scaled[train_scaled["cycle"] <= healthy_early_cycles]
    logger.info("Healthy rows kept: %d / %d", len(healthy), len(train_scaled))

    X, _ = make_sliding_windows(healthy, feats, window_size=settings.window_size)
    logger.info("Windowed training tensor: %s", X.shape)

    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = LSTMAutoencoder(
        n_features=len(feats),
        window_size=settings.window_size,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "subset": subset,
                "epochs": epochs,
                "batch_size": batch_size,
                "hidden_dim": hidden_dim,
                "latent_dim": latent_dim,
                "lr": lr,
                "window_size": settings.window_size,
                "healthy_early_cycles": healthy_early_cycles,
                "val_fraction": val_fraction,
                "n_features": len(feats),
            }
        )

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for (batch,) in loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = loss_fn(recon, batch)
                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg = epoch_loss / max(n_batches, 1)
            mlflow.log_metric("train_mse", avg, step=epoch)
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                logger.info("Epoch %d/%d — MSE %.6f", epoch, epochs, avg)

        # Calibrate threshold on validation engines' healthy windows.
        val_healthy = val_scaled[val_scaled["cycle"] <= healthy_early_cycles]
        X_val, _ = make_sliding_windows(val_healthy, feats, window_size=settings.window_size)
        model.eval()
        threshold = 0.0
        if X_val.shape[0] > 0:
            with torch.no_grad():
                val_tensor = torch.from_numpy(X_val).to(device)
                val_errors = ((model(val_tensor) - val_tensor) ** 2).mean(dim=(1, 2)).cpu().numpy()
            threshold = float(np.percentile(val_errors, settings.anomaly_threshold_percentile))
            mlflow.log_metric("anomaly_threshold", threshold)
            logger.info(
                "Anomaly threshold (P%.0f, val): %.6f",
                settings.anomaly_threshold_percentile,
                threshold,
            )
        else:
            logger.warning("No validation windows — threshold set to 0.")

        # Save artifacts.
        ckpt_path = Path("checkpoints") / f"autoencoder_{subset}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "feature_columns": feats,
                "subset": subset,
                "window_size": settings.window_size,
                "hidden_dim": hidden_dim,
                "latent_dim": latent_dim,
                "num_layers": 1,
                "anomaly_threshold": threshold,
            },
            ckpt_path,
        )
        mlflow.log_artifact(str(ckpt_path))


if __name__ == "__main__":
    app()
