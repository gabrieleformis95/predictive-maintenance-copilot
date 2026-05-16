"""Shared fixtures available to all test modules."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def fake_checkpoint(tmp_path_factory):
    """Minimal LSTMAutoencoder checkpoint usable across the test session."""
    from src.models.autoencoder import LSTMAutoencoder

    tmp = tmp_path_factory.mktemp("checkpoints")
    feature_columns = ["s1", "s2", "s3"]
    model = LSTMAutoencoder(n_features=3, window_size=30, hidden_dim=16, latent_dim=8)
    ckpt = {
        "model_state": model.state_dict(),
        "feature_columns": feature_columns,
        "window_size": 30,
        "hidden_dim": 16,
        "latent_dim": 8,
        "num_layers": 1,
        "anomaly_threshold_f1": 0.33,
        "anomaly_threshold": 0.33,
    }
    path = tmp / "test_autoencoder.pt"
    torch.save(ckpt, path)
    return str(path)


@pytest.fixture
def cmapss_dir(tmp_path):
    """Directory with minimal valid C-MAPSS .txt files (FD001)."""
    from src.data.loaders import COLUMN_NAMES

    rng = np.random.default_rng(42)
    data_dir = tmp_path / "CMAPSSData"
    data_dir.mkdir()

    import pandas as pd

    def _make_df(n_units: int, cycles: int) -> pd.DataFrame:
        rows = []
        for u in range(1, n_units + 1):
            for c in range(1, cycles + 1):
                rows.append([u, c] + rng.normal(size=24).tolist())
        return pd.DataFrame(rows, columns=COLUMN_NAMES)

    _make_df(3, 50).to_csv(data_dir / "train_FD001.txt", sep=" ", header=False, index=False)
    _make_df(3, 20).to_csv(data_dir / "test_FD001.txt", sep=" ", header=False, index=False)
    np.savetxt(data_dir / "RUL_FD001.txt", rng.integers(50, 150, size=3).astype(float))

    return data_dir
