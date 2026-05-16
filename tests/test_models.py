"""Smoke tests for the LSTM autoencoder."""

import torch

from src.models.autoencoder import LSTMAutoencoder, reconstruction_error


def test_autoencoder_forward_preserves_shape():
    batch, window, features = 4, 30, 14
    model = LSTMAutoencoder(n_features=features, window_size=window, hidden_dim=16, latent_dim=8)
    x = torch.randn(batch, window, features)
    y = model(x)
    assert y.shape == x.shape


def test_reconstruction_error_returns_per_window_scalar():
    batch, window, features = 4, 30, 14
    model = LSTMAutoencoder(n_features=features, window_size=window, hidden_dim=16, latent_dim=8)
    x = torch.randn(batch, window, features)
    err = reconstruction_error(model, x)
    assert err.shape == (batch,)
    assert err.dtype == torch.float32
