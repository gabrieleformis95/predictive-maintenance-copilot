"""Smoke tests for the LSTM autoencoder."""

from __future__ import annotations

import torch

from src.models.autoencoder import LSTMAutoencoder, reconstruction_error


def test_autoencoder_forward_preserves_shape():
    """GIVEN a random input tensor of shape (batch=4, window=30, features=14).
    WHEN the model forward pass is run.
    THEN the output has the same shape as the input.
    """
    batch, window, features = 4, 30, 14
    model = LSTMAutoencoder(n_features=features, window_size=window, hidden_dim=16, latent_dim=8)
    x = torch.randn(batch, window, features)
    assert model(x).shape == x.shape


def test_reconstruction_error_returns_per_window_scalar():
    """GIVEN a random input tensor of shape (batch=4, window=30, features=14).
    WHEN reconstruction_error is called with reduction='mean'.
    THEN a float32 tensor of shape (batch,) is returned.
    """
    batch, window, features = 4, 30, 14
    model = LSTMAutoencoder(n_features=features, window_size=window, hidden_dim=16, latent_dim=8)
    x = torch.randn(batch, window, features)
    err = reconstruction_error(model, x)
    assert err.shape == (batch,)
    assert err.dtype == torch.float32


def test_reconstruction_error_reduction_none():
    """GIVEN a random input tensor.
    WHEN reconstruction_error is called with reduction='none'.
    THEN a tensor of shape (batch, window, features) is returned.
    """
    batch, window, features = 2, 10, 5
    model = LSTMAutoencoder(n_features=features, window_size=window, hidden_dim=8, latent_dim=4)
    x = torch.randn(batch, window, features)
    err = reconstruction_error(model, x, reduction="none")
    assert err.shape == (batch, window, features)


def test_reconstruction_error_reduction_sum():
    """GIVEN a random input tensor.
    WHEN reconstruction_error is called with reduction='sum'.
    THEN the sum result is >= the mean result for the same input.
    """
    batch, window, features = 2, 10, 5
    model = LSTMAutoencoder(n_features=features, window_size=window, hidden_dim=8, latent_dim=4)
    x = torch.randn(batch, window, features)
    err_sum = reconstruction_error(model, x, reduction="sum")
    err_mean = reconstruction_error(model, x, reduction="mean")
    assert (err_sum >= err_mean).all()
