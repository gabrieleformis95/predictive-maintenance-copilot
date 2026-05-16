"""LSTM autoencoder for multivariate time-series anomaly detection.

The encoder compresses a window of sensor readings into a latent vector;
the decoder reconstructs the window. At inference time, a high reconstruction
error signals that the input deviates from the patterns seen during training
(which contain mostly healthy operation).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """Symmetric LSTM autoencoder.

    Parameters
    ----------
    n_features: number of input channels (sensors).
    window_size: temporal length of each input window.
    hidden_dim: hidden size of the LSTM layers.
    latent_dim: dimension of the bottleneck.
    num_layers: stacked LSTM layers in encoder/decoder.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.num_layers = num_layers
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        # Project latent to initial hidden and cell states for the decoder.
        self.latent_to_h = nn.Linear(latent_dim, num_layers * hidden_dim)
        self.latent_to_c = nn.Linear(latent_dim, num_layers * hidden_dim)

        self.decoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input window.

        x: shape (batch, window_size, n_features)
        returns: shape (batch, window_size, n_features)
        """
        # Encode: take the last hidden state as a window summary.
        _, (h_n, _) = self.encoder(x)
        latent = self.to_latent(h_n[-1])  # (batch, latent_dim)

        # Initialise decoder hidden/cell from latent; feed zeros as input.
        batch = x.size(0)
        h_0 = self.latent_to_h(latent).view(self.num_layers, batch, self.hidden_dim)
        c_0 = self.latent_to_c(latent).view(self.num_layers, batch, self.hidden_dim)
        decoder_input = torch.zeros(batch, self.window_size, self.n_features, device=x.device)
        decoded, _ = self.decoder(decoder_input, (h_0, c_0))
        return self.output_layer(decoded)  # type: ignore[no-any-return]


def reconstruction_error(
    model: LSTMAutoencoder,
    x: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute per-window reconstruction error (squared error).

    Returns a tensor of shape (batch,) when reduction='mean' or 'sum',
    or (batch, window_size, n_features) when reduction='none'.
    """
    model.eval()
    with torch.no_grad():
        x_hat = model(x)
        sq_err = (x_hat - x) ** 2
        if reduction == "mean":
            return sq_err.mean(dim=(1, 2))  # type: ignore[no-any-return]
        if reduction == "sum":
            return sq_err.sum(dim=(1, 2))  # type: ignore[no-any-return]
        return sq_err  # type: ignore[no-any-return]
