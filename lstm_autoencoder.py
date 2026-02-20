# lstm_autoencoder.py
from __future__ import annotations

import os
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMAutoencoder(nn.Module):
    """
    Simple LSTM autoencoder for (N, seq_len, n_features) float32 sequences.

    Public API used by ml_pipeline.py:
      - fit(X, epochs=..., batch_size=..., lr=...)
      - reconstruction_error(X) -> np.ndarray shape (N,)
      - save(path), load(path)
    """

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.n_features = int(n_features)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        self.encoder = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.decoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.output_layer = nn.Linear(self.hidden_size, self.n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        enc_out, (h_n, c_n) = self.encoder(x)  # enc_out: (B, T, H)

        # Decode by feeding zeros with the final hidden state as initial state
        B, T, H = enc_out.shape
        dec_in = torch.zeros((B, T, H), device=x.device, dtype=x.dtype)
        dec_out, _ = self.decoder(dec_in, (h_n, c_n))  # (B, T, H)

        y = self.output_layer(dec_out)  # (B, T, F)
        return y

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 12,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ) -> None:
        if X.ndim != 3:
            raise ValueError(f"Expected X shape (N, T, F). Got {X.shape}")

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)
        self.train()

        X_t = torch.as_tensor(X, dtype=torch.float32, device=dev)
        ds = TensorDataset(X_t)
        dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=False)

        optimizer = torch.optim.Adam(self.parameters(), lr=float(lr))
        loss_fn = nn.MSELoss(reduction="mean")  # IMPORTANT: callable, do not shadow

        for _ in range(int(epochs)):
            for (batch,) in dl:
                optimizer.zero_grad(set_to_none=True)
                recon = self.forward(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def reconstruction_error(self, X: np.ndarray, device: Optional[str] = None) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"Expected X shape (N, T, F). Got {X.shape}")

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)
        self.eval()

        X_t = torch.as_tensor(X, dtype=torch.float32, device=dev)
        dl = DataLoader(TensorDataset(X_t), batch_size=256, shuffle=False, drop_last=False)

        errs = []
        for (batch,) in dl:
            recon = self.forward(batch)
            # per-sequence mean squared error over (T,F)
            e = torch.mean((recon - batch) ** 2, dim=(1, 2))
            errs.append(e.detach().cpu().numpy())

        return np.concatenate(errs, axis=0) if errs else np.zeros((0,), dtype=np.float32)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "n_features": self.n_features,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
            },
            path,
        )

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        ckpt = torch.load(path, map_location=map_location or "cpu")
        # NOTE: we assume constructor matches these; if not, rebuild the object.
        self.load_state_dict(ckpt["state_dict"], strict=True)