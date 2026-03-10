"""Evidence accumulation modules (GRU default, mamba optional fallback)."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba  # type: ignore
except Exception:  # pragma: no cover
    Mamba = None


class GRUAccumulator(nn.Module):
    """GRU accumulator for sequential evidence integration."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, seq: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        states, _ = self.gru(seq)
        if mask is None:
            final = states[:, -1, :]
        else:
            lengths = mask.sum(dim=1).long().clamp(min=1) - 1
            final = states[torch.arange(states.size(0), device=states.device), lengths]
        return states, final


class MambaAccumulator(nn.Module):
    """Optional mamba accumulator with graceful fallback behavior."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if Mamba is None:
            raise RuntimeError("mamba-ssm is unavailable")
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.block = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.in_proj(seq)
        states = self.block(x)
        if mask is None:
            final = states[:, -1, :]
        else:
            lengths = mask.sum(dim=1).long().clamp(min=1) - 1
            final = states[torch.arange(states.size(0), device=states.device), lengths]
        return states, final


class SelectiveAccumulator(nn.Module):
    """Factory wrapper: choose GRU or mamba accumulator."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        backend: str = "gru",
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        backend = backend.lower()
        if backend == "mamba":
            if Mamba is not None:
                self.inner = MambaAccumulator(input_dim=input_dim, hidden_dim=hidden_dim)
            else:
                warnings.warn("mamba backend requested but unavailable; fallback to GRU")
                self.inner = GRUAccumulator(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        else:
            self.inner = GRUAccumulator(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inner(seq, mask)
