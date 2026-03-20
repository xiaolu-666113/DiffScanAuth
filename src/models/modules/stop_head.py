"""Stop-decision head for sequential evidence models."""

from __future__ import annotations

import torch
import torch.nn as nn


class StopHead(nn.Module):
    """Predict stop probability from sequential hidden states."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states).squeeze(-1)
