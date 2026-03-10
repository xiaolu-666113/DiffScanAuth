"""Prediction heads."""

from __future__ import annotations

import torch
import torch.nn as nn


class BinaryClassificationHead(nn.Module):
    """Simple MLP binary classification head returning logits."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
