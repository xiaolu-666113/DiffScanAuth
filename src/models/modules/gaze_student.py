"""Causal gaze policy student with transformer-first and GRU fallback."""

from __future__ import annotations

import torch
import torch.nn as nn


class _TokenEncoder(nn.Module):
    """Embed scanpath tokens into a shared latent space."""

    def __init__(self, num_patches: int, embed_dim: int) -> None:
        super().__init__()
        self.start_idx = num_patches
        self.patch_emb = nn.Embedding(num_patches + 1, embed_dim)
        self.coord_proj = nn.Linear(2, embed_dim)
        self.delta_proj = nn.Linear(2, embed_dim)
        self.dur_proj = nn.Linear(1, embed_dim)

    def forward(
        self,
        patch_idx: torch.Tensor,
        coords: torch.Tensor,
        delta: torch.Tensor,
        durations: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.patch_emb(patch_idx)
            + self.coord_proj(coords)
            + self.delta_proj(delta)
            + self.dur_proj(durations.unsqueeze(-1))
        )


class TransformerGazeStudent(nn.Module):
    """Autoregressive gaze policy based on a causal transformer encoder."""

    def __init__(
        self,
        global_dim: int,
        hidden_dim: int,
        num_patches: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.token_encoder = _TokenEncoder(num_patches=num_patches, embed_dim=hidden_dim)
        self.global_proj = nn.Linear(global_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.patch_head = nn.Linear(hidden_dim, num_patches)
        self.coord_head = nn.Linear(hidden_dim, 2)
        self.delta_head = nn.Linear(hidden_dim, 2)
        self.dur_head = nn.Linear(hidden_dim, 1)
        self.stop_head = nn.Linear(hidden_dim, 1)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        global_context: torch.Tensor,
        prev_patch_idx: torch.Tensor,
        prev_coords: torch.Tensor,
        prev_delta: torch.Tensor,
        prev_durations: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tokens = self.token_encoder(prev_patch_idx, prev_coords, prev_delta, prev_durations)
        tokens = tokens + self.global_proj(global_context).unsqueeze(1)
        hidden = self.encoder(tokens, mask=self._causal_mask(tokens.size(1), tokens.device))
        coord_raw = self.coord_head(hidden)
        return {
            "hidden": hidden,
            "patch_logits": self.patch_head(hidden),
            "coord_pred": torch.sigmoid(coord_raw),
            "coord_residual": torch.tanh(coord_raw),
            "delta_pred": torch.tanh(self.delta_head(hidden)),
            "dur_pred": self.dur_head(hidden).squeeze(-1),
            "stop_logits": self.stop_head(hidden).squeeze(-1),
        }


class GRUGazeStudent(nn.Module):
    """GRU-based gaze policy fallback."""

    def __init__(
        self,
        global_dim: int,
        hidden_dim: int,
        num_patches: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.token_encoder = _TokenEncoder(num_patches=num_patches, embed_dim=hidden_dim)
        self.global_proj = nn.Linear(global_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.patch_head = nn.Linear(hidden_dim, num_patches)
        self.coord_head = nn.Linear(hidden_dim, 2)
        self.delta_head = nn.Linear(hidden_dim, 2)
        self.dur_head = nn.Linear(hidden_dim, 1)
        self.stop_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        global_context: torch.Tensor,
        prev_patch_idx: torch.Tensor,
        prev_coords: torch.Tensor,
        prev_delta: torch.Tensor,
        prev_durations: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tokens = self.token_encoder(prev_patch_idx, prev_coords, prev_delta, prev_durations)
        tokens = tokens + self.global_proj(global_context).unsqueeze(1)
        hidden, _ = self.gru(tokens)
        coord_raw = self.coord_head(hidden)
        return {
            "hidden": hidden,
            "patch_logits": self.patch_head(hidden),
            "coord_pred": torch.sigmoid(coord_raw),
            "coord_residual": torch.tanh(coord_raw),
            "delta_pred": torch.tanh(self.delta_head(hidden)),
            "dur_pred": self.dur_head(hidden).squeeze(-1),
            "stop_logits": self.stop_head(hidden).squeeze(-1),
        }


class GazeStudent(nn.Module):
    """Wrapper that selects transformer or GRU policy by config."""

    def __init__(
        self,
        global_dim: int,
        hidden_dim: int,
        num_patches: int,
        policy_type: str = "transformer",
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        policy_type = policy_type.lower()
        if policy_type == "gru":
            self.inner = GRUGazeStudent(
                global_dim=global_dim,
                hidden_dim=hidden_dim,
                num_patches=num_patches,
                num_layers=max(1, num_layers),
                dropout=dropout,
            )
        else:
            self.inner = TransformerGazeStudent(
                global_dim=global_dim,
                hidden_dim=hidden_dim,
                num_patches=num_patches,
                num_layers=max(1, num_layers),
                num_heads=max(1, num_heads),
                dropout=dropout,
            )

    def forward(
        self,
        global_context: torch.Tensor,
        prev_patch_idx: torch.Tensor,
        prev_coords: torch.Tensor,
        prev_delta: torch.Tensor,
        prev_durations: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.inner(global_context, prev_patch_idx, prev_coords, prev_delta, prev_durations)
