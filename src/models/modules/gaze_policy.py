"""Causal gaze policy networks."""

from __future__ import annotations

import torch
import torch.nn as nn


class GazePolicyGRU(nn.Module):
    """Teacher-forced causal policy for next fixation prediction."""

    def __init__(
        self,
        global_dim: int,
        hidden_dim: int,
        num_patches: int,
        patch_emb_dim: int = 128,
        global_proj_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.start_idx = num_patches  # extra token id

        self.patch_emb = nn.Embedding(num_patches + 1, patch_emb_dim)
        self.global_proj = nn.Linear(global_dim, global_proj_dim)
        in_dim = patch_emb_dim + 2 + 1 + global_proj_dim

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.patch_head = nn.Linear(hidden_dim, num_patches)
        self.coord_head = nn.Linear(hidden_dim, 2)
        self.dur_head = nn.Linear(hidden_dim, 1)
        self.stop_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        global_feat: torch.Tensor,
        teacher_patch_idx: torch.Tensor,
        teacher_xy: torch.Tensor,
        teacher_dur: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict patch / coord / duration / stop at each time step."""
        b, t = teacher_patch_idx.shape
        device = teacher_patch_idx.device

        start = torch.full((b, 1), self.start_idx, device=device, dtype=torch.long)
        prev_patch = torch.cat([start, teacher_patch_idx[:, :-1]], dim=1)

        zero_xy = torch.zeros((b, 1, 2), device=device, dtype=teacher_xy.dtype)
        prev_xy = torch.cat([zero_xy, teacher_xy[:, :-1, :]], dim=1)

        zero_dur = torch.zeros((b, 1), device=device, dtype=teacher_dur.dtype)
        prev_dur = torch.cat([zero_dur, teacher_dur[:, :-1]], dim=1)

        patch_emb = self.patch_emb(prev_patch)
        g = self.global_proj(global_feat).unsqueeze(1).expand(-1, t, -1)

        x = torch.cat([patch_emb, prev_xy, prev_dur.unsqueeze(-1), g], dim=-1)
        h, _ = self.gru(x)

        patch_logits = self.patch_head(h)
        coord_pred = torch.sigmoid(self.coord_head(h))
        dur_pred = self.dur_head(h).squeeze(-1)
        stop_logits = self.stop_head(h).squeeze(-1)

        return {
            "hidden": h,
            "patch_logits": patch_logits,
            "coord_pred": coord_pred,
            "dur_pred": dur_pred,
            "stop_logits": stop_logits,
        }
