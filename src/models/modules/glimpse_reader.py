"""Foveated glimpse reader based on differentiable bilinear sampling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlimpseReader(nn.Module):
    """Read local evidence from dense feature map around fixation coordinates."""

    def __init__(self, in_channels: int, global_dim: int, out_dim: int = 256) -> None:
        super().__init__()
        self.local_proj = nn.Linear(in_channels, out_dim)
        self.global_proj = nn.Linear(global_dim, out_dim)
        self.fuse = nn.Sequential(
            nn.Linear(out_dim * 2 + 3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def _sample(self, fmap: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Sample C-dim local features from fmap using normalized coords in [0,1]."""
        b, c, _, _ = fmap.shape
        _, t, _ = coords.shape

        grid = coords * 2.0 - 1.0
        grid = grid.view(b * t, 1, 1, 2)

        fmap_bt = fmap.unsqueeze(1).expand(-1, t, -1, -1, -1).reshape(b * t, c, fmap.shape[2], fmap.shape[3])
        sampled = F.grid_sample(fmap_bt, grid, mode="bilinear", align_corners=True)
        sampled = sampled.view(b, t, c)
        return sampled

    def forward(
        self,
        fmap: torch.Tensor,
        coords: torch.Tensor,
        global_feat: torch.Tensor,
        durations: torch.Tensor,
    ) -> torch.Tensor:
        local = self._sample(fmap, coords)
        local = self.local_proj(local)
        g = self.global_proj(global_feat).unsqueeze(1).expand_as(local)

        fused = torch.cat([local, g, coords, durations.unsqueeze(-1)], dim=-1)
        return self.fuse(fused)
