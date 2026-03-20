"""Foveated evidence reader combining center/periphery samples from dual streams."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FoveatedEvidenceReader(nn.Module):
    """Differentiable foveated reader over global and local feature maps."""

    def __init__(
        self,
        global_dim: int,
        local_dim: int,
        out_dim: int = 256,
        peripheral_radius: float = 0.08,
    ) -> None:
        super().__init__()
        self.peripheral_radius = peripheral_radius
        self.global_proj = nn.Linear(global_dim * 5, out_dim)
        self.local_proj = nn.Linear(local_dim * 5, out_dim)
        self.fuse = nn.Sequential(
            nn.Linear(out_dim * 2 + 5, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def _make_offsets(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        r = self.peripheral_radius
        return torch.tensor(
            [[0.0, 0.0], [r, 0.0], [-r, 0.0], [0.0, r], [0.0, -r]],
            device=device,
            dtype=dtype,
        )

    def _sample_multi(self, fmap: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = fmap.shape
        _, t, _ = coords.shape
        offsets = self._make_offsets(coords.device, coords.dtype).view(1, 1, 5, 2)
        sample_coords = (coords.unsqueeze(2) + offsets).clamp(0.0, 1.0)
        grid = sample_coords * 2.0 - 1.0
        fmap_bt = fmap.unsqueeze(1).expand(-1, t, -1, -1, -1).reshape(b * t, c, fmap.shape[2], fmap.shape[3])
        grid = grid.reshape(b * t, 5, 1, 2)
        sampled = F.grid_sample(fmap_bt, grid, mode="bilinear", align_corners=True)
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, t, c * 5)
        return sampled

    def forward(
        self,
        global_map: torch.Tensor,
        local_map: torch.Tensor,
        coords: torch.Tensor,
        delta: torch.Tensor,
        durations: torch.Tensor,
        global_context: torch.Tensor,
    ) -> torch.Tensor:
        global_local = self._sample_multi(global_map, coords)
        local_local = self._sample_multi(local_map, coords)
        g = self.global_proj(global_local)
        l = self.local_proj(local_local)
        del global_context
        fuse_in = torch.cat([g, l, coords, delta, durations.unsqueeze(-1)], dim=-1)
        return self.fuse(fuse_in)
