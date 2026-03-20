"""Simplified conditional latent diffusion teacher for scanpath modeling."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionScanpathTeacher(nn.Module):
    """Minimal working diffusion-style teacher over fixation token sequences.

    The module learns to denoise scanpath token sequences conditioned on image
    context and a task token. It is lightweight enough for dummy-data runs while
    remaining a real training component rather than a stub.
    """

    def __init__(
        self,
        global_dim: int,
        num_patches: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        diffusion_steps: int = 50,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.diffusion_steps = diffusion_steps

        self.global_proj = nn.Linear(global_dim, hidden_dim)
        self.task_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.input_proj = nn.Linear(num_patches + 5, hidden_dim)
        self.time_emb = nn.Embedding(diffusion_steps, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, num_patches + 5)

        betas = torch.linspace(1e-4, 2e-2, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def _pack_targets(
        self,
        patch_idx: torch.Tensor,
        coords: torch.Tensor,
        delta: torch.Tensor,
        durations: torch.Tensor,
    ) -> torch.Tensor:
        patch_onehot = F.one_hot(patch_idx.long(), num_classes=self.num_patches).float()
        return torch.cat([patch_onehot, coords, delta, durations.unsqueeze(-1)], dim=-1)

    def _split_outputs(self, packed: torch.Tensor) -> dict[str, torch.Tensor]:
        patch_logits = packed[..., : self.num_patches]
        rest = packed[..., self.num_patches :]
        return {
            "patch_logits": patch_logits,
            "coord_pred": torch.sigmoid(rest[..., :2]),
            "delta_pred": torch.tanh(rest[..., 2:4]),
            "dur_pred": rest[..., 4],
        }

    def _q_sample(self, x0: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t_idx].view(-1, 1, 1)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise

    def forward(
        self,
        global_context: torch.Tensor,
        patch_idx: torch.Tensor,
        coords: torch.Tensor,
        delta: torch.Tensor,
        durations: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, t, _ = coords.shape
        packed = self._pack_targets(patch_idx, coords, delta, durations)
        t_idx = torch.randint(0, self.diffusion_steps, (b,), device=coords.device)
        noise = torch.randn_like(packed)
        noisy = self._q_sample(packed, t_idx=t_idx, noise=noise)

        cond = self.global_proj(global_context).unsqueeze(1)
        task = self.task_token.expand(b, -1, -1)
        hidden = self.input_proj(noisy) + self.time_emb(t_idx).unsqueeze(1) + cond + task
        hidden = self.encoder(hidden)
        pred_noise = self.out_proj(hidden)
        denoised = noisy - pred_noise

        noise_loss = F.mse_loss(pred_noise, noise, reduction="none").mean(dim=-1)
        masked_noise_loss = (noise_loss * mask).sum() / (mask.sum() + 1e-6)

        decoded = self._split_outputs(denoised)
        decoded["teacher_loss"] = masked_noise_loss
        decoded["teacher_tokens"] = denoised
        return decoded
