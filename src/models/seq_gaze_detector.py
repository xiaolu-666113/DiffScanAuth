"""Main model: gaze-supervised sequential detector."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.backbones import VisionBackbone
from src.models.modules.accumulator import SelectiveAccumulator
from src.models.modules.gaze_policy import GazePolicyGRU
from src.models.modules.glimpse_reader import GlimpseReader
from src.models.modules.heads import BinaryClassificationHead


class SeqGazeDetector(nn.Module):
    """Image + fixation sequence model with evidence accumulation."""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        patch_grid_size: int = 24,
        policy_hidden_dim: int = 256,
        glimpse_dim: int = 256,
        accumulator_hidden_dim: int = 256,
        accumulator_backend: str = "gru",
        stop_mode: str = "fixed_k",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stop_mode = stop_mode
        self.patch_grid_size = patch_grid_size
        self.num_patches = patch_grid_size * patch_grid_size

        self.backbone = VisionBackbone(model_name=backbone_name, pretrained=pretrained)
        global_dim = self.backbone.out_channels

        self.policy = GazePolicyGRU(
            global_dim=global_dim,
            hidden_dim=policy_hidden_dim,
            num_patches=self.num_patches,
            patch_emb_dim=128,
            global_proj_dim=128,
            num_layers=1,
            dropout=dropout,
        )

        self.reader = GlimpseReader(in_channels=self.backbone.out_channels, global_dim=global_dim, out_dim=glimpse_dim)
        self.accumulator = SelectiveAccumulator(
            input_dim=glimpse_dim,
            hidden_dim=accumulator_hidden_dim,
            backend=accumulator_backend,
            num_layers=1,
            dropout=dropout,
        )
        self.cls_head = BinaryClassificationHead(
            in_dim=accumulator_hidden_dim,
            hidden_dim=accumulator_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        teacher_forcing_ratio: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        image = batch["image"]
        teacher_patch = batch["patch_idx"].long()
        teacher_xy = batch["fix_xy"].float()
        teacher_dur = batch["fix_dur"].float()
        mask = batch["mask"].float()

        fmap, global_feat = self.backbone(image)
        policy_out = self.policy(global_feat, teacher_patch, teacher_xy, teacher_dur)

        use_teacher_coords = self.training and (teacher_forcing_ratio >= 1.0 or torch.rand(1, device=image.device) < teacher_forcing_ratio)
        coords_for_read = teacher_xy if use_teacher_coords else policy_out["coord_pred"]

        evidence = self.reader(
            fmap=fmap,
            coords=coords_for_read,
            global_feat=global_feat,
            durations=teacher_dur,
        )
        _, final = self.accumulator(evidence, mask=mask)
        cls_logits = self.cls_head(final)

        return {
            "cls_logits": cls_logits,
            "patch_logits": policy_out["patch_logits"],
            "coord_pred": policy_out["coord_pred"],
            "dur_pred": policy_out["dur_pred"],
            "stop_logits": policy_out["stop_logits"],
        }
