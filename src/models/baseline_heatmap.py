"""Baseline B: image classifier with gaze heatmap auxiliary supervision."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import VisionBackbone
from src.models.modules.heads import BinaryClassificationHead


class BaselineHeatmapAux(nn.Module):
    """Joint classification + heatmap regression model."""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        heatmap_size: int = 96,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = VisionBackbone(model_name=backbone_name, pretrained=pretrained)
        c = self.backbone.out_channels
        self.cls_head = BinaryClassificationHead(in_dim=c, hidden_dim=256, dropout=dropout)
        self.hm_head = nn.Sequential(
            nn.Conv2d(c, c // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c // 2, 1, kernel_size=1),
        )
        self.heatmap_size = heatmap_size

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        fmap, pooled = self.backbone(x)
        cls_logits = self.cls_head(pooled)
        hm = self.hm_head(fmap)
        hm = F.interpolate(hm, size=(self.heatmap_size, self.heatmap_size), mode="bilinear", align_corners=False)
        hm = torch.sigmoid(hm)
        return {"logits": cls_logits, "heatmap": hm}
