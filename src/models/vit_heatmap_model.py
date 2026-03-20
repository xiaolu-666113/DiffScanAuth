"""ViT-B/16 + human gaze heatmap supervision baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import VisionBackbone
from src.models.modules.heads import BinaryClassificationHead


class ViTHeatmapModel(nn.Module):
    """Static classifier with auxiliary heatmap prediction head."""

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        heatmap_size: int = 96,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = VisionBackbone(model_name=backbone_name, pretrained=pretrained)
        c = self.backbone.out_channels
        self.cls_head = BinaryClassificationHead(c, hidden_dim=max(128, c // 2), dropout=dropout)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(c, c // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c // 2, 1, kernel_size=1),
        )
        self.heatmap_size = heatmap_size

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        fmap, pooled = self.backbone(image)
        logits = self.cls_head(pooled)
        heatmap = torch.sigmoid(self.heatmap_head(fmap))
        heatmap = F.interpolate(heatmap, size=(self.heatmap_size, self.heatmap_size), mode="bilinear", align_corners=False)
        return {"logits": logits, "heatmap": heatmap}
