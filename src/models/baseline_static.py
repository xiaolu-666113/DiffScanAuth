"""Baseline A: static image classifier (without gaze)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.backbones import VisionBackbone
from src.models.modules.heads import BinaryClassificationHead


class BaselineStaticClassifier(nn.Module):
    """Backbone + binary head for real/fake classification."""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = VisionBackbone(model_name=backbone_name, pretrained=pretrained)
        self.head = BinaryClassificationHead(in_dim=self.backbone.out_channels, hidden_dim=256, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, pooled = self.backbone(x)
        return self.head(pooled)
