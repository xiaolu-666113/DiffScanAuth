"""ViT-B/16 image classifier baseline."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from src.models.backbones import ClassifierBackbone
from src.models.modules.heads import BinaryClassificationHead


class ViTB16Classifier(nn.Module):
    """Static ViT-B/16-style binary classifier with robust fallback."""

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        try:
            self.backbone = ClassifierBackbone(model_name=backbone_name, pretrained=pretrained, in_chans=3)
        except Exception as exc:
            warnings.warn(f"Falling back to resnet18-style classifier backbone because '{backbone_name}' failed: {exc}")
            self.backbone = ClassifierBackbone(model_name="resnet18", pretrained=False, in_chans=3)
        self.head = BinaryClassificationHead(self.backbone.out_channels, hidden_dim=max(128, self.backbone.out_channels // 2), dropout=dropout)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(image)
        return self.head(feat)
