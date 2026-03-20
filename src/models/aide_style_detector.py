"""AIDE-style hybrid forensic detector baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.backbones import ClassifierBackbone
from src.models.modules.heads import BinaryClassificationHead


class _FrequencyTransform(nn.Module):
    """Build a frequency/artifact representation from RGB input."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        gray = image.mean(dim=1, keepdim=True)
        fft = torch.fft.fft2(gray, norm="ortho")
        mag = torch.log1p(torch.abs(torch.fft.fftshift(fft, dim=(-2, -1))))
        mag = (mag - mag.mean(dim=(-2, -1), keepdim=True)) / (mag.std(dim=(-2, -1), keepdim=True) + 1e-6)
        return mag.repeat(1, 3, 1, 1)


class AIDEStyleDetector(nn.Module):
    """Hybrid RGB + frequency/static forensic detector."""

    def __init__(
        self,
        rgb_backbone_name: str = "convnext_tiny",
        artifact_backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.freq = _FrequencyTransform()
        self.rgb_backbone = ClassifierBackbone(model_name=rgb_backbone_name, pretrained=pretrained, in_chans=3)
        self.artifact_backbone = ClassifierBackbone(model_name=artifact_backbone_name, pretrained=False, in_chans=3)
        fusion_dim = self.rgb_backbone.out_channels + self.artifact_backbone.out_channels
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = BinaryClassificationHead(fusion_dim // 2, hidden_dim=max(128, fusion_dim // 4), dropout=dropout)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.rgb_backbone(image)
        artifact_feat = self.artifact_backbone(self.freq(image))
        fused = self.fusion(torch.cat([rgb_feat, artifact_feat], dim=-1))
        return self.head(fused)
