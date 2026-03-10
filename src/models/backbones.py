"""Backbone wrappers with timm-first and torchvision fallback."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception:  # pragma: no cover - optional dependency
    timm = None

try:
    from torchvision.models import resnet18
except Exception:  # pragma: no cover - optional dependency
    resnet18 = None


class VisionBackbone(nn.Module):
    """Feature-map backbone wrapper.

    Returns:
        fmap: (B, C, H, W)
        pooled: (B, C)
    """

    def __init__(
        self,
        model_name: str = "convnext_tiny",
        pretrained: bool = True,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        self.backend = "unknown"

        if timm is not None:
            try:
                self.model = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    in_chans=in_chans,
                    features_only=True,
                    out_indices=(-1,),
                )
                self.out_channels = int(self.model.feature_info.channels()[-1])
                self.backend = "timm"
                return
            except Exception as e:
                warnings.warn(f"timm backbone '{model_name}' unavailable, fallback to torchvision: {e}")

        if resnet18 is None:
            raise RuntimeError("Neither timm nor torchvision resnet18 is available")

        m = resnet18(weights=None)
        if in_chans != 3:
            m.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layers = nn.Sequential(m.layer1, m.layer2, m.layer3, m.layer4)
        self.out_channels = 512
        self.backend = "torchvision"

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backend == "timm":
            fmap = self.model(x)[-1]
        else:
            fmap = self.layers(self.stem(x))
        pooled = F.adaptive_avg_pool2d(fmap, output_size=1).flatten(1)
        return fmap, pooled
