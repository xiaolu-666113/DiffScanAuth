"""Backbone wrappers with timm-first and torchvision fallback."""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception:  # pragma: no cover
    timm = None

try:
    from torchvision.models import resnet18, vit_b_16
except Exception:  # pragma: no cover
    resnet18 = None
    vit_b_16 = None


def _tokens_to_feature_map(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert transformer tokens into a feature map and pooled vector."""
    if tokens.ndim != 3:
        raise ValueError(f"Expected [B, N, C] tokens, got shape={tuple(tokens.shape)}")
    b, n, c = tokens.shape
    spatial_n = n
    start = 0
    side = int(math.sqrt(spatial_n))
    if side * side != spatial_n and n > 1:
        spatial_n = n - 1
        start = 1
        side = int(math.sqrt(spatial_n))
    if side * side != spatial_n:
        pooled = tokens.mean(dim=1)
        fmap = pooled.view(b, c, 1, 1)
        return fmap, pooled
    spatial = tokens[:, start : start + spatial_n, :]
    fmap = spatial.transpose(1, 2).reshape(b, c, side, side)
    pooled = spatial.mean(dim=1)
    return fmap, pooled


class ClassifierBackbone(nn.Module):
    """Backbone wrapper for global pooled features."""

    def __init__(self, model_name: str, pretrained: bool = True, in_chans: int = 3) -> None:
        super().__init__()
        self.model_name = model_name
        self.backend = "unknown"

        if timm is not None:
            try:
                self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=in_chans)
                self.out_channels = int(getattr(self.model, "num_features", 768))
                self.backend = "timm"
                return
            except Exception as exc:
                warnings.warn(f"Unable to build timm classifier backbone '{model_name}', falling back: {exc}")

        if vit_b_16 is not None and "vit" in model_name.lower():
            weights = None
            self.model = vit_b_16(weights=weights)
            if in_chans != 3:
                self.model.conv_proj = nn.Conv2d(in_chans, 768, kernel_size=16, stride=16)
            self.model.heads = nn.Identity()
            self.out_channels = 768
            self.backend = "torchvision_vit"
            return

        if resnet18 is None:
            raise RuntimeError("No available classifier backbone backend")
        m = resnet18(weights=None)
        if in_chans != 3:
            m.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Identity()
        self.model = m
        self.out_channels = 512
        self.backend = "torchvision_resnet"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.model(x)
        if feat.ndim > 2:
            feat = feat.flatten(1)
        return feat


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
        self.model_name = model_name

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
                self.backend = "timm_features_only"
                return
            except Exception:
                pass

            try:
                self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=in_chans)
                self.out_channels = int(getattr(self.model, "num_features", 768))
                self.backend = "timm_forward_features"
                return
            except Exception as exc:
                warnings.warn(f"timm backbone '{model_name}' unavailable, fallback to torchvision: {exc}")

        if resnet18 is None:
            raise RuntimeError("Neither timm nor torchvision resnet18 is available")

        m = resnet18(weights=None)
        if in_chans != 3:
            m.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layers = nn.Sequential(m.layer1, m.layer2, m.layer3, m.layer4)
        self.out_channels = 512
        self.backend = "torchvision_resnet"

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backend == "timm_features_only":
            fmap = self.model(x)[-1]
            pooled = F.adaptive_avg_pool2d(fmap, output_size=1).flatten(1)
            return fmap, pooled
        if self.backend == "timm_forward_features":
            feat = self.model.forward_features(x)
            if isinstance(feat, (list, tuple)):
                feat = feat[-1]
            if isinstance(feat, dict):
                if "x_norm_patchtokens" in feat:
                    feat = feat["x_norm_patchtokens"]
                elif "last_hidden_state" in feat:
                    feat = feat["last_hidden_state"]
                else:
                    feat = next(iter(feat.values()))
            if feat.ndim == 4:
                fmap = feat
                pooled = F.adaptive_avg_pool2d(fmap, output_size=1).flatten(1)
                return fmap, pooled
            fmap, pooled = _tokens_to_feature_map(feat)
            return fmap, pooled
        fmap = self.layers(self.stem(x))
        pooled = F.adaptive_avg_pool2d(fmap, output_size=1).flatten(1)
        return fmap, pooled
