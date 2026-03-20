"""Dual-stream visual encoder with preferred/fallback stream adapters."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import torch
import torch.nn as nn

from src.models.backbones import VisionBackbone


@dataclass
class StreamFeatures:
    """Container for one visual stream."""

    fmap: torch.Tensor
    tokens: torch.Tensor
    pooled: torch.Tensor
    model_name: str
    backend_used: str


def _fallback_name(requested_name: str, fallback_name: str) -> str:
    name = requested_name.lower()
    if "siglip" in name or "clip" in name:
        return "vit_small_patch16_224"
    if "dinov2" in name or "dino" in name:
        return "convnext_tiny"
    return requested_name


class StreamAdapter(nn.Module):
    """Adapter around a feature-map backbone with preferred-model fallback."""

    def __init__(
        self,
        requested_name: str,
        fallback_name: str,
        pretrained: bool,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        resolved_name = _fallback_name(requested_name, fallback_name)
        self.requested_name = requested_name
        self.resolved_name = resolved_name
        self.backbone = VisionBackbone(model_name=resolved_name, pretrained=pretrained, in_chans=in_chans)
        if resolved_name != requested_name:
            warnings.warn(
                f"Requested stream backbone '{requested_name}' unavailable in this environment; "
                f"using fallback '{resolved_name}' instead."
            )

    @property
    def out_channels(self) -> int:
        return self.backbone.out_channels

    @property
    def backend(self) -> str:
        return self.backbone.backend

    def forward(self, x: torch.Tensor) -> StreamFeatures:
        fmap, pooled = self.backbone(x)
        tokens = fmap.flatten(2).transpose(1, 2)
        return StreamFeatures(
            fmap=fmap,
            tokens=tokens,
            pooled=pooled,
            model_name=self.resolved_name,
            backend_used=self.backend,
        )


class DualStreamEncoder(nn.Module):
    """Dual-stream visual encoder for DiffScanAuth.

    Intended paper version:
    - global stream: SigLIP2-like semantic encoder
    - local stream: DINOv2-like dense artifact encoder

    Runtime fallback:
    - timm / torchvision-compatible backbones through :class:`VisionBackbone`
    """

    def __init__(
        self,
        global_stream_name: str = "siglip2-so400m-patch16-224",
        local_stream_name: str = "dinov2_vits14",
        pretrained: bool = True,
        use_local_stream: bool = True,
    ) -> None:
        super().__init__()
        self.global_stream = StreamAdapter(
            requested_name=global_stream_name,
            fallback_name="convnext_tiny",
            pretrained=pretrained,
        )
        self.use_local_stream = use_local_stream
        self.local_stream = (
            StreamAdapter(
                requested_name=local_stream_name,
                fallback_name="convnext_tiny",
                pretrained=pretrained,
            )
            if use_local_stream
            else None
        )

    @property
    def global_dim(self) -> int:
        return self.global_stream.out_channels

    @property
    def local_dim(self) -> int:
        if self.local_stream is None:
            return self.global_dim
        return self.local_stream.out_channels

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor | str]:
        global_feats = self.global_stream(image)
        if self.local_stream is None:
            local_feats = StreamFeatures(
                fmap=global_feats.fmap,
                tokens=global_feats.tokens,
                pooled=global_feats.pooled,
                model_name=global_feats.model_name,
                backend_used=global_feats.backend_used,
            )
        else:
            local_feats = self.local_stream(image)

        return {
            "global_map": global_feats.fmap,
            "global_tokens": global_feats.tokens,
            "global_pooled": global_feats.pooled,
            "local_map": local_feats.fmap,
            "local_tokens": local_feats.tokens,
            "local_pooled": local_feats.pooled,
            "global_backbone_used": global_feats.model_name,
            "local_backbone_used": local_feats.model_name,
        }
