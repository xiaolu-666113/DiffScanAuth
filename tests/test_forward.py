from __future__ import annotations

import torch

from src.models.baseline_heatmap import BaselineHeatmapAux
from src.models.baseline_static import BaselineStaticClassifier
from src.models.seq_gaze_detector import SeqGazeDetector


def test_static_forward() -> None:
    model = BaselineStaticClassifier(backbone_name="resnet18", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2,)


def test_heatmap_forward() -> None:
    model = BaselineHeatmapAux(backbone_name="resnet18", pretrained=False, heatmap_size=32)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out["logits"].shape == (2,)
    assert out["heatmap"].shape == (2, 1, 32, 32)


def test_seq_forward() -> None:
    model = SeqGazeDetector(backbone_name="resnet18", pretrained=False, patch_grid_size=8)
    b, t = 2, 6
    batch = {
        "image": torch.randn(b, 3, 224, 224),
        "patch_idx": torch.randint(0, 64, (b, t)),
        "fix_xy": torch.rand(b, t, 2),
        "fix_dur": torch.rand(b, t),
        "mask": torch.ones(b, t),
    }
    out = model(batch)
    assert out["cls_logits"].shape == (b,)
    assert out["patch_logits"].shape == (b, t, 64)
