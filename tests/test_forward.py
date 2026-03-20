from __future__ import annotations

import torch

from src.lightning.lit_diffscanauth import LitDiffScanAuth
from src.models.aide_style_detector import AIDEStyleDetector
from src.models.baseline_heatmap import BaselineHeatmapAux
from src.models.baseline_static import BaselineStaticClassifier
from src.models.diffscanauth import DiffScanAuth
from src.models.seq_gaze_detector import SeqGazeDetector
from src.models.vit_b16_classifier import ViTB16Classifier


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


def test_vit_b16_forward() -> None:
    model = ViTB16Classifier(backbone_name="vit_base_patch16_224", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2,)


def test_aide_style_forward() -> None:
    model = AIDEStyleDetector(rgb_backbone_name="resnet18", artifact_backbone_name="resnet18", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2,)


def test_diffscanauth_forward() -> None:
    model = DiffScanAuth(
        global_stream_name="resnet18",
        local_stream_name="resnet18",
        pretrained=False,
        use_local_stream=True,
        use_teacher=True,
        patch_grid_size=8,
        max_steps=6,
        policy_hidden_dim=64,
        glimpse_dim=64,
        accumulator_hidden_dim=64,
        teacher_hidden_dim=64,
        policy_type="gru",
        policy_layers=1,
        policy_heads=2,
    )
    b, t = 2, 6
    batch = {
        "image": torch.randn(b, 3, 224, 224),
        "patch_idx": torch.randint(0, 64, (b, t)),
        "fix_xy": torch.rand(b, t, 2),
        "fix_delta": torch.rand(b, t, 2),
        "fix_dur": torch.rand(b, t),
        "mask": torch.ones(b, t),
    }
    out = model(batch, teacher_forcing_ratio=1.0, scheduled_sampling_ratio=0.0, use_gaze_inputs=True)
    assert out["cls_logits"].shape == (b,)
    assert out["patch_logits"].shape == (b, t, 64)
    assert out["coord_pred"].shape == (b, t, 2)


def test_diffscanauth_teacher_checkpoint_loading(tmp_path) -> None:
    model_cfg = {
        "global_stream_name": "resnet18",
        "local_stream_name": "resnet18",
        "pretrained": False,
        "use_local_stream": True,
        "use_teacher": True,
        "patch_grid_size": 8,
        "max_steps": 6,
        "policy_hidden_dim": 64,
        "glimpse_dim": 64,
        "accumulator_hidden_dim": 64,
        "teacher_hidden_dim": 64,
        "policy_type": "gru",
        "policy_layers": 1,
        "policy_heads": 2,
    }
    optim_cfg = {"lr": 1e-3, "weight_decay": 1e-4}
    source_module = LitDiffScanAuth(model_cfg=model_cfg, optim_cfg=optim_cfg)
    ckpt_path = tmp_path / "teacher.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in source_module.model.state_dict().items()}}, ckpt_path)

    target_cfg = {
        **model_cfg,
        "teacher_ckpt": str(ckpt_path),
        "freeze_teacher": True,
    }
    target_module = LitDiffScanAuth(model_cfg=target_cfg, optim_cfg=optim_cfg)

    assert torch.allclose(
        source_module.model.teacher.out_proj.weight,
        target_module.model.teacher.out_proj.weight,
    )
    assert all(not param.requires_grad for param in target_module.model.teacher.parameters())
