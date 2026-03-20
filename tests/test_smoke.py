from __future__ import annotations

from pathlib import Path

from src.utils.pipeline import (
    build_dataloaders,
    build_lightning_module,
    run_build_metadata,
    run_make_splits,
    run_preprocess_eye_tracking,
)


def test_smoke_data_and_model_init(tmp_path: Path) -> None:
    data_cfg = {
        "raw_dir": str(tmp_path / "raw"),
        "processed_dir": str(tmp_path / "processed"),
        "splits_dir": str(tmp_path / "splits"),
        "metadata_csv": str(tmp_path / "processed" / "metadata.csv"),
        "eye_tracking_csv": str(tmp_path / "processed" / "eye_tracking.csv"),
        "processed_fixations_csv": str(tmp_path / "processed" / "processed_fixations.csv"),
        "heatmap_dir": str(tmp_path / "processed" / "heatmaps"),
        "inspect_report_json": str(tmp_path / "processed" / "inspect.json"),
        "split_report_json": str(tmp_path / "splits" / "split_report.json"),
        "allow_synthetic": True,
        "synthetic_num_images": 24,
        "synthetic_seed": 11,
        "synthetic_num_subjects": 4,
        "patch_grid_size": 8,
        "duration_norm_mode": "log_zscore",
        "min_duration_ms": 40.0,
        "max_duration_ms": 2000.0,
        "split": {"train": 0.7, "val": 0.15, "test": 0.15, "seed": 11},
        "heatmap_size": 32,
        "image_size": 224,
        "max_fixations": 8,
        "loader": {"batch_size": 2, "num_workers": 0, "pin_memory": False},
    }

    run_build_metadata(data_cfg)
    run_preprocess_eye_tracking(data_cfg)
    run_make_splits(data_cfg)

    model_cfg = {
        "name": "diffscanauth",
        "global_stream_name": "resnet18",
        "local_stream_name": "resnet18",
        "pretrained": False,
        "use_local_stream": True,
        "use_teacher": True,
        "use_gaze_supervision": True,
        "use_teacher_distill": True,
        "policy_type": "gru",
        "teacher_hidden_dim": 64,
        "patch_grid_size": 8,
        "policy_hidden_dim": 64,
        "glimpse_dim": 64,
        "accumulator_hidden_dim": 64,
        "accumulator_backend": "gru",
        "max_steps": 8,
        "fixed_steps": 8,
        "stop_mode": "learned_stop",
        "dropout": 0.1,
        "teacher_forcing_ratio": 1.0,
        "teacher_forcing_final": 1.0,
        "scheduled_sampling_ratio": 0.0,
        "scheduled_sampling_final": 0.0,
        "align_loss": "mse",
        "loss_cls": 1.0,
        "loss_gaze": 1.0,
        "loss_distill": 0.5,
        "loss_stop": 0.2,
        "loss_align": 0.1,
        "loss_rl": 0.0,
        "optimizer": {"lr": 1e-3, "weight_decay": 0.0, "pos_weight": None},
    }

    loaders = build_dataloaders(data_cfg, model_name="diffscanauth", model_cfg=model_cfg, train_aug=False)
    lit = build_lightning_module(model_cfg)
    batch = next(iter(loaders["train"]))
    out = lit.model(batch)
    assert out["cls_logits"].shape[0] == batch["image"].shape[0]
