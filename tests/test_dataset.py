from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.pipeline import run_build_metadata, run_make_splits, run_preprocess_eye_tracking


def test_dataset_pipeline_with_synthetic(tmp_path: Path) -> None:
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
        "synthetic_seed": 7,
        "patch_grid_size": 16,
        "duration_norm_mode": "log_zscore",
        "min_duration_ms": 40.0,
        "max_duration_ms": 2000.0,
        "split": {"train": 0.7, "val": 0.15, "test": 0.15, "seed": 7},
        "heatmap_size": 64,
    }

    m = run_build_metadata(data_cfg)
    f = run_preprocess_eye_tracking(data_cfg)
    r = run_make_splits(data_cfg)

    assert len(m) > 0
    assert len(f) > 0
    assert Path(data_cfg["metadata_csv"]).exists()
    assert Path(data_cfg["processed_fixations_csv"]).exists()
    assert Path(data_cfg["split_report_json"]).exists()

    md = pd.read_csv(data_cfg["metadata_csv"])
    assert "split" in md.columns
    assert set(md["split"].unique()).issubset({"train", "val", "test"})
    assert r["image_leakage_check"]["train_val_overlap"] == 0
