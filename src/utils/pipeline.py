"""Reusable dataset preparation and experiment build pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.collate import collate_gaze, collate_static
from src.datasets.dataset_adapter import build_eye_tracking, build_metadata
from src.datasets.gaze_dataset import GazeSequenceDataset
from src.datasets.image_dataset import StaticImageDataset
from src.datasets.split_utils import make_splits
from src.features.fixation_tokenizer import add_patch_tokens
from src.features.gaze_processing import clean_and_normalize_fixations, normalize_columns, normalize_duration, to_fixation_level
from src.features.heatmap import build_image_heatmaps, save_heatmaps
from src.lightning.lit_aide import LitAIDE
from src.lightning.lit_diffscanauth import LitDiffScanAuth
from src.lightning.lit_heatmap import LitHeatmap
from src.lightning.lit_seq import LitSeq
from src.lightning.lit_seqdet import LitSeqDet
from src.lightning.lit_static import LitStatic
from src.lightning.lit_vit import LitViT
from src.utils.io import ensure_dir, save_csv, save_json
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def run_build_metadata(data_cfg: dict[str, Any]) -> pd.DataFrame:
    """Build standardized metadata.csv."""
    ensure_dir(data_cfg["processed_dir"])
    metadata_df = build_metadata(
        raw_dir=data_cfg["raw_dir"],
        output_csv=data_cfg["metadata_csv"],
        output_report_json=data_cfg.get("inspect_report_json"),
        allow_synthetic=bool(data_cfg.get("allow_synthetic", True)),
        synthetic_num_images=int(data_cfg.get("synthetic_num_images", 120)),
        seed=int(data_cfg.get("synthetic_seed", 42)),
    )
    return metadata_df


def run_preprocess_eye_tracking(data_cfg: dict[str, Any]) -> pd.DataFrame:
    """Build eye_tracking.csv and processed_fixations.csv."""
    metadata_df = pd.read_csv(data_cfg["metadata_csv"])

    eye_csv = Path(data_cfg["eye_tracking_csv"])
    if bool(data_cfg.get("force_rebuild_eye_tracking", False)) or not eye_csv.exists():
        build_eye_tracking(
            raw_dir=data_cfg["raw_dir"],
            metadata_csv=data_cfg["metadata_csv"],
            output_csv=data_cfg["eye_tracking_csv"],
            allow_synthetic=bool(data_cfg.get("allow_synthetic", True)),
            synthetic_num_subjects=int(data_cfg.get("synthetic_num_subjects", 10)),
            seed=int(data_cfg.get("synthetic_seed", 42)),
        )

    eye_df = pd.read_csv(data_cfg["eye_tracking_csv"])
    eye_df = normalize_columns(eye_df)
    fix_df = to_fixation_level(eye_df)
    fix_df = clean_and_normalize_fixations(
        fix_df,
        metadata_df,
        min_duration_ms=float(data_cfg.get("min_duration_ms", 40.0)),
        max_duration_ms=float(data_cfg.get("max_duration_ms", 2000.0)),
    )

    fix_df["duration_norm"] = normalize_duration(
        fix_df["duration_ms"].to_numpy(dtype=np.float32),
        mode=str(data_cfg.get("duration_norm_mode", "log_zscore")),
    )

    fix_df = add_patch_tokens(fix_df, grid_size=int(data_cfg.get("patch_grid_size", 24)))

    if "split" in metadata_df.columns:
        split_map = metadata_df[["image_id", "split"]].drop_duplicates("image_id")
        fix_df = fix_df.drop(columns=["split"], errors="ignore").merge(split_map, on="image_id", how="left")
        fix_df["split"] = fix_df["split"].fillna("")
    else:
        fix_df["split"] = ""

    keep_cols = [
        "subject_id",
        "image_id",
        "fixation_idx",
        "x_norm",
        "y_norm",
        "duration_norm",
        "duration_ms",
        "delta_x",
        "delta_y",
        "patch_index",
        "split",
    ]
    fix_df = fix_df[keep_cols].copy()
    save_csv(fix_df, data_cfg["processed_fixations_csv"])
    return fix_df


def run_make_splits(data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Create train/val/test split files and split report; update metadata/fixations."""
    ensure_dir(data_cfg["splits_dir"])
    metadata_df = pd.read_csv(data_cfg["metadata_csv"])
    fix_df = pd.read_csv(data_cfg["processed_fixations_csv"])

    split_cfg = data_cfg.get("split", {})
    split_res = make_splits(
        image_df=metadata_df,
        seed=int(split_cfg.get("seed", 42)),
        train_ratio=float(split_cfg.get("train", 0.7)),
        val_ratio=float(split_cfg.get("val", 0.15)),
        test_ratio=float(split_cfg.get("test", 0.15)),
    )

    train_csv = Path(data_cfg["splits_dir"]) / "train.csv"
    val_csv = Path(data_cfg["splits_dir"]) / "val.csv"
    test_csv = Path(data_cfg["splits_dir"]) / "test.csv"

    save_csv(split_res.train, train_csv)
    save_csv(split_res.val, val_csv)
    save_csv(split_res.test, test_csv)
    save_json(split_res.report, data_cfg["split_report_json"])

    merged = pd.concat([split_res.train, split_res.val, split_res.test], ignore_index=True)
    merged = merged.drop_duplicates(subset=["image_id"])
    save_csv(merged, data_cfg["metadata_csv"])

    split_map = merged[["image_id", "split"]].drop_duplicates("image_id")
    fix_df = fix_df.drop(columns=["split"], errors="ignore").merge(split_map, on="image_id", how="left")
    fix_df["split"] = fix_df["split"].fillna("")
    save_csv(fix_df, data_cfg["processed_fixations_csv"])

    heatmaps: dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        heatmaps.update(
            build_image_heatmaps(
                fix_df=fix_df,
                split=split,
                size=int(data_cfg.get("heatmap_size", 96)),
                sigma=float(data_cfg.get("heatmap_sigma", 2.5)),
            )
        )
    save_heatmaps(heatmaps, data_cfg["heatmap_dir"])

    return split_res.report


def ensure_prepared_data(data_cfg: dict[str, Any]) -> None:
    """Ensure metadata, fixations, and split artifacts are available."""
    metadata_csv = Path(data_cfg["metadata_csv"])
    fix_csv = Path(data_cfg["processed_fixations_csv"])
    split_report = Path(data_cfg["split_report_json"])

    if not metadata_csv.exists():
        LOGGER.info("metadata.csv missing, running build_metadata")
        run_build_metadata(data_cfg)

    need_fix = not fix_csv.exists()
    if not need_fix:
        fix_df = pd.read_csv(fix_csv, nrows=5)
        need_fix = any(col not in fix_df.columns for col in ["delta_x", "delta_y"])
    if need_fix:
        LOGGER.info("processed_fixations.csv missing or outdated, running preprocess_eye_tracking")
        run_preprocess_eye_tracking(data_cfg)

    need_split = not split_report.exists()
    if not need_split:
        md = pd.read_csv(metadata_csv)
        need_split = "split" not in md.columns or md["split"].astype(str).eq("").all()
    if need_split:
        LOGGER.info("split files missing or empty, running make_splits")
        run_make_splits(data_cfg)


def build_dataloaders(
    data_cfg: dict[str, Any],
    model_name: str | None = None,
    train_aug: bool = True,
    model_cfg: dict[str, Any] | None = None,
) -> dict[str, DataLoader]:
    """Construct train/val/test dataloaders for selected model type."""
    if model_cfg is not None and model_name is None:
        model_name = str(model_cfg.get("name"))
    if model_name is None:
        raise ValueError("Either model_name or model_cfg must be provided to build_dataloaders")

    loader_cfg = data_cfg.get("loader", {})
    batch_size = int(loader_cfg.get("batch_size", 8))
    num_workers = int(loader_cfg.get("num_workers", 4))
    pin_memory = bool(loader_cfg.get("pin_memory", True))

    if model_name in {"baseline_static", "baseline_heatmap", "vit_b16", "aide_style", "vit_gaze_heatmap"}:
        use_heatmap = model_name in {"baseline_heatmap", "vit_gaze_heatmap"}
        train_ds = StaticImageDataset(
            metadata_csv=data_cfg["metadata_csv"],
            split="train",
            image_size=int(data_cfg.get("image_size", 384)),
            train=True,
            use_aug=train_aug,
            heatmap_dir=data_cfg.get("heatmap_dir") if use_heatmap else None,
            heatmap_size=int(data_cfg.get("heatmap_size", 96)),
        )
        val_ds = StaticImageDataset(
            metadata_csv=data_cfg["metadata_csv"],
            split="val",
            image_size=int(data_cfg.get("image_size", 384)),
            train=False,
            use_aug=False,
            heatmap_dir=data_cfg.get("heatmap_dir") if use_heatmap else None,
            heatmap_size=int(data_cfg.get("heatmap_size", 96)),
        )
        test_ds = StaticImageDataset(
            metadata_csv=data_cfg["metadata_csv"],
            split="test",
            image_size=int(data_cfg.get("image_size", 384)),
            train=False,
            use_aug=False,
            heatmap_dir=data_cfg.get("heatmap_dir") if use_heatmap else None,
            heatmap_size=int(data_cfg.get("heatmap_size", 96)),
        )
        collate_fn = collate_static
    else:
        shuffled_gaze = bool((model_cfg or {}).get("shuffled_gaze", False))
        shuffle_seed = int((model_cfg or {}).get("shuffle_seed", data_cfg.get("synthetic_seed", 42)))
        train_ds = GazeSequenceDataset(
            metadata_csv=data_cfg["metadata_csv"],
            fixations_csv=data_cfg["processed_fixations_csv"],
            split="train",
            image_size=int(data_cfg.get("image_size", 384)),
            max_fixations=int(data_cfg.get("max_fixations", 12)),
            patch_grid_size=int(data_cfg.get("patch_grid_size", 24)),
            duration_norm_mode=str(data_cfg.get("duration_norm_mode", "log_zscore")),
            train=True,
            use_aug=train_aug,
            heatmap_size=int(data_cfg.get("heatmap_size", 96)),
            shuffled_gaze=shuffled_gaze,
            shuffle_seed=shuffle_seed,
        )
        val_ds = GazeSequenceDataset(
            metadata_csv=data_cfg["metadata_csv"],
            fixations_csv=data_cfg["processed_fixations_csv"],
            split="val",
            image_size=int(data_cfg.get("image_size", 384)),
            max_fixations=int(data_cfg.get("max_fixations", 12)),
            patch_grid_size=int(data_cfg.get("patch_grid_size", 24)),
            duration_norm_mode=str(data_cfg.get("duration_norm_mode", "log_zscore")),
            train=False,
            use_aug=False,
            heatmap_size=int(data_cfg.get("heatmap_size", 96)),
            shuffled_gaze=shuffled_gaze,
            shuffle_seed=shuffle_seed,
        )
        test_ds = GazeSequenceDataset(
            metadata_csv=data_cfg["metadata_csv"],
            fixations_csv=data_cfg["processed_fixations_csv"],
            split="test",
            image_size=int(data_cfg.get("image_size", 384)),
            max_fixations=int(data_cfg.get("max_fixations", 12)),
            patch_grid_size=int(data_cfg.get("patch_grid_size", 24)),
            duration_norm_mode=str(data_cfg.get("duration_norm_mode", "log_zscore")),
            train=False,
            use_aug=False,
            heatmap_size=int(data_cfg.get("heatmap_size", 96)),
            shuffled_gaze=shuffled_gaze,
            shuffle_seed=shuffle_seed,
        )
        collate_fn = collate_gaze

    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        ),
    }


def build_lightning_module(model_cfg: dict[str, Any]):
    """Instantiate corresponding lightning module from model config."""
    name = str(model_cfg.get("name"))
    optim_cfg = dict(model_cfg.get("optimizer", {}))

    if name == "vit_b16":
        return LitViT(model_cfg=dict(model_cfg), optim_cfg=optim_cfg)
    if name == "aide_style":
        return LitAIDE(model_cfg=dict(model_cfg), optim_cfg=optim_cfg)
    if name == "vit_gaze_heatmap":
        return LitHeatmap(model_cfg=dict(model_cfg), optim_cfg=optim_cfg)
    if name == "seqdet_no_gaze":
        return LitSeqDet(model_cfg=dict(model_cfg), optim_cfg=optim_cfg)
    if name == "diffscanauth":
        return LitDiffScanAuth(model_cfg=dict(model_cfg), optim_cfg=optim_cfg)

    if name == "baseline_static":
        return LitStatic(model_cfg=dict(model_cfg), optim_cfg=optim_cfg)
    if name == "baseline_heatmap":
        return LitHeatmap(model_cfg=dict(model_cfg), optim_cfg=optim_cfg)
    if name == "seq_gaze_detector":
        return LitSeq(model_cfg=dict(model_cfg), optim_cfg=optim_cfg)

    raise ValueError(f"Unsupported model name: {name}")
