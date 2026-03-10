"""Utilities for leakage-safe image-level data splitting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitResult:
    """Container for split dataframes and report payload."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    report: dict


def _make_strata(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    return df[list(cols)].fillna("NA").astype(str).agg("|".join, axis=1)


def _can_stratify(strata: pd.Series, min_count: int = 2) -> bool:
    counts = strata.value_counts()
    return len(counts) > 1 and counts.min() >= min_count


def _distribution(df: pd.DataFrame, col: str) -> dict[str, int]:
    if col not in df.columns:
        return {}
    return df[col].fillna("NA").astype(str).value_counts().to_dict()


def _report_split(df: pd.DataFrame) -> dict:
    return {
        "n_images": int(df["image_id"].nunique()),
        "n_rows": int(len(df)),
        "label_dist": _distribution(df, "label"),
        "scene_dist": _distribution(df, "scene"),
        "generator_dist": _distribution(df, "generator"),
    }


def make_splits(
    image_df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitResult:
    """Create train/val/test split by image_id with stratification fallback."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")

    image_level = image_df.drop_duplicates(subset=["image_id"]).copy()
    if image_level["image_id"].duplicated().any():
        raise ValueError("image_id must be unique at image-level splitting")

    candidates: list[list[str]] = []
    if "generator" in image_level.columns and image_level["generator"].fillna("").ne("").any():
        candidates.append(["label", "scene", "generator"])
    candidates.extend([["label", "scene"], ["label"]])

    selected_cols: list[str] | None = None
    selected_strata: pd.Series | None = None
    downgrade_log: list[str] = []

    for cols in candidates:
        if any(c not in image_level.columns for c in cols):
            downgrade_log.append(f"skip {cols}: missing columns")
            continue
        strata = _make_strata(image_level, cols)
        if _can_stratify(strata, min_count=2):
            selected_cols = cols
            selected_strata = strata
            break
        downgrade_log.append(f"degrade from {cols}: insufficient per-stratum counts")

    if selected_cols is None or selected_strata is None:
        selected_cols = []
        selected_strata = None
        downgrade_log.append("all stratified modes unavailable, fallback to random split")

    idx = np.arange(len(image_level))
    strat = selected_strata if selected_strata is not None else None

    train_idx, temp_idx = train_test_split(
        idx,
        test_size=(1 - train_ratio),
        random_state=seed,
        stratify=strat,
    )

    temp_df = image_level.iloc[temp_idx]
    if strat is not None:
        temp_strata = _make_strata(temp_df, selected_cols)
        temp_strat = temp_strata if _can_stratify(temp_strata, min_count=2) else None
        if temp_strat is None:
            downgrade_log.append("second-stage split degraded to random due to rare classes")
    else:
        temp_strat = None

    val_size = val_ratio / (val_ratio + test_ratio)
    val_rel_idx, test_rel_idx = train_test_split(
        np.arange(len(temp_idx)),
        test_size=(1 - val_size),
        random_state=seed,
        stratify=temp_strat,
    )

    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    train_ids = set(image_level.iloc[train_idx]["image_id"].tolist())
    val_ids = set(image_level.iloc[val_idx]["image_id"].tolist())
    test_ids = set(image_level.iloc[test_idx]["image_id"].tolist())

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("image leakage detected among splits")

    full = image_df.copy()
    full["split"] = ""
    full.loc[full["image_id"].isin(train_ids), "split"] = "train"
    full.loc[full["image_id"].isin(val_ids), "split"] = "val"
    full.loc[full["image_id"].isin(test_ids), "split"] = "test"

    train_df = full[full["split"] == "train"].copy()
    val_df = full[full["split"] == "val"].copy()
    test_df = full[full["split"] == "test"].copy()

    image_to_split = full.drop_duplicates("image_id")["split"].value_counts().to_dict()
    report = {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "stratify_cols": selected_cols,
        "downgrade_log": downgrade_log,
        "image_split_counts": image_to_split,
        "train": _report_split(train_df),
        "val": _report_split(val_df),
        "test": _report_split(test_df),
        "image_leakage_check": {
            "train_val_overlap": len(train_ids & val_ids),
            "train_test_overlap": len(train_ids & test_ids),
            "val_test_overlap": len(val_ids & test_ids),
        },
    }

    return SplitResult(train=train_df, val=val_df, test=test_df, report=report)
