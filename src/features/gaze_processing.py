"""Gaze preprocessing utilities: standardization, fixation aggregation, and cleaning."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


EYE_SCHEMA_COLS = [
    "subject_id",
    "image_id",
    "t",
    "x",
    "y",
    "duration",
    "event_type",
    "validity",
    "pupil",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw eye-tracking table to unified column names."""
    aliases = {
        "subject_id": ["subject_id", "subject", "participant", "pid", "user_id"],
        "image_id": ["image_id", "image", "stimulus", "stimulus_id", "img", "filename"],
        "t": ["t", "time", "timestamp", "ts"],
        "x": ["x", "gaze_x", "fix_x", "pos_x", "x_pos"],
        "y": ["y", "gaze_y", "fix_y", "pos_y", "y_pos"],
        "duration": ["duration", "dur", "fix_duration", "duration_ms", "dwell"],
        "event_type": ["event_type", "event", "type"],
        "validity": ["validity", "valid", "confidence"],
        "pupil": ["pupil", "pupil_size"],
    }

    ren = {}
    lowered = {c.lower().strip(): c for c in df.columns}
    for target, cand in aliases.items():
        for c in cand:
            if c in lowered:
                ren[lowered[c]] = target
                break

    out = df.rename(columns=ren).copy()
    for c in EYE_SCHEMA_COLS:
        if c not in out.columns:
            out[c] = np.nan

    out["subject_id"] = out["subject_id"].fillna("unknown_subject").astype(str)
    out["image_id"] = out["image_id"].fillna("").astype(str)
    out["event_type"] = out["event_type"].fillna("fixation").astype(str).str.lower()

    for col in ["t", "x", "y", "duration", "validity", "pupil"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out[EYE_SCHEMA_COLS]


def aggregate_points_to_fixations(
    df: pd.DataFrame,
    dispersion_thresh: float = 35.0,
    max_gap_ms: float = 120.0,
) -> pd.DataFrame:
    """Approximate fixation aggregation from gaze points with dispersion grouping.

    Args:
        df: Gaze points in screen pixel coordinates.
        dispersion_thresh: Maximum centroid distance (pixels) before starting new fixation.
        max_gap_ms: Maximum time gap within one fixation.
    """
    rows = []
    work = df.sort_values(["subject_id", "image_id", "t"]).copy()

    for (sid, iid), g in work.groupby(["subject_id", "image_id"], sort=False):
        g = g.reset_index(drop=True)
        if len(g) == 0:
            continue

        current_idx = [0]

        def emit(indices: list[int]) -> None:
            chunk = g.iloc[indices]
            t0 = float(chunk["t"].iloc[0]) if not math.isnan(float(chunk["t"].iloc[0])) else 0.0
            t1 = float(chunk["t"].iloc[-1]) if not math.isnan(float(chunk["t"].iloc[-1])) else t0
            dur = t1 - t0
            if dur <= 0:
                dur = float(chunk["duration"].fillna(100).mean())
            rows.append(
                {
                    "subject_id": sid,
                    "image_id": iid,
                    "t": t0,
                    "x": float(chunk["x"].mean()),
                    "y": float(chunk["y"].mean()),
                    "duration": dur,
                    "event_type": "fixation",
                    "validity": float(chunk["validity"].mean()) if chunk["validity"].notna().any() else np.nan,
                    "pupil": float(chunk["pupil"].mean()) if chunk["pupil"].notna().any() else np.nan,
                }
            )

        for i in range(1, len(g)):
            prev = g.iloc[current_idx]
            cx = float(prev["x"].mean())
            cy = float(prev["y"].mean())
            px = float(g.at[i, "x"])
            py = float(g.at[i, "y"])
            dt = float(g.at[i, "t"] - g.at[i - 1, "t"]) if pd.notna(g.at[i, "t"]) and pd.notna(g.at[i - 1, "t"]) else 0.0
            dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            if dist <= dispersion_thresh and dt <= max_gap_ms:
                current_idx.append(i)
            else:
                emit(current_idx)
                current_idx = [i]

        emit(current_idx)

    return pd.DataFrame(rows, columns=EYE_SCHEMA_COLS)


def to_fixation_level(df: pd.DataFrame) -> pd.DataFrame:
    """Convert normalized eye records into fixation-level records."""
    if df.empty:
        return df.copy()

    fixation_like = df["event_type"].fillna("").str.contains("fix", case=False).mean() > 0.5
    if fixation_like:
        out = df.copy()
        out["event_type"] = "fixation"
        return out

    return aggregate_points_to_fixations(df)


def clean_and_normalize_fixations(
    fixation_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    min_duration_ms: float = 40.0,
    max_duration_ms: float = 2000.0,
) -> pd.DataFrame:
    """Remove outliers, normalize coordinates, and attach metadata fields."""
    df = fixation_df.copy()
    meta = metadata_df[["image_id", "width", "height", "split"]].drop_duplicates("image_id")
    df = df.merge(meta, on="image_id", how="left")

    df = df[df["width"].notna() & df["height"].notna()].copy()
    df["duration"] = df["duration"].fillna(120.0)

    df = df[(df["duration"] >= min_duration_ms) & (df["duration"] <= max_duration_ms)].copy()

    df["x_norm"] = df["x"] / df["width"].clip(lower=1)
    df["y_norm"] = df["y"] / df["height"].clip(lower=1)
    df = df[(df["x_norm"] >= 0.0) & (df["x_norm"] <= 1.0) & (df["y_norm"] >= 0.0) & (df["y_norm"] <= 1.0)].copy()

    df = df.sort_values(["subject_id", "image_id", "t"]).reset_index(drop=True)
    df["fixation_idx"] = df.groupby(["subject_id", "image_id"]).cumcount()
    df = df.rename(columns={"duration": "duration_ms"})

    return df


def normalize_duration(
    durations: Iterable[float],
    mode: str = "log_zscore",
) -> np.ndarray:
    """Normalize fixation duration with configurable strategy."""
    arr = np.asarray(list(durations), dtype=np.float32)
    if mode == "none":
        return arr
    if mode == "log":
        return np.log1p(arr)

    if mode == "zscore":
        mu = float(arr.mean()) if arr.size else 0.0
        std = float(arr.std()) if arr.size else 1.0
        std = std if std > 1e-6 else 1.0
        return (arr - mu) / std

    if mode == "log_zscore":
        arr = np.log1p(arr)
        mu = float(arr.mean()) if arr.size else 0.0
        std = float(arr.std()) if arr.size else 1.0
        std = std if std > 1e-6 else 1.0
        return (arr - mu) / std

    raise ValueError(f"Unsupported duration normalization mode: {mode}")
