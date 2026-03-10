"""Heatmap generation from fixation coordinates."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils.io import ensure_dir


def gaussian_heatmap(
    coords_xy_norm: np.ndarray,
    size: int = 96,
    sigma: float = 2.5,
) -> np.ndarray:
    """Create a normalized heatmap from normalized XY coordinates."""
    heat = np.zeros((size, size), dtype=np.float32)
    if coords_xy_norm.size == 0:
        return heat

    xs = np.clip((coords_xy_norm[:, 0] * (size - 1)).astype(int), 0, size - 1)
    ys = np.clip((coords_xy_norm[:, 1] * (size - 1)).astype(int), 0, size - 1)
    for x, y in zip(xs, ys):
        heat[y, x] += 1.0

    k = int(max(3, round(sigma * 6)))
    if k % 2 == 0:
        k += 1
    heat = cv2.GaussianBlur(heat, (k, k), sigmaX=sigma, sigmaY=sigma)
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat.astype(np.float32)


def build_image_heatmaps(
    fix_df: pd.DataFrame,
    split: str,
    size: int = 96,
    sigma: float = 2.5,
) -> dict[str, np.ndarray]:
    """Build per-image heatmaps for a split."""
    sub = fix_df[fix_df["split"] == split].copy()
    heatmaps: dict[str, np.ndarray] = {}
    for image_id, g in sub.groupby("image_id"):
        coords = g[["x_norm", "y_norm"]].to_numpy(dtype=np.float32)
        heatmaps[str(image_id)] = gaussian_heatmap(coords, size=size, sigma=sigma)
    return heatmaps


def save_heatmaps(heatmaps: dict[str, np.ndarray], out_dir: str | Path) -> None:
    """Persist heatmaps as .npy files keyed by image_id."""
    out = ensure_dir(out_dir)
    for image_id, arr in heatmaps.items():
        np.save(out / f"{image_id}.npy", arr)
