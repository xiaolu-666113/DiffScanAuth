"""Utilities for discretizing fixation coordinates into patch tokens."""

from __future__ import annotations

import numpy as np
import pandas as pd


def xy_to_patch_index(x_norm: float, y_norm: float, grid_size: int) -> int:
    """Convert normalized coordinates into flattened patch index."""
    x_bin = min(grid_size - 1, max(0, int(x_norm * grid_size)))
    y_bin = min(grid_size - 1, max(0, int(y_norm * grid_size)))
    return y_bin * grid_size + x_bin


def patch_index_to_center(index: int, grid_size: int) -> tuple[float, float]:
    """Return normalized center coordinate for a flattened patch index."""
    y_bin = index // grid_size
    x_bin = index % grid_size
    step = 1.0 / grid_size
    return (x_bin + 0.5) * step, (y_bin + 0.5) * step


def add_patch_tokens(df: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    """Add patch index column from normalized fixation coordinates."""
    out = df.copy()
    out["patch_index"] = [
        xy_to_patch_index(float(x), float(y), grid_size)
        for x, y in zip(out["x_norm"].to_numpy(), out["y_norm"].to_numpy())
    ]
    return out


def patch_histogram(patch_indices: np.ndarray, num_patches: int) -> np.ndarray:
    """Compute normalized histogram over patch tokens."""
    hist = np.bincount(patch_indices.astype(int), minlength=num_patches).astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist
