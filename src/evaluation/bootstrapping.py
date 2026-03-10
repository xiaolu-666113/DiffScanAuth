"""Bootstrap helper for confidence intervals."""

from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrap mean and confidence interval for a metric."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(float(metric_fn(y_true[idx], y_prob[idx])))
    arr = np.asarray(vals)
    lo = float(np.quantile(arr, alpha / 2))
    hi = float(np.quantile(arr, 1 - alpha / 2))
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci_low": lo,
        "ci_high": hi,
    }
