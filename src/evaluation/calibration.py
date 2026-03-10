"""Calibration metrics for binary classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute ECE with equally spaced confidence bins."""
    y_true = y_true.astype(np.float32)
    y_prob = y_prob.astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (y_prob >= lo) & (y_prob <= hi)
        else:
            m = (y_prob >= lo) & (y_prob < hi)
        if not np.any(m):
            continue
        acc = float((y_true[m] == (y_prob[m] >= 0.5)).mean())
        conf = float(y_prob[m].mean())
        ece += abs(acc - conf) * (m.mean())
    return float(ece)


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return Brier score."""
    return float(brier_score_loss(y_true.astype(int), y_prob.astype(float)))
