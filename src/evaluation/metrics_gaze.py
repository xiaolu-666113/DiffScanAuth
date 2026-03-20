"""Gaze/scanpath metric utilities."""

from __future__ import annotations

import numpy as np


def fixation_position_mae(pred_xy: np.ndarray, true_xy: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute error on fixation coordinates."""
    err = np.abs(pred_xy - true_xy).mean(axis=-1)
    denom = np.maximum(mask.sum(), 1.0)
    return float((err * mask).sum() / denom)


def fixation_euclidean_distance(pred_xy: np.ndarray, true_xy: np.ndarray, mask: np.ndarray) -> float:
    """Mean Euclidean distance on fixation positions."""
    err = np.sqrt(((pred_xy - true_xy) ** 2).sum(axis=-1))
    denom = np.maximum(mask.sum(), 1.0)
    return float((err * mask).sum() / denom)


def duration_mae(pred_dur: np.ndarray, true_dur: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute error on fixation duration (normalized scale)."""
    err = np.abs(pred_dur - true_dur)
    denom = np.maximum(mask.sum(), 1.0)
    return float((err * mask).sum() / denom)


def stop_step_error(pred_stop_step: np.ndarray, true_stop_step: np.ndarray) -> float:
    """MAE between predicted and true stop step index."""
    return float(np.abs(pred_stop_step - true_stop_step).mean())


def average_decision_steps(decision_steps: np.ndarray) -> float:
    """Average number of decision steps."""
    return float(np.asarray(decision_steps, dtype=np.float32).mean())


def scanpath_similarity(pred_xy: np.ndarray, true_xy: np.ndarray) -> float:
    """Simplified scanpath similarity based on mean Euclidean alignment."""
    pred = np.asarray(pred_xy, dtype=np.float32)
    true = np.asarray(true_xy, dtype=np.float32)
    length = min(len(pred), len(true))
    if length == 0:
        return 0.0
    return float(1.0 / (1.0 + np.sqrt(((pred[:length] - true[:length]) ** 2).sum(axis=-1)).mean()))


def dtw_distance(pred_xy: np.ndarray, true_xy: np.ndarray) -> float:
    """Simple DTW distance over fixation coordinates."""
    pred = np.asarray(pred_xy, dtype=np.float32)
    true = np.asarray(true_xy, dtype=np.float32)
    n, m = len(pred), len(true)
    if n == 0 or m == 0:
        return float(max(n, m))
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.sqrt(((pred[i - 1] - true[j - 1]) ** 2).sum())
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / max(n, m))
