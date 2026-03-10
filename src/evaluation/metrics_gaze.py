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
