"""Classification metrics utilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluation.calibration import brier, expected_calibration_error


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | list[list[int]]]:
    """Compute binary classification metrics from probabilities."""
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    unique = np.unique(y_true)
    has_both_classes = len(unique) > 1
    positive_count = int((y_true == 1).sum())

    out: dict[str, float | list[list[int]]] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auprc": float(average_precision_score(y_true, y_prob))
        if positive_count not in {0, len(y_true)}
        else float(positive_count == len(y_true)),
        "ece": float(expected_calibration_error(y_true, y_prob)),
        "brier": float(brier(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

    try:
        if not has_both_classes:
            raise ValueError("single-class targets")
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auroc"] = 0.5

    return out
