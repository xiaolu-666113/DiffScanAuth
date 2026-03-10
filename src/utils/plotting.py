"""Plotting helpers for evaluation reports."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

from src.utils.io import ensure_dir

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))
ensure_dir(Path(os.environ["MPLCONFIGDIR"]))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs") / ".cache"))
ensure_dir(Path(os.environ["XDG_CACHE_HOME"]))
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path) -> None:
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    out = Path(out_path)
    ensure_dir(out.parent)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: str | Path) -> None:
    """Save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    out = Path(out_path)
    ensure_dir(out.parent)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
