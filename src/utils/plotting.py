"""Plotting helpers for evaluation reports."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from src.utils.io import ensure_dir

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))
ensure_dir(Path(os.environ["MPLCONFIGDIR"]))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs") / ".cache"))
ensure_dir(Path(os.environ["XDG_CACHE_HOME"]))
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path) -> None:
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
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
    fig, ax = plt.subplots(figsize=(4, 4))
    if len(np.unique(y_true)) < 2:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax.text(0.5, 0.5, "ROC undefined\n(single-class targets)", ha="center", va="center")
    else:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label="ROC")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    out = Path(out_path)
    ensure_dir(out.parent)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: str | Path) -> None:
    """Save precision-recall curve."""
    fig, ax = plt.subplots(figsize=(4, 4))
    positive_count = int((np.asarray(y_true) == 1).sum())
    if positive_count in {0, len(y_true)}:
        baseline = float(positive_count == len(y_true))
        ax.plot([0, 1], [baseline, baseline], label="PR")
        ax.text(0.5, baseline, "PR degenerate\n(single-class targets)", ha="center", va="bottom")
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(recall, precision, label="PR")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    out = Path(out_path)
    ensure_dir(out.parent)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_confidence_over_time(
    sequences: list[list[float]],
    out_path: str | Path,
    title: str = "Confidence Over Time",
) -> None:
    """Plot mean confidence trajectory across samples."""
    if not sequences:
        return
    max_len = max(len(seq) for seq in sequences)
    arr = np.full((len(sequences), max_len), np.nan, dtype=np.float32)
    for i, seq in enumerate(sequences):
        arr[i, : len(seq)] = np.asarray(seq, dtype=np.float32)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    x = np.arange(1, max_len + 1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, mean, label="Mean confidence")
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend(loc="lower right")
    out = Path(out_path)
    ensure_dir(out.parent)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_scanpath_overlay(
    image_path: str | Path,
    points: np.ndarray,
    out_path: str | Path,
    title: str = "Scanpath Overlay",
    color: str = "tab:red",
) -> None:
    """Overlay one scanpath on top of an image."""
    image = np.asarray(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return
    pts_px = np.stack([pts[:, 0] * w, pts[:, 1] * h], axis=1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.plot(pts_px[:, 0], pts_px[:, 1], color=color, linewidth=2)
    ax.scatter(pts_px[:, 0], pts_px[:, 1], c=np.arange(len(pts_px)), cmap="viridis", s=35)
    ax.set_title(title)
    ax.axis("off")
    out = Path(out_path)
    ensure_dir(out.parent)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_scanpath_comparison(
    image_path: str | Path,
    human_points: np.ndarray,
    ours_points: np.ndarray,
    no_gaze_points: np.ndarray | None,
    out_path: str | Path,
    title: str = "Human vs Ours vs No-Gaze",
) -> None:
    """Overlay multiple scanpaths for qualitative comparison."""
    image = np.asarray(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)

    def _draw(points: np.ndarray, color: str, label: str) -> None:
        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0:
            return
        pts_px = np.stack([pts[:, 0] * w, pts[:, 1] * h], axis=1)
        ax.plot(pts_px[:, 0], pts_px[:, 1], color=color, linewidth=2, label=label)
        ax.scatter(pts_px[:, 0], pts_px[:, 1], color=color, s=18)

    _draw(human_points, "tab:blue", "Human")
    _draw(ours_points, "tab:red", "Ours")
    if no_gaze_points is not None:
        _draw(no_gaze_points, "tab:green", "No-Gaze")

    ax.set_title(title)
    ax.axis("off")
    ax.legend(loc="lower right")
    out = Path(out_path)
    ensure_dir(out.parent)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
