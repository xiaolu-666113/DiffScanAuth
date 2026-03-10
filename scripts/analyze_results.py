"""Analyze prediction files and produce metrics + figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

MPL_DIR = Path("outputs/.mplconfig")
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR.resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")
CACHE_DIR = Path("outputs/.cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR.resolve()))

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics_classification import classification_metrics
from src.utils.io import ensure_dir, save_json
from src.utils.plotting import plot_confusion_matrix, plot_roc_curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prediction CSV files")
    parser.add_argument("--predictions-dir", type=str, default="outputs/predictions")
    parser.add_argument("--metrics-dir", type=str, default="outputs/metrics")
    parser.add_argument("--figures-dir", type=str, default="outputs/figures")
    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    metric_dir = ensure_dir(args.metrics_dir)
    fig_dir = ensure_dir(args.figures_dir)

    pred_files = sorted(pred_dir.glob("*.csv"))
    if not pred_files:
        print("No prediction files found.")
        return

    summary_rows = []
    for pf in pred_files:
        df = pd.read_csv(pf)
        if not {"label", "prob"}.issubset(df.columns):
            continue
        y_true = df["label"].to_numpy()
        y_prob = df["prob"].to_numpy()
        metrics = classification_metrics(y_true=y_true, y_prob=y_prob)

        stem = pf.stem
        save_json(metrics, metric_dir / f"{stem}_analysis.json")
        plot_confusion_matrix(y_true, (y_prob >= 0.5).astype(int), fig_dir / f"{stem}_confusion_matrix.png")
        plot_roc_curve(y_true, y_prob, fig_dir / f"{stem}_roc_curve.png")

        summary_rows.append({"file": pf.name, **{k: v for k, v in metrics.items() if isinstance(v, float)}})

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        out_summary = metric_dir / "analysis_summary.csv"
        summary.to_csv(out_summary, index=False)
        print(f"Saved summary: {out_summary}")
    else:
        print("No valid prediction files with label/prob columns found.")


if __name__ == "__main__":
    main()
