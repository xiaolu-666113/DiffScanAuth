"""Create leakage-safe train/val/test splits and split report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MPL_DIR = Path("outputs/.mplconfig")
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR.resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")
CACHE_DIR = Path("outputs/.cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR.resolve()))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.pipeline import run_make_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Make train/val/test splits")
    parser.add_argument("--metadata-csv", type=str, default="data/processed/metadata.csv")
    parser.add_argument("--processed-fixations-csv", type=str, default="data/processed/processed_fixations.csv")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--split-report-json", type=str, default="data/splits/split_report.json")
    parser.add_argument("--heatmap-dir", type=str, default="data/processed/heatmaps")
    parser.add_argument("--heatmap-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    data_cfg = {
        "metadata_csv": args.metadata_csv,
        "processed_fixations_csv": args.processed_fixations_csv,
        "splits_dir": args.splits_dir,
        "split_report_json": args.split_report_json,
        "heatmap_dir": args.heatmap_dir,
        "heatmap_size": args.heatmap_size,
        "split": {
            "seed": args.seed,
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
    }

    report = run_make_splits(data_cfg)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
