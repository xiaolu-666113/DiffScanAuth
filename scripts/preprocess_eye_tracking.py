"""Standardize eye-tracking records and build processed_fixations.csv."""

from __future__ import annotations

import argparse
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

from src.utils.pipeline import run_preprocess_eye_tracking


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess eye-tracking data")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--metadata-csv", type=str, default="data/processed/metadata.csv")
    parser.add_argument("--eye-tracking-csv", type=str, default="data/processed/eye_tracking.csv")
    parser.add_argument("--processed-fixations-csv", type=str, default="data/processed/processed_fixations.csv")
    parser.add_argument("--duration-norm-mode", type=str, default="log_zscore")
    parser.add_argument("--min-duration-ms", type=float, default=40.0)
    parser.add_argument("--max-duration-ms", type=float, default=2000.0)
    parser.add_argument("--patch-grid-size", type=int, default=24)
    parser.add_argument("--allow-synthetic", action="store_true", default=True)
    parser.add_argument("--force-rebuild-eye-tracking", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_cfg = {
        "raw_dir": args.raw_dir,
        "metadata_csv": args.metadata_csv,
        "eye_tracking_csv": args.eye_tracking_csv,
        "processed_fixations_csv": args.processed_fixations_csv,
        "duration_norm_mode": args.duration_norm_mode,
        "min_duration_ms": args.min_duration_ms,
        "max_duration_ms": args.max_duration_ms,
        "patch_grid_size": args.patch_grid_size,
        "allow_synthetic": args.allow_synthetic,
        "force_rebuild_eye_tracking": args.force_rebuild_eye_tracking,
        "synthetic_seed": args.seed,
    }
    df = run_preprocess_eye_tracking(data_cfg)
    print(f"Saved processed fixations: {args.processed_fixations_csv}, rows={len(df)}")


if __name__ == "__main__":
    main()
