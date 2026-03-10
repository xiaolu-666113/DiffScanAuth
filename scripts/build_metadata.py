"""Build standardized metadata.csv by scanning raw images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.pipeline import run_build_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build metadata.csv")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--metadata-csv", type=str, default="data/processed/metadata.csv")
    parser.add_argument("--inspect-report-json", type=str, default="data/processed/dataset_inspection.json")
    parser.add_argument("--allow-synthetic", action="store_true", default=True)
    parser.add_argument("--synthetic-num-images", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_cfg = {
        "raw_dir": args.raw_dir,
        "processed_dir": args.processed_dir,
        "metadata_csv": args.metadata_csv,
        "inspect_report_json": args.inspect_report_json,
        "allow_synthetic": args.allow_synthetic,
        "synthetic_num_images": args.synthetic_num_images,
        "synthetic_seed": args.seed,
    }
    df = run_build_metadata(data_cfg)
    print(f"Saved metadata: {args.metadata_csv}, rows={len(df)}")


if __name__ == "__main__":
    main()
