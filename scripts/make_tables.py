"""Aggregate experiment metrics into compact result tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build summary tables from metrics JSON files")
    parser.add_argument("--metrics-dir", type=str, default="outputs/metrics")
    parser.add_argument("--out-csv", type=str, default="outputs/metrics/paper_results_table.csv")
    parser.add_argument("--out-json", type=str, default="outputs/metrics/paper_results_table.json")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    rows = []
    for path in sorted(metrics_dir.glob("*_metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        row = {"file": path.name, "experiment": payload.get("experiment", path.stem)}
        row.update(payload.get("classification", {}))
        row.update(payload.get("gaze", {}))
        rows.append(row)

    if not rows:
        print("No metrics files found.")
        return

    df = pd.DataFrame(rows).sort_values("experiment")
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    ensure_dir(out_csv.parent)
    ensure_dir(out_json.parent)
    df.to_csv(out_csv, index=False)
    df.to_json(out_json, orient="records", indent=2)
    print(f"Saved summary table: {out_csv}")
    print(f"Saved summary json: {out_json}")


if __name__ == "__main__":
    main()
