"""Generate qualitative scanpath comparison figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir
from src.utils.plotting import plot_scanpath_comparison, plot_scanpath_overlay


def _load_points(cell: str) -> np.ndarray:
    if not isinstance(cell, str) or not cell:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.asarray(json.loads(cell), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.squeeze(0)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate qualitative scanpath figures")
    parser.add_argument("--ours", type=str, required=True, help="Prediction CSV from DiffScanAuth")
    parser.add_argument("--no-gaze", type=str, default="", help="Prediction CSV from SeqDet w/o gaze")
    parser.add_argument("--out-dir", type=str, default="outputs/figures/qualitative")
    parser.add_argument("--num-examples", type=int, default=5)
    args = parser.parse_args()

    ours_df = pd.read_csv(args.ours)
    no_gaze_df = pd.read_csv(args.no_gaze) if args.no_gaze else pd.DataFrame()
    out_dir = ensure_dir(args.out_dir)

    for idx, row in ours_df.head(args.num_examples).iterrows():
        image_path = row["image_path"]
        human_points = _load_points(row.get("human_scanpath_json", "[]"))
        ours_points = _load_points(row.get("pred_scanpath_json", "[]"))
        no_gaze_points = None
        if not no_gaze_df.empty:
            match = no_gaze_df[no_gaze_df["image_id"] == row["image_id"]]
            if len(match) > 0:
                no_gaze_points = _load_points(match.iloc[0].get("pred_scanpath_json", "[]"))

        plot_scanpath_overlay(
            image_path=image_path,
            points=ours_points,
            out_path=Path(out_dir) / f"{idx:02d}_{row['image_id']}_ours_overlay.png",
            title=f"Ours: {row['image_id']}",
        )
        plot_scanpath_comparison(
            image_path=image_path,
            human_points=human_points,
            ours_points=ours_points,
            no_gaze_points=no_gaze_points,
            out_path=Path(out_dir) / f"{idx:02d}_{row['image_id']}_comparison.png",
            title=f"Human vs Ours vs No-Gaze ({row['image_id']})",
        )

    print(f"Saved qualitative figures under: {out_dir}")


if __name__ == "__main__":
    main()
