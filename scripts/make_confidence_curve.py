"""Create a confidence-over-time figure from sequence prediction CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.plotting import plot_confidence_over_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot confidence-over-time from prediction CSV")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    df = pd.read_csv(pred_path)
    if "step_probs_json" not in df.columns:
        raise ValueError(f"{pred_path} does not contain step_probs_json")

    sequences = [json.loads(x) for x in df["step_probs_json"].dropna().tolist()]
    out = Path(args.out) if args.out else Path("outputs/figures") / f"{pred_path.stem}_confidence_curve.png"
    plot_confidence_over_time(sequences, out, title=pred_path.stem.replace("_", " "))
    print(f"Saved confidence curve: {out}")


if __name__ == "__main__":
    main()
