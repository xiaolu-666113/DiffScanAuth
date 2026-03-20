"""Run the full experiment suite sequentially."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


EXPERIMENTS = [
    "exp_vit_b16",
    "exp_aide_style",
    "exp_vit_heatmap",
    "exp_seqdet_no_gaze",
    "exp_diffscanauth",
    "ablation_no_gaze_supervision",
    "ablation_heatmap_instead_of_scanpath",
    "ablation_no_teacher_distillation",
    "ablation_no_local_stream",
    "ablation_fixed_k",
    "control_shuffled_gaze",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all paper experiments sequentially")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Optionally run only the first N experiments")
    args = parser.parse_args()

    experiments = EXPERIMENTS[: args.limit] if args.limit > 0 else EXPERIMENTS
    root = Path(__file__).resolve().parents[1]

    for exp in experiments:
        cmd = [
            sys.executable,
            str(root / "scripts" / "train.py"),
            f"experiment={exp}",
            f"trainer.max_epochs={args.epochs}",
            "data.loader.num_workers=0",
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=root)


if __name__ == "__main__":
    main()
