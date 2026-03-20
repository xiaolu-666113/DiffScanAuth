"""Inspect raw dataset structure and dump summary report."""

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

from src.datasets.dataset_adapter import inspect_and_dump


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dataset folder")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--out", type=str, default="data/processed/dataset_inspection.json")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = inspect_and_dump(args.raw_dir, args.out)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
