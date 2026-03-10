"""Quick smoke test for full data + model pipeline."""

from __future__ import annotations

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

import pytorch_lightning as pl
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.pipeline import (
    build_dataloaders,
    build_lightning_module,
    run_build_metadata,
    run_make_splits,
    run_preprocess_eye_tracking,
)
from src.utils.seed import set_global_seed


def load_yaml(path: str):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def run_data_pipeline(data_cfg: dict) -> None:
    run_build_metadata(data_cfg)
    run_preprocess_eye_tracking(data_cfg)
    run_make_splits(data_cfg)


def run_model_smoke(data_cfg: dict, model_cfg_path: str) -> None:
    model_cfg = load_yaml(model_cfg_path)
    model_name = model_cfg["name"]

    loaders = build_dataloaders(data_cfg, model_name=model_name, train_aug=False)
    lit = build_lightning_module(model_cfg)

    trainer = pl.Trainer(
        max_epochs=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(lit, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])
    trainer.test(lit, dataloaders=loaders["test"])


def main() -> None:
    set_global_seed(42)

    data_cfg = load_yaml(str(Path("configs/data/default.yaml")))
    data_cfg["loader"]["batch_size"] = 2
    data_cfg["loader"]["num_workers"] = 0

    run_data_pipeline(data_cfg)

    run_model_smoke(data_cfg, "configs/model/baseline_static.yaml")
    run_model_smoke(data_cfg, "configs/model/baseline_heatmap.yaml")
    run_model_smoke(data_cfg, "configs/model/seq_gaze_detector.yaml")

    print("Smoke test passed for data pipeline + 3 model families.")


if __name__ == "__main__":
    main()
