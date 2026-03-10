"""Hydra entrypoint for training experiments."""

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

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics_classification import classification_metrics
from src.utils.io import ensure_dir, save_json
from src.utils.pipeline import build_dataloaders, build_lightning_module, ensure_prepared_data
from src.utils.plotting import plot_confusion_matrix, plot_roc_curve
from src.utils.seed import set_global_seed


def _build_logger(exp_name: str):
    log_root = ensure_dir(Path("outputs/logs"))
    use_wandb = False

    if use_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger

            return WandbLogger(project="DiffScanAuth", name=exp_name, save_dir=str(log_root))
        except Exception:
            return TensorBoardLogger(save_dir=str(log_root), name=exp_name)

    try:
        return TensorBoardLogger(save_dir=str(log_root), name=exp_name)
    except ModuleNotFoundError:
        return CSVLogger(save_dir=str(log_root), name=exp_name)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    exp_name = str(cfg.experiment.name)

    ensure_prepared_data(data_cfg)
    loaders = build_dataloaders(data_cfg, model_name=str(model_cfg["name"]), train_aug=True)
    lit = build_lightning_module(model_cfg)

    ckpt_dir = ensure_dir(Path("outputs/checkpoints") / exp_name)
    metric_dir = ensure_dir(Path("outputs/metrics"))
    pred_dir = ensure_dir(Path("outputs/predictions"))
    fig_dir = ensure_dir(Path("outputs/figures"))

    monitor = str(trainer_cfg.get("monitor", "val/auroc"))
    monitor_mode = str(trainer_cfg.get("monitor_mode", "max"))

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="{epoch:02d}-{val_auroc:.4f}",
        monitor=monitor,
        mode=monitor_mode,
        save_top_k=int(trainer_cfg.get("save_top_k", 1)),
        save_last=True,
    )
    early_cb = EarlyStopping(
        monitor=monitor,
        mode=monitor_mode,
        patience=int(trainer_cfg.get("early_stopping_patience", 5)),
    )

    precision = trainer_cfg.get("precision", "16-mixed")
    if not torch.cuda.is_available() and str(precision).startswith("16"):
        precision = "32-true"

    trainer = pl.Trainer(
        max_epochs=int(trainer_cfg.get("max_epochs", 8)),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", "auto"),
        precision=precision,
        deterministic=bool(trainer_cfg.get("deterministic", True)),
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 10)),
        callbacks=[checkpoint_cb, early_cb],
        logger=_build_logger(exp_name),
        fast_dev_run=bool(trainer_cfg.get("fast_dev_run", False)),
    )

    trainer.fit(lit, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])
    ckpt_path = checkpoint_cb.best_model_path if checkpoint_cb.best_model_path else None
    test_out = trainer.test(lit, dataloaders=loaders["test"], ckpt_path="best" if ckpt_path else None)

    metrics_payload: dict = {
        "experiment": exp_name,
        "checkpoint": ckpt_path,
        "trainer_test": test_out[0] if test_out else {},
    }

    pred_df = pd.DataFrame(lit.test_outputs)
    pred_path = pred_dir / f"{exp_name}_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    if not pred_df.empty and {"label", "prob"}.issubset(pred_df.columns):
        y_true = pred_df["label"].to_numpy()
        y_prob = pred_df["prob"].to_numpy()
        cls_metrics = classification_metrics(y_true=y_true, y_prob=y_prob)
        metrics_payload["classification"] = cls_metrics

        y_pred = (y_prob >= 0.5).astype(int)
        plot_confusion_matrix(y_true, y_pred, fig_dir / f"{exp_name}_confusion_matrix.png")
        plot_roc_curve(y_true, y_prob, fig_dir / f"{exp_name}_roc_curve.png")

        if {"coord_mae", "dur_mae"}.issubset(pred_df.columns):
            metrics_payload["gaze"] = {
                "fixation_position_mae": float(pred_df["coord_mae"].mean()),
                "duration_mae": float(pred_df["dur_mae"].mean()),
            }

    metrics_path = metric_dir / f"{exp_name}_test_metrics.json"
    save_json(metrics_payload, metrics_path)

    print(f"Training completed: {exp_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Predictions: {pred_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
