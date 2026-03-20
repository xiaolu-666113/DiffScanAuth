"""Evaluate a trained checkpoint on selected split."""

from __future__ import annotations

import argparse
import json
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

import pandas as pd
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics_classification import classification_metrics
from src.utils.io import ensure_dir, save_json
from src.utils.pipeline import build_dataloaders, build_lightning_module, ensure_prepared_data
from src.utils.plotting import plot_confusion_matrix, plot_confidence_over_time, plot_pr_curve, plot_roc_curve
from src.utils.seed import set_global_seed


def load_cfg(experiment: str, overrides: list[str]):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}"] + overrides)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint")
    parser.add_argument("--experiment", type=str, default="exp_diffscanauth")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args, unknown = parser.parse_known_args()

    cfg = load_cfg(args.experiment, unknown)
    set_global_seed(int(cfg.seed))

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    ensure_prepared_data(data_cfg)

    loaders = build_dataloaders(data_cfg, model_cfg=model_cfg, train_aug=False)
    lit = build_lightning_module(model_cfg)

    trainer = pl.Trainer(logger=False, enable_checkpointing=False, accelerator="auto", devices="auto")
    out = trainer.test(lit, dataloaders=loaders[args.split], ckpt_path=args.ckpt)

    pred_df = pd.DataFrame(lit.test_outputs)
    exp_name = str(cfg.experiment.name)
    suffix = f"{exp_name}_{args.split}"

    pred_dir = ensure_dir("outputs/predictions")
    fig_dir = ensure_dir("outputs/figures")
    metric_dir = ensure_dir("outputs/metrics")

    pred_path = pred_dir / f"{suffix}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    payload = {"trainer_test": out[0] if out else {}, "checkpoint": args.ckpt, "split": args.split}
    if not pred_df.empty and {"label", "prob"}.issubset(pred_df.columns):
        y_true = pred_df["label"].to_numpy()
        y_prob = pred_df["prob"].to_numpy()
        payload["classification"] = classification_metrics(y_true, y_prob)
        plot_confusion_matrix(y_true, (y_prob >= 0.5).astype(int), fig_dir / f"{suffix}_confusion_matrix.png")
        plot_roc_curve(y_true, y_prob, fig_dir / f"{suffix}_roc_curve.png")
        plot_pr_curve(y_true, y_prob, fig_dir / f"{suffix}_pr_curve.png")
        if {"coord_mae", "dur_mae", "decision_steps", "stop_step_error"}.issubset(pred_df.columns):
            payload["gaze"] = {
                "fixation_position_mae": float(pred_df["coord_mae"].mean()),
                "duration_mae": float(pred_df["dur_mae"].mean()),
                "avg_decision_steps": float(pred_df["decision_steps"].mean()),
                "stop_step_error": float(pred_df["stop_step_error"].mean()),
            }
        if "step_probs_json" in pred_df.columns:
            sequences = [json.loads(x) for x in pred_df["step_probs_json"].dropna().tolist()]
            plot_confidence_over_time(sequences, fig_dir / f"{suffix}_confidence_curve.png", title=f"{suffix} Confidence")

    out_path = metric_dir / f"{suffix}_metrics.json"
    save_json(payload, out_path)

    print(f"Evaluation finished: {out_path}")
    print(f"Predictions: {pred_path}")


if __name__ == "__main__":
    main()
