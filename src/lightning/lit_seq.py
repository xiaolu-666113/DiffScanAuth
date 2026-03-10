"""Lightning module for sequential gaze-guided detector."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryAveragePrecision, BinaryF1Score

from src.models.modules.losses import sequence_losses
from src.models.seq_gaze_detector import SeqGazeDetector


class LitSeq(pl.LightningModule):
    """Train/eval wrapper for main sequential model."""

    def __init__(self, model_cfg: dict[str, Any], optim_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters({"model_cfg": model_cfg, "optim_cfg": optim_cfg})

        self.model = SeqGazeDetector(
            backbone_name=model_cfg.get("backbone_name", "convnext_tiny"),
            pretrained=bool(model_cfg.get("pretrained", True)),
            patch_grid_size=int(model_cfg.get("patch_grid_size", 24)),
            policy_hidden_dim=int(model_cfg.get("policy_hidden_dim", 256)),
            glimpse_dim=int(model_cfg.get("glimpse_dim", 256)),
            accumulator_hidden_dim=int(model_cfg.get("accumulator_hidden_dim", 256)),
            accumulator_backend=str(model_cfg.get("accumulator_backend", "gru")),
            stop_mode=str(model_cfg.get("stop_mode", "fixed_k")),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )

        self.loss_weights = {
            "cls": float(model_cfg.get("loss_cls", 1.0)),
            "patch": float(model_cfg.get("loss_patch", 0.5)),
            "coord": float(model_cfg.get("loss_coord", 0.5)),
            "dur": float(model_cfg.get("loss_dur", 0.2)),
            "stop": float(model_cfg.get("loss_stop", 0.2)),
            "align": float(model_cfg.get("loss_align", 0.1)),
        }
        self.stop_mode = str(model_cfg.get("stop_mode", "fixed_k"))
        self.align_loss = str(model_cfg.get("align_loss", "mse"))

        self.teacher_forcing_start = float(model_cfg.get("teacher_forcing_ratio", 1.0))
        self.teacher_forcing_end = float(model_cfg.get("teacher_forcing_final", self.teacher_forcing_start))

        self.lr = float(optim_cfg.get("lr", 3e-4))
        self.weight_decay = float(optim_cfg.get("weight_decay", 1e-4))
        self.pos_weight = optim_cfg.get("pos_weight", None)

        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()
        self.test_auroc = BinaryAUROC()
        self.test_auprc = BinaryAveragePrecision()

        self.test_outputs: list[dict[str, Any]] = []

    def _teacher_forcing_ratio(self) -> float:
        if self.trainer is None or self.trainer.max_epochs <= 1:
            return self.teacher_forcing_start
        progress = min(1.0, self.current_epoch / max(1, self.trainer.max_epochs - 1))
        return self.teacher_forcing_start + (self.teacher_forcing_end - self.teacher_forcing_start) * progress

    def _step(self, batch: dict[str, Any], train: bool) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        tf_ratio = self._teacher_forcing_ratio() if train else 0.0
        outputs = self.model(batch, teacher_forcing_ratio=tf_ratio)
        losses = sequence_losses(
            outputs=outputs,
            batch=batch,
            weights=self.loss_weights,
            pos_weight=float(self.pos_weight) if self.pos_weight is not None else None,
            stop_mode=self.stop_mode,
            align_loss=self.align_loss,
        )
        prob = torch.sigmoid(outputs["cls_logits"])
        label = batch["label"].float()
        return losses, prob, label

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        losses, _, _ = self._step(batch, train=True)
        self.log("train/loss", losses["loss"], on_epoch=True, prog_bar=True)
        self.log("train/loss_cls", losses["loss_cls"], on_epoch=True)
        self.log("train/loss_patch", losses["loss_patch"], on_epoch=True)
        self.log("train/loss_coord", losses["loss_coord"], on_epoch=True)
        self.log("train/loss_dur", losses["loss_dur"], on_epoch=True)
        if self.stop_mode == "learned_stop":
            self.log("train/loss_stop", losses["loss_stop"], on_epoch=True)
        self.log("train/loss_align", losses["loss_align"], on_epoch=True)
        self.log("train/teacher_forcing", torch.tensor(self._teacher_forcing_ratio(), device=self.device), on_epoch=True)
        return losses["loss"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        losses, prob, label = self._step(batch, train=False)
        self.val_acc(prob, label.int())
        self.val_f1(prob, label.int())
        self.val_auroc(prob, label.int())
        self.val_auprc(prob, label.int())

        self.log("val/loss", losses["loss"], on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_epoch=True)
        self.log("val/f1", self.val_f1, on_epoch=True)
        self.log("val/auroc", self.val_auroc, on_epoch=True, prog_bar=True)
        self.log("val/auprc", self.val_auprc, on_epoch=True)

    def on_test_epoch_start(self) -> None:
        self.test_outputs = []

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        losses, prob, label = self._step(batch, train=False)
        self.test_acc(prob, label.int())
        self.test_f1(prob, label.int())
        self.test_auroc(prob, label.int())
        self.test_auprc(prob, label.int())
        self.log("test/loss", losses["loss"], on_epoch=True)
        self.log("test/acc", self.test_acc, on_epoch=True)
        self.log("test/f1", self.test_f1, on_epoch=True)
        self.log("test/auroc", self.test_auroc, on_epoch=True)
        self.log("test/auprc", self.test_auprc, on_epoch=True)

        outputs = self.model(batch, teacher_forcing_ratio=0.0)
        coord_pred = outputs["coord_pred"].detach().cpu()
        dur_pred = outputs["dur_pred"].detach().cpu()
        mask = batch["mask"].detach().cpu()

        for i, iid in enumerate(batch["image_id"]):
            self.test_outputs.append(
                {
                    "image_id": iid,
                    "subject_id": batch["subject_id"][i],
                    "label": int(label[i].detach().cpu().item()),
                    "prob": float(prob[i].detach().cpu().item()),
                    "coord_mae": float((torch.abs(coord_pred[i] - batch["fix_xy"][i].detach().cpu()).mean(dim=-1) * mask[i]).sum().item() / (mask[i].sum().item() + 1e-6)),
                    "dur_mae": float((torch.abs(dur_pred[i] - batch["fix_dur"][i].detach().cpu()) * mask[i]).sum().item() / (mask[i].sum().item() + 1e-6)),
                }
            )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
