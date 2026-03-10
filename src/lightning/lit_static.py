"""Lightning module for static baseline model."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryAveragePrecision, BinaryF1Score

from src.models.baseline_static import BaselineStaticClassifier


class LitStatic(pl.LightningModule):
    """Train/eval wrapper for Baseline A."""

    def __init__(self, model_cfg: dict[str, Any], optim_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters({"model_cfg": model_cfg, "optim_cfg": optim_cfg})
        self.model = BaselineStaticClassifier(
            backbone_name=model_cfg.get("backbone_name", "convnext_tiny"),
            pretrained=bool(model_cfg.get("pretrained", True)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
        self.lr = float(optim_cfg.get("lr", 3e-4))
        self.weight_decay = float(optim_cfg.get("weight_decay", 1e-4))
        self.pos_weight = optim_cfg.get("pos_weight", None)

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()
        self.test_auroc = BinaryAUROC()
        self.test_auprc = BinaryAveragePrecision()

        self.test_outputs: list[dict[str, Any]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _bce(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if self.pos_weight is not None:
            pw = torch.tensor([float(self.pos_weight)], device=logits.device, dtype=logits.dtype)
            return F.binary_cross_entropy_with_logits(logits, label, pos_weight=pw)
        return F.binary_cross_entropy_with_logits(logits, label)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        logits = self(batch["image"])
        label = batch["label"].float()
        loss = self._bce(logits, label)
        prob = torch.sigmoid(logits)

        self.train_acc(prob, label.int())
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        logits = self(batch["image"])
        label = batch["label"].float()
        loss = self._bce(logits, label)
        prob = torch.sigmoid(logits)

        self.val_acc(prob, label.int())
        self.val_f1(prob, label.int())
        self.val_auroc(prob, label.int())
        self.val_auprc(prob, label.int())

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_epoch=True)
        self.log("val/auroc", self.val_auroc, on_epoch=True, prog_bar=True)
        self.log("val/auprc", self.val_auprc, on_epoch=True)

    def on_test_epoch_start(self) -> None:
        self.test_outputs = []

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        logits = self(batch["image"])
        label = batch["label"].float()
        loss = self._bce(logits, label)
        prob = torch.sigmoid(logits)

        self.test_acc(prob, label.int())
        self.test_f1(prob, label.int())
        self.test_auroc(prob, label.int())
        self.test_auprc(prob, label.int())

        self.log("test/loss", loss, on_epoch=True)
        self.log("test/acc", self.test_acc, on_epoch=True)
        self.log("test/f1", self.test_f1, on_epoch=True)
        self.log("test/auroc", self.test_auroc, on_epoch=True)
        self.log("test/auprc", self.test_auprc, on_epoch=True)

        image_ids = batch["image_id"]
        for iid, y, p in zip(image_ids, label.detach().cpu().tolist(), prob.detach().cpu().tolist()):
            self.test_outputs.append({"image_id": iid, "label": int(y), "prob": float(p)})

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return opt
