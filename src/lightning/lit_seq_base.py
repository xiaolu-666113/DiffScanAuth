"""Common Lightning utilities for sequential DiffScanAuth-family models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryAveragePrecision, BinaryF1Score

from src.models.modules.losses import compute_seq_losses


class LitSequentialBase(pl.LightningModule):
    """Shared Lightning logic for sequential search models."""

    def __init__(
        self,
        model: torch.nn.Module,
        model_cfg: dict[str, Any],
        optim_cfg: dict[str, Any],
        use_gaze_supervision: bool,
        use_teacher_distill: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.use_gaze_supervision = use_gaze_supervision
        self.use_teacher_distill = use_teacher_distill

        self.loss_weights = {
            "cls": float(model_cfg.get("loss_cls", 1.0)),
            "gaze": float(model_cfg.get("loss_gaze", 1.0)),
            "distill": float(model_cfg.get("loss_distill", 0.5)),
            "align": float(model_cfg.get("loss_align", 0.1)),
            "stop": float(model_cfg.get("loss_stop", 0.2)),
            "rl": float(model_cfg.get("loss_rl", 0.0)),
        }
        self.stop_mode = str(model_cfg.get("stop_mode", "learned_stop"))
        self.align_loss = str(model_cfg.get("align_loss", "mse"))
        self.training_stage = str(model_cfg.get("training_stage", "student"))
        self.use_rl = bool(model_cfg.get("use_rl", False))

        self.teacher_forcing_start = float(model_cfg.get("teacher_forcing_ratio", 1.0))
        self.teacher_forcing_end = float(model_cfg.get("teacher_forcing_final", self.teacher_forcing_start))
        self.scheduled_sampling_start = float(model_cfg.get("scheduled_sampling_ratio", 0.0))
        self.scheduled_sampling_end = float(model_cfg.get("scheduled_sampling_final", self.scheduled_sampling_start))

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
        self._maybe_load_checkpoints()
        self._maybe_freeze_teacher()

    def _load_model_weights(
        self,
        ckpt_path: str,
        allowed_prefixes: tuple[str, ...] | None = None,
        strict: bool = False,
    ) -> None:
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        payload = torch.load(path, map_location="cpu")
        raw_state = payload.get("state_dict", payload)
        filtered_state: dict[str, torch.Tensor] = {}
        for key, value in raw_state.items():
            if key.startswith("model."):
                target_key = key[len("model.") :]
            else:
                target_key = key
            if allowed_prefixes is not None and not any(target_key.startswith(prefix) for prefix in allowed_prefixes):
                continue
            filtered_state[target_key] = value

        if not filtered_state:
            raise RuntimeError(f"No compatible model weights found in checkpoint: {path}")

        incompatible = self.model.load_state_dict(filtered_state, strict=False)
        if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            raise RuntimeError(
                f"Strict checkpoint loading failed for {path}. "
                f"Missing keys: {incompatible.missing_keys[:8]}, "
                f"Unexpected keys: {incompatible.unexpected_keys[:8]}"
            )
        print(
            f"Loaded {len(filtered_state)} tensors from {path} "
            f"(missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)})"
        )

    def _maybe_load_checkpoints(self) -> None:
        init_from_ckpt = str(self.model_cfg.get("init_from_ckpt", "")).strip()
        teacher_ckpt = str(self.model_cfg.get("teacher_ckpt", "")).strip()
        strict_load = bool(self.model_cfg.get("strict_load", False))

        if init_from_ckpt:
            self._load_model_weights(init_from_ckpt, allowed_prefixes=None, strict=strict_load)

        if teacher_ckpt:
            teacher_module = getattr(self.model, "teacher", None)
            if teacher_module is None:
                raise RuntimeError("teacher_ckpt was provided, but this model has no teacher module")
            self._load_model_weights(teacher_ckpt, allowed_prefixes=("encoder.", "teacher."), strict=False)

    def _maybe_freeze_teacher(self) -> None:
        should_freeze = bool(self.model_cfg.get("freeze_teacher", False))
        teacher_module = getattr(self.model, "teacher", None)
        if not should_freeze or teacher_module is None or self.training_stage == "teacher":
            return
        teacher_module.requires_grad_(False)

    def _anneal(self, start: float, end: float) -> float:
        if self.trainer is None or self.trainer.max_epochs <= 1:
            return start
        progress = min(1.0, self.current_epoch / max(1, self.trainer.max_epochs - 1))
        return start + (end - start) * progress

    def _teacher_forcing_ratio(self) -> float:
        return self._anneal(self.teacher_forcing_start, self.teacher_forcing_end)

    def _scheduled_sampling_ratio(self) -> float:
        return self._anneal(self.scheduled_sampling_start, self.scheduled_sampling_end)

    def _teacher_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        teacher_outputs = self.model.forward_teacher(batch)
        return {
            "loss": teacher_outputs["teacher_loss"],
            "loss_teacher": teacher_outputs["teacher_loss"],
        }

    def _student_step(self, batch: dict[str, Any], train: bool) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        tf_ratio = self._teacher_forcing_ratio() if train else 0.0
        ss_ratio = self._scheduled_sampling_ratio() if train else 1.0
        outputs = self.model(
            batch,
            teacher_forcing_ratio=tf_ratio,
            scheduled_sampling_ratio=ss_ratio,
            use_gaze_inputs=self.use_gaze_supervision,
        )
        losses = compute_seq_losses(
            outputs=outputs,
            batch=batch,
            weights=self.loss_weights,
            pos_weight=float(self.pos_weight) if self.pos_weight is not None else None,
            stop_mode=self.stop_mode,
            align_loss=self.align_loss,
            use_gaze_supervision=self.use_gaze_supervision,
            teacher_outputs=outputs.get("teacher_outputs"),
            use_teacher_distill=self.use_teacher_distill,
            use_rl=self.use_rl and self.training_stage == "refine",
        )
        prob = torch.sigmoid(outputs["cls_logits"])
        label = batch["label"].float()
        return losses, prob, label, outputs

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        if self.training_stage == "teacher":
            losses = self._teacher_step(batch)
            self.log("train/loss", losses["loss"], on_epoch=True, prog_bar=True)
            self.log("train/loss_teacher", losses["loss_teacher"], on_epoch=True)
            return losses["loss"]

        losses, _, _, _ = self._student_step(batch, train=True)
        for key, value in losses.items():
            self.log(f"train/{key}", value, on_epoch=True, prog_bar=(key == "loss"))
        self.log("train/teacher_forcing", torch.tensor(self._teacher_forcing_ratio(), device=self.device), on_epoch=True)
        self.log("train/scheduled_sampling", torch.tensor(self._scheduled_sampling_ratio(), device=self.device), on_epoch=True)
        return losses["loss"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        if self.training_stage == "teacher":
            losses = self._teacher_step(batch)
            self.log("val/loss", losses["loss"], on_epoch=True, prog_bar=True)
            return

        losses, prob, label, _ = self._student_step(batch, train=False)
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
        if self.training_stage == "teacher":
            losses = self._teacher_step(batch)
            self.log("test/loss", losses["loss"], on_epoch=True)
            return

        losses, prob, label, outputs = self._student_step(batch, train=False)
        self.test_acc(prob, label.int())
        self.test_f1(prob, label.int())
        self.test_auroc(prob, label.int())
        self.test_auprc(prob, label.int())
        self.log("test/loss", losses["loss"], on_epoch=True)
        self.log("test/acc", self.test_acc, on_epoch=True)
        self.log("test/f1", self.test_f1, on_epoch=True)
        self.log("test/auroc", self.test_auroc, on_epoch=True)
        self.log("test/auprc", self.test_auprc, on_epoch=True)

        pred_xy = outputs["coord_pred"].detach().cpu()
        pred_dur = outputs["dur_pred"].detach().cpu()
        step_probs = outputs["step_probs"].detach().cpu()
        stop_probs = torch.sigmoid(outputs["stop_logits"]).detach().cpu()
        mask = batch["mask"].detach().cpu()
        decision_steps = outputs["decision_steps"].detach().cpu()
        stop_target = mask.sum(dim=1).long().clamp(min=1) - 1

        for i, iid in enumerate(batch["image_id"]):
            coord_mae = float((torch.abs(pred_xy[i] - batch["fix_xy"][i].detach().cpu()).mean(dim=-1) * mask[i]).sum().item() / (mask[i].sum().item() + 1e-6))
            dur_mae = float((torch.abs(pred_dur[i] - batch["fix_dur"][i].detach().cpu()) * mask[i]).sum().item() / (mask[i].sum().item() + 1e-6))
            self.test_outputs.append(
                {
                    "image_id": iid,
                    "image_path": batch["image_path"][i],
                    "subject_id": batch["subject_id"][i],
                    "label": int(label[i].detach().cpu().item()),
                    "prob": float(prob[i].detach().cpu().item()),
                    "decision_steps": int(decision_steps[i].item() + 1),
                    "stop_step_error": float(abs(int(decision_steps[i].item()) - int(stop_target[i].item()))),
                    "coord_mae": coord_mae,
                    "dur_mae": dur_mae,
                    "step_probs_json": json.dumps(step_probs[i].tolist()),
                    "stop_probs_json": json.dumps(stop_probs[i].tolist()),
                    "pred_scanpath_json": json.dumps(pred_xy[i].tolist()),
                    "human_scanpath_json": json.dumps(batch["fix_xy"][i].detach().cpu().tolist()),
                    "shuffled_from_image_id": batch.get("shuffled_from_image_id", [""] * len(batch["image_id"]))[i],
                }
            )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
