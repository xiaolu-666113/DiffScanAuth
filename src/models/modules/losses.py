"""Loss helpers for sequential gaze detector."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)


def sequence_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
    pos_weight: float | None = None,
    stop_mode: str = "fixed_k",
    align_loss: str = "mse",
) -> dict[str, torch.Tensor]:
    """Compute total loss and components for seq detector."""
    label = batch["label"].float()
    mask = batch["mask"].float()

    cls_logits = outputs["cls_logits"]
    if pos_weight is not None and pos_weight > 0:
        pw = torch.tensor([pos_weight], device=cls_logits.device, dtype=cls_logits.dtype)
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, label, pos_weight=pw)
    else:
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, label)

    patch_logits = outputs["patch_logits"]
    target_patch = batch["patch_idx"].long()
    ce = F.cross_entropy(
        patch_logits.reshape(-1, patch_logits.size(-1)),
        target_patch.reshape(-1),
        reduction="none",
    ).reshape_as(mask)
    patch_loss = _masked_mean(ce, mask)

    pred_xy = outputs["coord_pred"]
    tgt_xy = batch["fix_xy"].float()
    coord_raw = F.smooth_l1_loss(pred_xy, tgt_xy, reduction="none").mean(dim=-1)
    coord_loss = _masked_mean(coord_raw, mask)

    pred_dur = outputs["dur_pred"]
    tgt_dur = batch["fix_dur"].float()
    dur_raw = F.l1_loss(pred_dur, tgt_dur, reduction="none")
    dur_loss = _masked_mean(dur_raw, mask)

    stop_logits = outputs["stop_logits"]
    if stop_mode == "learned_stop":
        b, t = mask.shape
        stop_target = torch.zeros_like(mask)
        lengths = mask.sum(dim=1).long().clamp(min=1) - 1
        stop_target[torch.arange(b, device=mask.device), lengths] = 1.0
        stop_raw = F.binary_cross_entropy_with_logits(stop_logits, stop_target, reduction="none")
        stop_loss = _masked_mean(stop_raw, mask)
    else:
        stop_loss = torch.zeros((), device=mask.device)

    patch_prob = torch.softmax(patch_logits, dim=-1)
    mask_w = mask / (mask.sum(dim=1, keepdim=True) + 1e-6)
    pred_dist = (patch_prob * mask_w.unsqueeze(-1)).sum(dim=1)
    target_dist = batch["patch_dist"].float()

    if align_loss == "kl":
        align = F.kl_div((pred_dist + 1e-6).log(), target_dist + 1e-6, reduction="batchmean")
    else:
        align = F.mse_loss(pred_dist, target_dist)

    total = (
        weights.get("cls", 1.0) * cls_loss
        + weights.get("patch", 0.5) * patch_loss
        + weights.get("coord", 0.5) * coord_loss
        + weights.get("dur", 0.2) * dur_loss
        + weights.get("stop", 0.2) * stop_loss
        + weights.get("align", 0.1) * align
    )

    return {
        "loss": total,
        "loss_cls": cls_loss,
        "loss_patch": patch_loss,
        "loss_coord": coord_loss,
        "loss_dur": dur_loss,
        "loss_stop": stop_loss,
        "loss_align": align,
    }
