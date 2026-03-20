"""Loss helpers for sequential gaze-aware models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)


def _cls_loss(logits: torch.Tensor, label: torch.Tensor, pos_weight: float | None = None) -> torch.Tensor:
    if pos_weight is not None and pos_weight > 0:
        weight = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, label, pos_weight=weight)
    return F.binary_cross_entropy_with_logits(logits, label)


def _gaze_supervision_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mask = batch["mask"].float()
    patch_logits = outputs["patch_logits"]
    target_patch = batch["patch_idx"].long()
    patch_loss = _masked_mean(
        F.cross_entropy(
            patch_logits.reshape(-1, patch_logits.size(-1)),
            target_patch.reshape(-1),
            reduction="none",
        ).reshape_as(mask),
        mask,
    )

    coord_loss = _masked_mean(
        F.smooth_l1_loss(outputs["coord_pred"], batch["fix_xy"].float(), reduction="none").mean(dim=-1),
        mask,
    )
    delta_loss = _masked_mean(
        F.smooth_l1_loss(outputs["delta_pred"], batch["fix_delta"].float(), reduction="none").mean(dim=-1),
        mask,
    )
    dur_loss = _masked_mean(
        F.l1_loss(outputs["dur_pred"], batch["fix_dur"].float(), reduction="none"),
        mask,
    )
    return {
        "loss_patch": patch_loss,
        "loss_coord": coord_loss,
        "loss_delta": delta_loss,
        "loss_dur": dur_loss,
        "loss_gaze": patch_loss + coord_loss + delta_loss + dur_loss,
    }


def _stop_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], stop_mode: str) -> torch.Tensor:
    if stop_mode != "learned_stop":
        return torch.zeros((), device=batch["mask"].device)
    mask = batch["mask"].float()
    b, _ = mask.shape
    stop_target = torch.zeros_like(mask)
    lengths = mask.sum(dim=1).long().clamp(min=1) - 1
    stop_target[torch.arange(b, device=mask.device), lengths] = 1.0
    return _masked_mean(F.binary_cross_entropy_with_logits(outputs["stop_logits"], stop_target, reduction="none"), mask)


def _align_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], align_loss: str) -> torch.Tensor:
    mask = batch["mask"].float()
    patch_prob = torch.softmax(outputs["patch_logits"], dim=-1)
    weight = mask / (mask.sum(dim=1, keepdim=True) + 1e-6)
    pred_dist = (patch_prob * weight.unsqueeze(-1)).sum(dim=1)
    target_dist = batch["patch_dist"].float()
    if align_loss == "kl":
        return F.kl_div((pred_dist + 1e-6).log(), target_dist + 1e-6, reduction="batchmean")
    return F.mse_loss(pred_dist, target_dist)


def _distill_loss(outputs: dict[str, torch.Tensor], teacher_outputs: dict[str, torch.Tensor] | None, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if teacher_outputs is None:
        return torch.zeros((), device=batch["mask"].device)
    mask = batch["mask"].float()
    student_log_prob = torch.log_softmax(outputs["patch_logits"], dim=-1)
    teacher_prob = torch.softmax(teacher_outputs["patch_logits"].detach(), dim=-1)
    patch_kl = _masked_mean(
        F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1),
        mask,
    )
    coord = _masked_mean(
        F.mse_loss(outputs["coord_pred"], teacher_outputs["coord_pred"].detach(), reduction="none").mean(dim=-1),
        mask,
    )
    delta = _masked_mean(
        F.mse_loss(outputs["delta_pred"], teacher_outputs["delta_pred"].detach(), reduction="none").mean(dim=-1),
        mask,
    )
    dur = _masked_mean(
        F.l1_loss(outputs["dur_pred"], teacher_outputs["dur_pred"].detach(), reduction="none"),
        mask,
    )
    return patch_kl + coord + delta + dur


def _sequence_efficiency_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    step_probs = torch.sigmoid(outputs["step_logits"])
    final_prob = torch.sigmoid(outputs["cls_logits"])
    label = batch["label"].float()
    mask = batch["mask"].float()

    correctness_reward = label * final_prob + (1.0 - label) * (1.0 - final_prob)
    info_gain = F.relu(step_probs[:, 1:] - step_probs[:, :-1]).mean(dim=1)
    decision_steps = outputs["decision_steps"].float() / mask.size(1)
    step_penalty = decision_steps
    return -(correctness_reward + 0.5 * info_gain - 0.25 * step_penalty).mean()


def compute_seq_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
    pos_weight: float | None = None,
    stop_mode: str = "learned_stop",
    align_loss: str = "mse",
    use_gaze_supervision: bool = True,
    teacher_outputs: dict[str, torch.Tensor] | None = None,
    use_teacher_distill: bool = True,
    use_rl: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute the final paper-aligned loss bundle."""
    label = batch["label"].float()
    cls = _cls_loss(outputs["cls_logits"], label, pos_weight=pos_weight)
    gaze = _gaze_supervision_losses(outputs, batch) if use_gaze_supervision else {
        "loss_patch": torch.zeros((), device=label.device),
        "loss_coord": torch.zeros((), device=label.device),
        "loss_delta": torch.zeros((), device=label.device),
        "loss_dur": torch.zeros((), device=label.device),
        "loss_gaze": torch.zeros((), device=label.device),
    }
    stop = _stop_loss(outputs, batch, stop_mode=stop_mode)
    align = _align_loss(outputs, batch, align_loss=align_loss) if use_gaze_supervision else torch.zeros((), device=label.device)
    distill = _distill_loss(outputs, teacher_outputs, batch) if use_teacher_distill else torch.zeros((), device=label.device)
    rl = _sequence_efficiency_loss(outputs, batch) if use_rl else torch.zeros((), device=label.device)

    total = (
        weights.get("cls", 1.0) * cls
        + weights.get("gaze", 1.0) * gaze["loss_gaze"]
        + weights.get("distill", 0.5) * distill
        + weights.get("align", 0.1) * align
        + weights.get("stop", 0.2) * stop
        + weights.get("rl", 0.0) * rl
    )

    return {
        "loss": total,
        "loss_cls": cls,
        "loss_patch": gaze["loss_patch"],
        "loss_coord": gaze["loss_coord"],
        "loss_delta": gaze["loss_delta"],
        "loss_dur": gaze["loss_dur"],
        "loss_gaze": gaze["loss_gaze"],
        "loss_distill": distill,
        "loss_align": align,
        "loss_stop": stop,
        "loss_rl": rl,
    }


def sequence_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
    pos_weight: float | None = None,
    stop_mode: str = "fixed_k",
    align_loss: str = "mse",
) -> dict[str, torch.Tensor]:
    """Backward-compatible wrapper for older sequential experiments."""
    return compute_seq_losses(
        outputs=outputs,
        batch=batch,
        weights={
            "cls": weights.get("cls", 1.0),
            "gaze": weights.get("patch", 0.5) + weights.get("coord", 0.5) + weights.get("dur", 0.2),
            "distill": 0.0,
            "align": weights.get("align", 0.1),
            "stop": weights.get("stop", 0.2),
            "rl": 0.0,
        },
        pos_weight=pos_weight,
        stop_mode=stop_mode,
        align_loss=align_loss,
        use_gaze_supervision=True,
        teacher_outputs=None,
        use_teacher_distill=False,
        use_rl=False,
    )
