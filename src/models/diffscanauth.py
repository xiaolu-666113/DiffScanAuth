"""Final paper-aligned DiffScanAuth model."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn

from src.models.modules.accumulator import SelectiveAccumulator
from src.models.modules.diffusion_teacher import DiffusionScanpathTeacher
from src.models.modules.dual_stream_encoder import DualStreamEncoder
from src.models.modules.foveated_reader import FoveatedEvidenceReader
from src.models.modules.gaze_student import GazeStudent
from src.models.modules.heads import BinaryClassificationHead
from src.models.modules.stop_head import StopHead


@dataclass
class DiffScanAuthConfig:
    """Typed subset of DiffScanAuth configuration."""

    global_stream_name: str = "siglip2-so400m-patch16-224"
    local_stream_name: str = "dinov2_vits14"
    pretrained: bool = True
    use_local_stream: bool = True
    use_teacher: bool = True
    policy_type: str = "transformer"
    teacher_hidden_dim: int = 256
    policy_hidden_dim: int = 256
    glimpse_dim: int = 256
    accumulator_hidden_dim: int = 256
    accumulator_backend: str = "gru"
    patch_grid_size: int = 24
    max_steps: int = 12
    stop_mode: str = "learned_stop"
    fixed_steps: int = 12
    confidence_threshold: float = 0.6
    stop_threshold: float = 0.5
    scheduled_sampling_ratio: float = 0.0
    dropout: float = 0.1
    policy_layers: int = 2
    policy_heads: int = 4


class DiffScanAuth(nn.Module):
    """Human-gaze-supervised sequential evidence accumulation model."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        cfg = DiffScanAuthConfig(**kwargs)
        self.cfg = cfg
        self.max_steps = cfg.max_steps
        self.stop_mode = cfg.stop_mode
        self.fixed_steps = cfg.fixed_steps
        self.confidence_threshold = cfg.confidence_threshold
        self.stop_threshold = cfg.stop_threshold
        self.patch_grid_size = cfg.patch_grid_size
        self.num_patches = cfg.patch_grid_size * cfg.patch_grid_size

        self.encoder = DualStreamEncoder(
            global_stream_name=cfg.global_stream_name,
            local_stream_name=cfg.local_stream_name,
            pretrained=cfg.pretrained,
            use_local_stream=cfg.use_local_stream,
        )
        self.student = GazeStudent(
            global_dim=self.encoder.global_dim,
            hidden_dim=cfg.policy_hidden_dim,
            num_patches=self.num_patches,
            policy_type=cfg.policy_type,
            num_layers=cfg.policy_layers,
            num_heads=cfg.policy_heads,
            dropout=cfg.dropout,
        )
        self.reader = FoveatedEvidenceReader(
            global_dim=self.encoder.global_dim,
            local_dim=self.encoder.local_dim,
            out_dim=cfg.glimpse_dim,
        )
        self.accumulator = SelectiveAccumulator(
            input_dim=cfg.glimpse_dim,
            hidden_dim=cfg.accumulator_hidden_dim,
            backend=cfg.accumulator_backend,
            num_layers=1,
            dropout=cfg.dropout,
        )
        self.intermediate_head = nn.Linear(cfg.accumulator_hidden_dim, 1)
        self.final_head = BinaryClassificationHead(
            in_dim=cfg.accumulator_hidden_dim,
            hidden_dim=cfg.accumulator_hidden_dim,
            dropout=cfg.dropout,
        )
        self.stop_head = StopHead(cfg.accumulator_hidden_dim)
        self.teacher = (
            DiffusionScanpathTeacher(
                global_dim=self.encoder.global_dim,
                num_patches=self.num_patches,
                hidden_dim=cfg.teacher_hidden_dim,
            )
            if cfg.use_teacher
            else None
        )

        patch_centers = []
        for idx in range(self.num_patches):
            y = idx // self.patch_grid_size
            x = idx % self.patch_grid_size
            step = 1.0 / self.patch_grid_size
            patch_centers.append([(x + 0.5) * step, (y + 0.5) * step])
        self.register_buffer("patch_centers", torch.tensor(patch_centers, dtype=torch.float32))

    def encode(self, image: torch.Tensor) -> dict[str, torch.Tensor | str]:
        return self.encoder(image)

    def _teacher_context_manager(self):
        if self.teacher is None:
            return nullcontext()
        teacher_params = list(self.teacher.parameters())
        if teacher_params and all(not param.requires_grad for param in teacher_params):
            return torch.no_grad()
        return nullcontext()

    def _run_teacher(
        self,
        global_context: torch.Tensor,
        patch_idx: torch.Tensor,
        coords: torch.Tensor,
        delta: torch.Tensor,
        durations: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.teacher is None:
            raise RuntimeError("Teacher module is disabled for this model configuration")
        with self._teacher_context_manager():
            return self.teacher(
                global_context=global_context,
                patch_idx=patch_idx,
                coords=coords,
                delta=delta,
                durations=durations,
                mask=mask,
            )

    def forward_teacher(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded = self.encode(batch["image"])
        return self._run_teacher(
            global_context=encoded["global_pooled"],
            patch_idx=batch["patch_idx"].long(),
            coords=batch["fix_xy"].float(),
            delta=batch["fix_delta"].float(),
            durations=batch["fix_dur"].float(),
            mask=batch["mask"].float(),
        )

    def _select_prev_token(
        self,
        teacher_token: torch.Tensor,
        predicted_token: torch.Tensor,
        use_teacher: bool,
    ) -> torch.Tensor:
        if use_teacher:
            return teacher_token
        return predicted_token

    def _compute_coords(self, patch_logits: torch.Tensor, coord_residual: torch.Tensor) -> torch.Tensor:
        patch_prob = torch.softmax(patch_logits, dim=-1)
        centers = patch_prob @ self.patch_centers.to(patch_prob.device)
        residual_scale = 0.5 / self.patch_grid_size
        coords = (centers + residual_scale * coord_residual).clamp(0.0, 1.0)
        return coords

    def _decision_steps(
        self,
        step_logits: torch.Tensor,
        stop_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        b, t = step_logits.shape
        valid_lengths = mask.sum(dim=1).long().clamp(min=1)
        if self.stop_mode == "fixed_k":
            return torch.clamp(torch.full((b,), self.fixed_steps - 1, device=mask.device), max=valid_lengths - 1)

        confidence = torch.sigmoid(step_logits)
        stop_prob = torch.sigmoid(stop_logits)
        decision_steps = valid_lengths - 1
        for i in range(b):
            valid_t = valid_lengths[i].item()
            for step in range(valid_t):
                if confidence[i, step] >= self.confidence_threshold and stop_prob[i, step] >= self.stop_threshold:
                    decision_steps[i] = step
                    break
        return decision_steps

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        teacher_forcing_ratio: float = 1.0,
        scheduled_sampling_ratio: float | None = None,
        use_gaze_inputs: bool = True,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode(batch["image"])
        global_context = encoded["global_pooled"]
        global_map = encoded["global_map"]
        local_map = encoded["local_map"]

        teacher_patch = batch["patch_idx"].long()
        teacher_xy = batch["fix_xy"].float()
        teacher_delta = batch["fix_delta"].float()
        teacher_dur = batch["fix_dur"].float()
        mask = batch["mask"].float()

        batch_size, seq_len = teacher_patch.shape
        max_steps = min(seq_len, self.max_steps)
        if scheduled_sampling_ratio is None:
            scheduled_sampling_ratio = self.cfg.scheduled_sampling_ratio

        start_patch = torch.full((batch_size,), self.num_patches, device=teacher_patch.device, dtype=torch.long)
        start_xy = torch.zeros((batch_size, 2), device=teacher_xy.device, dtype=teacher_xy.dtype)
        start_delta = torch.zeros((batch_size, 2), device=teacher_xy.device, dtype=teacher_xy.dtype)
        start_dur = torch.zeros((batch_size,), device=teacher_dur.device, dtype=teacher_dur.dtype)

        history_patch = [start_patch]
        history_xy = [start_xy]
        history_delta = [start_delta]
        history_dur = [start_dur]

        patch_logits_seq = []
        coord_seq = []
        delta_seq = []
        dur_seq = []
        student_stop_seq = []
        evidence_seq = []

        for step in range(max_steps):
            prev_patch = torch.stack(history_patch, dim=1)
            prev_xy = torch.stack(history_xy, dim=1)
            prev_delta = torch.stack(history_delta, dim=1)
            prev_dur = torch.stack(history_dur, dim=1)

            student_out = self.student(
                global_context=global_context,
                prev_patch_idx=prev_patch,
                prev_coords=prev_xy,
                prev_delta=prev_delta,
                prev_durations=prev_dur,
            )
            cur_patch_logits = student_out["patch_logits"][:, -1, :]
            cur_delta = student_out["delta_pred"][:, -1, :]
            cur_coord = self._compute_coords(cur_patch_logits, student_out["coord_residual"][:, -1, :])
            cur_dur = student_out["dur_pred"][:, -1]
            cur_stop = student_out["stop_logits"][:, -1]

            patch_logits_seq.append(cur_patch_logits)
            coord_seq.append(cur_coord)
            delta_seq.append(cur_delta)
            dur_seq.append(cur_dur)
            student_stop_seq.append(cur_stop)

            predicted_patch = cur_patch_logits.argmax(dim=-1)
            if self.training and use_gaze_inputs:
                teacher_for_history = bool(torch.rand(1, device=teacher_patch.device) >= scheduled_sampling_ratio)
                teacher_for_reader = bool(torch.rand(1, device=teacher_patch.device) < teacher_forcing_ratio)
            else:
                teacher_for_history = False
                teacher_for_reader = False

            next_patch = self._select_prev_token(teacher_patch[:, step], predicted_patch, teacher_for_history and step < teacher_patch.size(1))
            next_xy = self._select_prev_token(teacher_xy[:, step, :], cur_coord, teacher_for_history and step < teacher_xy.size(1))
            next_delta = self._select_prev_token(teacher_delta[:, step, :], cur_delta, teacher_for_history and step < teacher_delta.size(1))
            next_dur = self._select_prev_token(teacher_dur[:, step], cur_dur, teacher_for_history and step < teacher_dur.size(1))

            read_xy = self._select_prev_token(teacher_xy[:, step, :], cur_coord, teacher_for_reader and step < teacher_xy.size(1))
            read_delta = self._select_prev_token(teacher_delta[:, step, :], cur_delta, teacher_for_reader and step < teacher_delta.size(1))
            read_dur = self._select_prev_token(teacher_dur[:, step], cur_dur, teacher_for_reader and step < teacher_dur.size(1))

            evidence = self.reader(
                global_map=global_map,
                local_map=local_map,
                coords=read_xy.unsqueeze(1),
                delta=read_delta.unsqueeze(1),
                durations=read_dur.unsqueeze(1),
                global_context=global_context,
            ).squeeze(1)
            evidence_seq.append(evidence)

            history_patch.append(next_patch)
            history_xy.append(next_xy)
            history_delta.append(next_delta)
            history_dur.append(next_dur)

        evidence_tensor = torch.stack(evidence_seq, dim=1)
        patch_logits_tensor = torch.stack(patch_logits_seq, dim=1)
        coord_tensor = torch.stack(coord_seq, dim=1)
        delta_tensor = torch.stack(delta_seq, dim=1)
        dur_tensor = torch.stack(dur_seq, dim=1)
        student_stop_tensor = torch.stack(student_stop_seq, dim=1)

        states, _ = self.accumulator(evidence_tensor, mask=mask[:, :max_steps])
        step_logits = self.intermediate_head(states).squeeze(-1)
        state_stop_logits = self.stop_head(states)
        stop_logits = 0.5 * (student_stop_tensor + state_stop_logits)
        decision_steps = self._decision_steps(step_logits=step_logits, stop_logits=stop_logits, mask=mask[:, :max_steps])
        final_hidden = states[torch.arange(batch_size, device=states.device), decision_steps]
        cls_logits = self.final_head(final_hidden)

        teacher_outputs = None
        if self.teacher is not None:
            teacher_outputs = self._run_teacher(
                global_context=global_context,
                patch_idx=teacher_patch[:, :max_steps],
                coords=teacher_xy[:, :max_steps, :],
                delta=teacher_delta[:, :max_steps, :],
                durations=teacher_dur[:, :max_steps],
                mask=mask[:, :max_steps],
            )

        outputs = {
            "cls_logits": cls_logits,
            "step_logits": step_logits,
            "patch_logits": patch_logits_tensor,
            "coord_pred": coord_tensor,
            "delta_pred": delta_tensor,
            "dur_pred": dur_tensor,
            "stop_logits": stop_logits,
            "decision_steps": decision_steps,
            "step_probs": torch.sigmoid(step_logits),
            "teacher_outputs": teacher_outputs,
        }
        if teacher_outputs is not None:
            outputs["teacher_patch_logits"] = teacher_outputs["patch_logits"]
            outputs["teacher_coord_pred"] = teacher_outputs["coord_pred"]
            outputs["teacher_delta_pred"] = teacher_outputs["delta_pred"]
            outputs["teacher_dur_pred"] = teacher_outputs["dur_pred"]
            outputs["teacher_loss"] = teacher_outputs["teacher_loss"]
        return outputs
