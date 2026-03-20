"""Lightning module for the SeqDet without human gaze baseline."""

from __future__ import annotations

from typing import Any

from src.lightning.lit_seq_base import LitSequentialBase
from src.models.seqdet_no_gaze import SeqDetNoGaze


class LitSeqDet(LitSequentialBase):
    """Sequential detector without human-gaze supervision."""

    def __init__(self, model_cfg: dict[str, Any], optim_cfg: dict[str, Any]) -> None:
        model = SeqDetNoGaze(
            global_stream_name=str(model_cfg.get("global_stream_name", "siglip2-so400m-patch16-224")),
            local_stream_name=str(model_cfg.get("local_stream_name", "dinov2_vits14")),
            pretrained=bool(model_cfg.get("pretrained", True)),
            use_local_stream=bool(model_cfg.get("use_local_stream", True)),
            use_teacher=False,
            policy_type=str(model_cfg.get("policy_type", "transformer")),
            teacher_hidden_dim=int(model_cfg.get("teacher_hidden_dim", 256)),
            policy_hidden_dim=int(model_cfg.get("policy_hidden_dim", 256)),
            glimpse_dim=int(model_cfg.get("glimpse_dim", 256)),
            accumulator_hidden_dim=int(model_cfg.get("accumulator_hidden_dim", 256)),
            accumulator_backend=str(model_cfg.get("accumulator_backend", "gru")),
            patch_grid_size=int(model_cfg.get("patch_grid_size", 24)),
            max_steps=int(model_cfg.get("max_steps", 12)),
            stop_mode=str(model_cfg.get("stop_mode", "learned_stop")),
            fixed_steps=int(model_cfg.get("fixed_steps", model_cfg.get("max_steps", 12))),
            confidence_threshold=float(model_cfg.get("confidence_threshold", 0.6)),
            stop_threshold=float(model_cfg.get("stop_threshold", 0.5)),
            scheduled_sampling_ratio=float(model_cfg.get("scheduled_sampling_ratio", 1.0)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            policy_layers=int(model_cfg.get("policy_layers", 2)),
            policy_heads=int(model_cfg.get("policy_heads", 4)),
        )
        super().__init__(
            model=model,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            use_gaze_supervision=False,
            use_teacher_distill=False,
        )
