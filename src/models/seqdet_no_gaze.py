"""Sequential detector baseline without human gaze supervision."""

from __future__ import annotations

from src.models.diffscanauth import DiffScanAuth


class SeqDetNoGaze(DiffScanAuth):
    """Sequential detector that keeps the architecture but removes human-gaze training."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("use_teacher", False)
        kwargs.setdefault("stop_mode", "learned_stop")
        super().__init__(**kwargs)
