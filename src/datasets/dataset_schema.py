"""Dataset schema dataclasses for metadata and eye tracking rows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MetadataRecord:
    """Single image-level metadata record."""

    image_id: str
    image_path: str
    label: int
    scene: str
    source_type: str
    generator: str
    width: int
    height: int
    split: str = ""


@dataclass(slots=True)
class EyeTrackingRecord:
    """Single eye-tracking row (point or fixation-like event)."""

    subject_id: str
    image_id: str
    t: float
    x: float
    y: float
    duration: float
    event_type: str
    validity: float | None = None
    pupil: float | None = None


@dataclass(slots=True)
class ProcessedFixationRecord:
    """Processed fixation-level record used in training."""

    subject_id: str
    image_id: str
    fixation_idx: int
    x_norm: float
    y_norm: float
    duration_norm: float
    duration_ms: float
    patch_index: int
    split: str = ""
