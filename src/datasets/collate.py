"""Collate functions for static and sequence datasets."""

from __future__ import annotations

from typing import Any

import torch


def collate_static(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate image-level samples."""
    out: dict[str, Any] = {
        "image": torch.stack([x["image"] for x in batch], dim=0),
        "label": torch.stack([x["label"] for x in batch], dim=0),
        "image_id": [x["image_id"] for x in batch],
        "image_path": [x["image_path"] for x in batch],
    }
    if "heatmap" in batch[0]:
        out["heatmap"] = torch.stack([x["heatmap"] for x in batch], dim=0)
    return out


def collate_gaze(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate gaze sequence samples."""
    return {
        "image": torch.stack([x["image"] for x in batch], dim=0),
        "label": torch.stack([x["label"] for x in batch], dim=0),
        "image_id": [x["image_id"] for x in batch],
        "image_path": [x["image_path"] for x in batch],
        "subject_id": [x["subject_id"] for x in batch],
        "patch_idx": torch.stack([x["patch_idx"] for x in batch], dim=0),
        "fix_xy": torch.stack([x["fix_xy"] for x in batch], dim=0),
        "fix_delta": torch.stack([x["fix_delta"] for x in batch], dim=0),
        "fix_dur": torch.stack([x["fix_dur"] for x in batch], dim=0),
        "mask": torch.stack([x["mask"] for x in batch], dim=0),
        "patch_dist": torch.stack([x["patch_dist"] for x in batch], dim=0),
        "heatmap": torch.stack([x["heatmap"] for x in batch], dim=0),
        "source_type": [x["source_type"] for x in batch],
        "scene": [x["scene"] for x in batch],
        "generator": [x["generator"] for x in batch],
        "shuffled_from_image_id": [x["shuffled_from_image_id"] for x in batch],
    }
