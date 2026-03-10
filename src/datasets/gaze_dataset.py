"""Subject-image gaze sequence dataset for sequential detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.transforms import build_image_transform
from src.features.fixation_tokenizer import patch_histogram
from src.features.heatmap import gaussian_heatmap
from src.utils.io import project_root


class GazeSequenceDataset(Dataset):
    """Build fixed-length fixation sequences grouped by (subject_id, image_id)."""

    def __init__(
        self,
        metadata_csv: str | Path,
        fixations_csv: str | Path,
        split: str,
        image_size: int = 384,
        max_fixations: int = 12,
        patch_grid_size: int = 24,
        duration_norm_mode: str = "log_zscore",
        train: bool = False,
        use_aug: bool = False,
        heatmap_size: int = 96,
    ) -> None:
        del duration_norm_mode  # already normalized during preprocessing
        self.root = project_root()
        self.meta = pd.read_csv(metadata_csv)
        self.meta = self.meta[self.meta["split"] == split].copy()
        self.fix = pd.read_csv(fixations_csv)
        if "split" in self.fix.columns:
            self.fix = self.fix[self.fix["split"] == split].copy()
        else:
            valid_ids = set(self.meta["image_id"].tolist())
            self.fix = self.fix[self.fix["image_id"].isin(valid_ids)].copy()

        self.meta_by_id = self.meta.set_index("image_id")
        self.transform = build_image_transform(image_size=image_size, train=train, use_aug=use_aug)

        self.max_fixations = max_fixations
        self.patch_grid_size = patch_grid_size
        self.num_patches = patch_grid_size * patch_grid_size
        self.heatmap_size = heatmap_size

        grouped = self.fix.groupby(["subject_id", "image_id"], sort=False)
        self.keys: list[tuple[str, str]] = []
        for (sid, iid), g in grouped:
            if iid not in self.meta_by_id.index:
                continue
            if len(g) == 0:
                continue
            self.keys.append((str(sid), str(iid)))

        if len(self.keys) == 0:
            raise ValueError(f"No gaze sequence samples for split='{split}'")

    def __len__(self) -> int:
        return len(self.keys)

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        return self.root / path

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sid, iid = self.keys[index]
        m = self.meta_by_id.loc[iid]
        g = self.fix[(self.fix["subject_id"] == sid) & (self.fix["image_id"] == iid)].sort_values("fixation_idx")

        image_path = self._resolve_path(str(m["image_path"]))
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        image = self.transform(image)

        n = min(len(g), self.max_fixations)
        mask = np.zeros(self.max_fixations, dtype=np.float32)
        patch_idx = np.zeros(self.max_fixations, dtype=np.int64)
        fix_xy = np.zeros((self.max_fixations, 2), dtype=np.float32)
        fix_dur = np.zeros(self.max_fixations, dtype=np.float32)

        gx = g["x_norm"].to_numpy(dtype=np.float32)[:n]
        gy = g["y_norm"].to_numpy(dtype=np.float32)[:n]
        gp = g["patch_index"].to_numpy(dtype=np.int64)[:n]
        gd = g["duration_norm"].to_numpy(dtype=np.float32)[:n]

        if n > 0:
            mask[:n] = 1.0
            patch_idx[:n] = gp
            fix_xy[:n, 0] = gx
            fix_xy[:n, 1] = gy
            fix_dur[:n] = gd

        patch_dist = patch_histogram(patch_idx[:n], self.num_patches)
        heatmap = gaussian_heatmap(np.stack([gx, gy], axis=1) if n > 0 else np.empty((0, 2)), size=self.heatmap_size)

        return {
            "image": image,
            "label": torch.tensor(float(m["label"]), dtype=torch.float32),
            "image_id": iid,
            "subject_id": sid,
            "patch_idx": torch.from_numpy(patch_idx),
            "fix_xy": torch.from_numpy(fix_xy),
            "fix_dur": torch.from_numpy(fix_dur),
            "mask": torch.from_numpy(mask),
            "patch_dist": torch.from_numpy(patch_dist.astype(np.float32)),
            "heatmap": torch.from_numpy(heatmap).unsqueeze(0),
        }
