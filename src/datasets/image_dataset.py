"""Image-level datasets for static and heatmap baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.transforms import build_image_transform
from src.utils.io import project_root


class StaticImageDataset(Dataset):
    """Image classification dataset with optional heatmap targets."""

    def __init__(
        self,
        metadata_csv: str | Path,
        split: str,
        image_size: int = 384,
        train: bool = False,
        use_aug: bool = False,
        heatmap_dir: str | Path | None = None,
        heatmap_size: int = 96,
    ) -> None:
        self.root = project_root()
        self.df = pd.read_csv(metadata_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.transform = build_image_transform(image_size=image_size, train=train, use_aug=use_aug)
        self.heatmap_dir = Path(heatmap_dir) if heatmap_dir is not None else None
        self.heatmap_size = heatmap_size

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split='{split}' in {metadata_csv}")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        return self.root / path

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[index]
        image_path = self._resolve_path(str(row["image_path"]))
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        image = self.transform(image)

        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        item: dict[str, torch.Tensor | str] = {
            "image": image,
            "label": label,
            "image_id": str(row["image_id"]),
            "image_path": str(image_path),
        }

        if self.heatmap_dir is not None:
            hm_path = self.heatmap_dir / f"{row['image_id']}.npy"
            if hm_path.exists():
                heat = np.load(hm_path).astype(np.float32)
            else:
                heat = np.zeros((self.heatmap_size, self.heatmap_size), dtype=np.float32)
            item["heatmap"] = torch.from_numpy(heat).unsqueeze(0)

        return item
