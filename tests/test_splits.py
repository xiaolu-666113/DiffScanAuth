from __future__ import annotations

import pandas as pd

from src.datasets.split_utils import make_splits


def test_make_splits_no_image_leakage() -> None:
    rows = []
    for i in range(60):
        rows.append(
            {
                "image_id": f"img_{i:03d}",
                "image_path": f"data/raw/img_{i:03d}.png",
                "label": i % 2,
                "scene": "cat" if i % 3 == 0 else "dog",
                "generator": "sdxl" if i % 4 == 0 else "",
                "width": 384,
                "height": 384,
                "split": "",
            }
        )
    df = pd.DataFrame(rows)

    res = make_splits(df, seed=123)
    train_ids = set(res.train["image_id"].tolist())
    val_ids = set(res.val["image_id"].tolist())
    test_ids = set(res.test["image_id"].tolist())

    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0
