"""I/O utilities for project paths and serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def project_root() -> Path:
    """Return repository root based on this file location."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save dict as formatted JSON."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON file to dict."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save dataframe as CSV."""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False)
