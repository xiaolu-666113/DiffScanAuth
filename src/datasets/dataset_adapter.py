"""Dataset adapter for automatic scanning and schema normalization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.features.gaze_processing import normalize_columns
from src.utils.io import ensure_dir, project_root, save_csv, save_json


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TAB_EXTS = {".csv", ".tsv", ".txt"}

GENERATOR_KEYWORDS = [
    "sdxl",
    "midjourney",
    "flux",
    "dall-e",
    "dalle",
    "stable-diffusion",
    "imagen",
    "firefly",
    "aigc",
    "fake",
]
SOURCE_REAL_KEYWORDS = ["real", "authentic", "natural"]
SOURCE_FAKE_KEYWORDS = ["fake", "aigc", "synthetic", "generated", "ai"]


@dataclass
class DatasetScanReport:
    """Summary report after scanning raw data directory."""

    raw_dir: str
    n_images: int
    n_tables: int
    image_example_paths: list[str]
    table_example_paths: list[str]



def _norm_path_str(path: Path) -> str:
    return path.as_posix()


def _contains_any(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


def _infer_source_type(path: Path) -> str:
    s = path.as_posix().lower()
    if _contains_any(s, SOURCE_FAKE_KEYWORDS):
        return "aigc"
    if _contains_any(s, SOURCE_REAL_KEYWORDS):
        return "real"
    return "unknown"


def _infer_generator(path: Path) -> str:
    s = path.as_posix().lower()
    for k in GENERATOR_KEYWORDS:
        if k in s:
            if k == "dalle":
                return "dall-e"
            return k
    return ""


def _infer_scene(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    deny = {
        "data",
        "raw",
        "processed",
        "splits",
        "synthetic",
        "real",
        "fake",
        "aigc",
        "generated",
        "images",
    }
    deny.update(GENERATOR_KEYWORDS)
    for token in reversed(parts[:-1]):
        if token not in deny and len(token) > 1:
            return token
    return "unknown"


def _infer_label(path: Path, source_type: str) -> int:
    s = path.as_posix().lower()
    if _contains_any(s, SOURCE_FAKE_KEYWORDS):
        return 1
    if _contains_any(s, SOURCE_REAL_KEYWORDS):
        return 0
    if source_type == "aigc":
        return 1
    return 0


def _safe_image_id(path: Path, existing: set[str]) -> str:
    base = path.stem
    if base not in existing:
        return base
    h = hashlib.md5(path.as_posix().encode("utf-8")).hexdigest()[:8]
    return f"{base}_{h}"


def scan_raw_dataset(raw_dir: str | Path) -> DatasetScanReport:
    """Scan raw dataset directory and return summary report."""
    root = Path(raw_dir)
    image_files = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    table_files = sorted([p for p in root.rglob("*") if p.suffix.lower() in TAB_EXTS])
    return DatasetScanReport(
        raw_dir=str(root),
        n_images=len(image_files),
        n_tables=len(table_files),
        image_example_paths=[_norm_path_str(p) for p in image_files[:10]],
        table_example_paths=[_norm_path_str(p) for p in table_files[:10]],
    )


def _load_metadata_hints(raw_dir: Path) -> pd.DataFrame:
    """Load optional metadata CSV hints if available."""
    candidates = [
        p
        for p in raw_dir.rglob("*.csv")
        if any(k in p.name.lower() for k in ["meta", "label", "annotation"])
    ]
    if not candidates:
        return pd.DataFrame()

    for c in candidates:
        try:
            df = pd.read_csv(c)
        except Exception:
            continue

        cols = {x.lower().strip(): x for x in df.columns}
        image_col = None
        for k in ["image_path", "image", "filename", "img", "path"]:
            if k in cols:
                image_col = cols[k]
                break
        if image_col is None:
            continue

        out = pd.DataFrame()
        out["_hint_key"] = df[image_col].astype(str).map(lambda x: Path(x).stem)

        label_col = None
        for k in ["label", "is_fake", "target", "source_type"]:
            if k in cols:
                label_col = cols[k]
                break
        if label_col is not None:
            vals = df[label_col].astype(str).str.lower()
            out["label"] = vals.map(
                lambda x: 1 if x in {"1", "fake", "aigc", "synthetic", "generated", "true"} else 0
            )

        scene_col = next((cols[k] for k in ["scene", "category", "class"] if k in cols), None)
        gen_col = next((cols[k] for k in ["generator", "model", "engine"] if k in cols), None)

        if scene_col is not None:
            out["scene"] = df[scene_col].fillna("unknown").astype(str)
        if gen_col is not None:
            out["generator"] = df[gen_col].fillna("").astype(str)

        out = out.dropna(subset=["_hint_key"]).drop_duplicates("_hint_key")
        if len(out) > 0:
            return out

    return pd.DataFrame()


def _create_synthetic_images(
    raw_dir: Path,
    n_images: int = 120,
    image_size: int = 384,
    seed: int = 42,
) -> list[Path]:
    """Create a synthetic dataset when no raw image exists."""
    rng = np.random.default_rng(seed)
    scenes = ["cat", "dog", "landscape", "building"]
    generators = ["sdxl", "midjourney", "flux", "dall-e"]

    created: list[Path] = []
    per_label = n_images // 2

    for i in range(n_images):
        is_fake = i >= per_label
        scene = scenes[i % len(scenes)]
        source = "aigc" if is_fake else "real"
        gen = generators[i % len(generators)] if is_fake else "camera"

        out_dir = raw_dir / "synthetic" / source / scene / gen
        ensure_dir(out_dir)

        arr = rng.normal(128, 45, size=(image_size, image_size, 3)).clip(0, 255).astype(np.uint8)

        if is_fake:
            # Synthetic-like repetitive block artifacts.
            tile = rng.integers(80, 180, size=(16, 16, 3), dtype=np.uint8)
            for y in range(0, image_size, 32):
                for x in range(0, image_size, 32):
                    arr[y : y + 16, x : x + 16] = tile
        else:
            # Real-like smooth luminance gradient.
            yy, xx = np.mgrid[0:image_size, 0:image_size]
            grad = ((xx + yy) / (2 * image_size) * 60).astype(np.uint8)
            arr[:, :, 1] = (arr[:, :, 1] * 0.7 + grad * 0.3).astype(np.uint8)

        out_path = out_dir / f"syn_{i:05d}.png"
        Image.fromarray(arr).save(out_path)
        created.append(out_path)

    return created


def build_metadata(
    raw_dir: str | Path,
    output_csv: str | Path,
    output_report_json: str | Path | None = None,
    allow_synthetic: bool = True,
    synthetic_num_images: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    """Scan raw images and build standardized metadata CSV."""
    root = Path(raw_dir)
    ensure_dir(root)
    scan = scan_raw_dataset(root)

    if scan.n_images == 0 and allow_synthetic:
        _create_synthetic_images(root, n_images=synthetic_num_images, seed=seed)
        scan = scan_raw_dataset(root)

    image_files = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    hints = _load_metadata_hints(root)
    hint_map = hints.set_index("_hint_key").to_dict(orient="index") if not hints.empty else {}

    records = []
    existing_ids: set[str] = set()
    repo_root = project_root()
    for p in image_files:
        try:
            rel = p.resolve().relative_to(repo_root.resolve())
        except ValueError:
            rel = p.resolve()
        source_type = _infer_source_type(p)
        scene = _infer_scene(p)
        generator = _infer_generator(p)
        label = _infer_label(p, source_type)

        hint = hint_map.get(p.stem)
        if hint is not None:
            label = int(hint.get("label", label))
            scene = str(hint.get("scene", scene))
            generator = str(hint.get("generator", generator))

        image_id = _safe_image_id(p, existing_ids)
        existing_ids.add(image_id)

        try:
            with Image.open(p) as im:
                width, height = im.size
        except Exception:
            width, height = 0, 0

        if source_type == "unknown":
            source_type = "aigc" if label == 1 else "real"

        records.append(
            {
                "image_id": image_id,
                "image_path": rel.as_posix(),
                "label": int(label),
                "scene": scene,
                "source_type": source_type,
                "generator": generator if generator != "camera" else "",
                "width": int(width),
                "height": int(height),
                "split": "",
            }
        )

    df = pd.DataFrame(records).sort_values("image_id").reset_index(drop=True)
    save_csv(df, output_csv)

    if output_report_json is not None:
        report = {
            "raw_dir": str(root),
            "n_images": int(len(df)),
            "n_unique_scenes": int(df["scene"].nunique()),
            "n_fake": int(df["label"].sum()),
            "n_real": int((df["label"] == 0).sum()),
            "scan": scan.__dict__,
        }
        save_json(report, output_report_json)

    return df


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path)


def _map_image_ids(df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Map eye-tracking image identifiers to metadata image_id if needed."""
    out = df.copy()
    valid_ids = set(metadata_df["image_id"].astype(str).tolist())
    stem_to_id = {Path(p).stem: iid for iid, p in zip(metadata_df["image_id"], metadata_df["image_path"]) }

    def mapper(v: str) -> str:
        if v in valid_ids:
            return v
        stem = Path(str(v)).stem
        return stem_to_id.get(stem, v)

    out["image_id"] = out["image_id"].astype(str).map(mapper)
    out = out[out["image_id"].isin(valid_ids)].copy()
    return out


def _generate_synthetic_eye_tracking(
    metadata_df: pd.DataFrame,
    n_subjects: int = 10,
    min_fix: int = 6,
    max_fix: int = 16,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic fixation sequences aligned with metadata."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, m in metadata_df.iterrows():
        image_id = str(m["image_id"])
        width = float(m["width"])
        height = float(m["height"])
        label = int(m["label"])

        for s in range(n_subjects):
            sid = f"subj_{s:03d}"
            n_fix = int(rng.integers(min_fix, max_fix + 1))
            t = 0.0
            for _ in range(n_fix):
                if label == 1:
                    # Fake images: more central and slightly longer dwell.
                    x_norm = float(np.clip(rng.normal(0.5, 0.16), 0.0, 1.0))
                    y_norm = float(np.clip(rng.normal(0.5, 0.16), 0.0, 1.0))
                    dur = float(np.clip(rng.normal(210, 55), 60, 600))
                else:
                    x_norm = float(np.clip(rng.normal(0.5, 0.26), 0.0, 1.0))
                    y_norm = float(np.clip(rng.normal(0.5, 0.26), 0.0, 1.0))
                    dur = float(np.clip(rng.normal(150, 45), 50, 500))

                rows.append(
                    {
                        "subject_id": sid,
                        "image_id": image_id,
                        "t": t,
                        "x": x_norm * width,
                        "y": y_norm * height,
                        "duration": dur,
                        "event_type": "fixation",
                        "validity": 1.0,
                        "pupil": float(np.clip(rng.normal(4.0, 0.6), 2.5, 6.5)),
                    }
                )
                t += dur

    return pd.DataFrame(rows)


def build_eye_tracking(
    raw_dir: str | Path,
    metadata_csv: str | Path,
    output_csv: str | Path,
    allow_synthetic: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Build standardized eye_tracking.csv from raw files or synthetic fallback."""
    root = Path(raw_dir)
    metadata_df = pd.read_csv(metadata_csv)

    candidates = [
        p
        for p in root.rglob("*")
        if p.suffix.lower() in TAB_EXTS
        and any(k in p.name.lower() for k in ["eye", "gaze", "fix", "scanpath", "et"]) 
    ]

    frames: list[pd.DataFrame] = []
    for p in candidates:
        try:
            raw = _read_table(p)
            std = normalize_columns(raw)
            if len(std) == 0:
                continue
            frames.append(std)
        except Exception:
            continue

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = _map_image_ids(df, metadata_df)
        df["subject_id"] = df["subject_id"].fillna("unknown_subject").astype(str)

        # Ensure time exists.
        if df["t"].isna().all():
            df = df.sort_values(["subject_id", "image_id"]).copy()
            df["duration"] = df["duration"].fillna(120)
            df["t"] = df.groupby(["subject_id", "image_id"])["duration"].cumsum() - df["duration"]

        df["event_type"] = df["event_type"].fillna("fixation")
    else:
        if not allow_synthetic:
            raise RuntimeError("No eye-tracking files found and synthetic fallback disabled")
        df = _generate_synthetic_eye_tracking(metadata_df, seed=seed)

    out = df[
        [
            "subject_id",
            "image_id",
            "t",
            "x",
            "y",
            "duration",
            "event_type",
            "validity",
            "pupil",
        ]
    ].copy()
    out = out.dropna(subset=["image_id", "x", "y"]).reset_index(drop=True)

    save_csv(out, output_csv)
    return out


def inspect_and_dump(raw_dir: str | Path, output_json: str | Path) -> dict:
    """Inspect raw directory and dump summary report JSON."""
    report = scan_raw_dataset(raw_dir)
    payload = report.__dict__.copy()
    payload["raw_dir_tree_top"] = sorted([p.name for p in Path(raw_dir).iterdir()]) if Path(raw_dir).exists() else []
    save_json(payload, output_json)
    return payload
