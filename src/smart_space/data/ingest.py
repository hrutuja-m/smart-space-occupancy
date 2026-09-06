"""Stage 1 of the data pipeline: Roboflow export -> ``data/raw`` canonical pool.

The Roboflow "People Detection Dataset v16" export may land either as::

    external_datasets/people_detection/{train,valid,test}/{images,labels}

or nested one level deeper inside a ``*_yolov8`` folder. This module resolves
either shape, merges all splits into a single flat pool under ``data/raw`` with
zero-padded names (``000000.jpg`` / ``000000.txt``), and optionally subsamples.

Run:  ``python -m smart_space.data.ingest``  or  ``smartspace data ingest``.
"""

from __future__ import annotations

import random
import shutil
from glob import glob
from pathlib import Path

from smart_space.config import RAW_DIR, data_config

EXPORT_ROOT = Path("external_datasets/people_detection")
SOURCE_SPLITS = ("train", "valid", "val", "test")


def resolve_export_root(base: Path = EXPORT_ROOT) -> Path:
    if not base.exists():
        raise FileNotFoundError(
            f"{base} not found. Download the dataset first:  bash scripts/download_data.sh"
        )
    if (base / "train" / "images").is_dir():
        return base
    for child in sorted(p for p in base.iterdir() if p.is_dir()):
        if (child / "train" / "images").is_dir():
            return child
    raise RuntimeError(f"Could not locate <root>/train/images under {base}")


def collect_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for split in SOURCE_SPLITS:
        images_dir, labels_dir = root / split / "images", root / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue
        for ext in ("jpg", "jpeg", "png"):
            for img in glob(str(images_dir / f"*.{ext}")):
                img_p = Path(img)
                lbl_p = labels_dir / f"{img_p.stem}.txt"
                if lbl_p.exists():
                    pairs.append((img_p, lbl_p))
    if not pairs:
        raise RuntimeError(f"No (image, label) pairs found under {root}")
    return pairs


def ingest(max_samples: int | None = None, seed: int | None = None) -> int:
    cfg = data_config()
    max_samples = cfg.max_samples if max_samples is None else max_samples
    seed = cfg.seed if seed is None else seed

    root = resolve_export_root()
    pairs = collect_pairs(root)
    random.Random(seed).shuffle(pairs)
    if max_samples and max_samples > 0:
        pairs = pairs[:max_samples]

    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    (RAW_DIR / "images").mkdir(parents=True)
    (RAW_DIR / "labels").mkdir(parents=True)

    for i, (img_src, lbl_src) in enumerate(pairs):
        stem = f"{i:06d}"
        ext = img_src.suffix.lower() if img_src.suffix.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
        shutil.copy2(img_src, RAW_DIR / "images" / f"{stem}{ext}")
        shutil.copy2(lbl_src, RAW_DIR / "labels" / f"{stem}.txt")

    print(f"Ingested {len(pairs)} image/label pairs -> {RAW_DIR} (source: {root})")
    return len(pairs)


if __name__ == "__main__":
    ingest()
