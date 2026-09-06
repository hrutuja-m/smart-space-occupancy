"""Stage 2: ``data/raw`` -> stratified train/val/test under ``data/processed``.

Why stratify: person counts in the export are very skewed (many empty frames, a
long tail of crowded ones). A naive random split can leave the test set with no
"high" density images, making the headline MAE meaningless. We bucket every
image by count and split each bucket independently so all three splits share the
same density distribution.

Also precomputes Gaussian density maps for the CSRNet-lite head so training does
not pay that cost every epoch.
"""

from __future__ import annotations

import random
import shutil
from collections import defaultdict

import numpy as np

from smart_space.config import (
    DENSITY_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    SPLITS,
    bucket_for_count,
    data_config,
)
from smart_space.data.density import count_from_label, density_map, read_yolo_label


def _clean_processed() -> None:
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    for split in SPLITS:
        (PROCESSED_DIR / split / "images").mkdir(parents=True)
        (PROCESSED_DIR / split / "labels").mkdir(parents=True)
        (DENSITY_DIR / split).mkdir(parents=True)


def _assign_splits(stems: list[str], ratios: tuple[float, float, float], seed: int) -> dict:
    by_bucket: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        c = count_from_label(RAW_DIR / "labels" / f"{stem}.txt")
        by_bucket[bucket_for_count(c)].append(stem)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for items in by_bucket.values():
        rng.shuffle(items)
        n = len(items)
        n_tr = int(n * ratios[0])
        n_va = int(n * ratios[1])
        for s in items[:n_tr]:
            assignment[s] = "train"
        for s in items[n_tr : n_tr + n_va]:
            assignment[s] = "val"
        for s in items[n_tr + n_va :]:
            assignment[s] = "test"
    return assignment


def split(seed: int | None = None) -> dict[str, int]:
    cfg = data_config()
    seed = cfg.seed if seed is None else seed
    ratios = (cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)

    images_dir = RAW_DIR / "images"
    if not images_dir.is_dir():
        raise RuntimeError("data/raw missing — run `smartspace data ingest` first")

    stems = sorted(p.stem for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    _clean_processed()
    assignment = _assign_splits(stems, ratios, seed)

    counts = {s: 0 for s in SPLITS}
    map_hw = (cfg.img_size // 8, cfg.img_size // 8)
    for stem in stems:
        split_name = assignment[stem]
        src_img = next(images_dir.glob(f"{stem}.*"))
        dst_img = PROCESSED_DIR / split_name / "images" / src_img.name
        shutil.copy2(src_img, dst_img)
        shutil.copy2(RAW_DIR / "labels" / f"{stem}.txt", PROCESSED_DIR / split_name / "labels" / f"{stem}.txt")

        boxes = read_yolo_label(RAW_DIR / "labels" / f"{stem}.txt")
        dmap = density_map(boxes, map_hw, sigma=cfg.density_sigma / 8)
        np.save(DENSITY_DIR / split_name / f"{stem}.npy", dmap.astype(np.float32))
        counts[split_name] += 1

    print("Split complete:", ", ".join(f"{k}={v}" for k, v in counts.items()))
    return counts


if __name__ == "__main__":
    split()
