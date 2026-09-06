"""Central configuration + path resolution for the Smart Space Occupancy project.

Everything downstream imports paths and the compute device from here so that the
data layout is defined in exactly one place (the old codebase had three
disagreeing layouts).

Canonical on-disk layout produced by ``smart_space.data.ingest`` /
``smart_space.data.split``::

    data/
      raw/                     # unmodified Roboflow export lives here
      processed/
        train/{images,labels}/
        val/{images,labels}/
        test/{images,labels}/
        density/{train,val,test}/*.npy   # precomputed Gaussian density maps

YOLO consumes ``configs/yolo_data.yaml`` which is generated from this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DENSITY_DIR = PROCESSED_DIR / "density"

RUNS_DIR = PROJECT_ROOT / "runs"
YOLO_RUN_DIR = RUNS_DIR / "yolo"
RESNET_RUN_DIR = RUNS_DIR / "resnet"
XGB_RUN_DIR = RUNS_DIR / "xgboost"
DENSITY_RUN_DIR = RUNS_DIR / "density"
EVAL_DIR = RUNS_DIR / "eval"
EXPORT_DIR = RUNS_DIR / "export"

CONFIG_DIR = PROJECT_ROOT / "configs"
DOCS_DIR = PROJECT_ROOT / "docs"

SPLITS = ("train", "val", "test")

# Density buckets used for stratified splitting *and* for the per-bucket error
# breakdown in evaluation. Chosen to reflect operational regimes of a room:
#   empty | a few people | a working group | a crowded / over-capacity room
DENSITY_BUCKETS = {
    "empty": (0, 0),
    "low": (1, 4),
    "medium": (5, 12),
    "high": (13, 10_000),
}


def bucket_for_count(count: int) -> str:
    for name, (lo, hi) in DENSITY_BUCKETS.items():
        if lo <= count <= hi:
            return name
    return "high"


# --------------------------------------------------------------------------- #
# Compute device
# --------------------------------------------------------------------------- #
def resolve_device(prefer: str | None = None) -> str:
    """Return the best available torch device string.

    Order: explicit ``prefer`` -> env ``SMARTSPACE_DEVICE`` -> cuda -> mps -> cpu.
    """
    import torch

    choice = prefer or os.getenv("SMARTSPACE_DEVICE")
    if choice:
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- #
# Experiment config (loaded from configs/*.yaml, overridable from CLI)
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    max_samples: int = 800          # cap for local (MPS/CPU) runs; set 0 for "all"
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    img_size: int = 224             # for regression / density models
    density_sigma: float = 4.0      # Gaussian kernel std (px) for density maps


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    device: str = field(default_factory=resolve_device)
    mlflow_experiment: str = "smart-space-occupancy"


@lru_cache
def load_yaml(name: str) -> dict:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def data_config(**overrides) -> DataConfig:
    cfg = {**load_yaml("data.yaml"), **overrides}
    return DataConfig(**{k: v for k, v in cfg.items() if k in DataConfig.__annotations__})


def train_config(name: str = "resnet.yaml", **overrides) -> TrainConfig:
    cfg = {**load_yaml(name), **overrides}
    return TrainConfig(**{k: v for k, v in cfg.items() if k in TrainConfig.__annotations__})


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, DENSITY_DIR, RUNS_DIR, EVAL_DIR, EXPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def write_yolo_data_yaml() -> Path:
    """Generate the Ultralytics dataset yaml from the canonical layout."""
    CONFIG_DIR.mkdir(exist_ok=True)
    out = CONFIG_DIR / "yolo_data.yaml"
    payload = {
        "path": str(PROCESSED_DIR),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": ["person"],
    }
    out.write_text(yaml.safe_dump(payload, sort_keys=False))
    return out
