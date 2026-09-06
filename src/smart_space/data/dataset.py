"""PyTorch datasets for the counting and density-estimation heads.

``OccupancyCountDataset``  -> (image_tensor, count)          for ResNet / XGBoost
``OccupancyDensityDataset`` -> (image_tensor, density_map)   for CSRNet-lite
"""

from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from smart_space.data.density import count_from_label, density_map, read_yolo_label

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _list_images(images_dir: Path) -> list[Path]:
    paths: list[str] = []
    for ext in ("jpg", "jpeg", "png"):
        paths += glob(str(images_dir / f"*.{ext}"))
    return sorted(Path(p) for p in paths)


def _base_transform(img_size: int, train: bool) -> transforms.Compose:
    aug: list = []
    if train:
        aug = [
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            *aug,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class OccupancyCountDataset(Dataset):
    """Image -> integer person count (as a float32 regression target)."""

    def __init__(self, split_dir: str | Path, img_size: int = 224, train: bool = False):
        split_dir = Path(split_dir)
        self.images_dir = split_dir / "images"
        self.labels_dir = split_dir / "labels"
        self.img_paths = _list_images(self.images_dir)
        if not self.img_paths:
            raise RuntimeError(f"No images under {self.images_dir}")
        self.transform = _base_transform(img_size, train)

    def __len__(self) -> int:
        return len(self.img_paths)

    def _label_path(self, img_path: Path) -> Path:
        return self.labels_dir / f"{img_path.stem}.txt"

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(float(count_from_label(self._label_path(img_path))), dtype=torch.float32)
        return x, y


class OccupancyDensityDataset(Dataset):
    """Image -> (C=1) density map. Map is downsampled by ``downsample`` (CSRNet uses 8)."""

    def __init__(
        self,
        split_dir: str | Path,
        img_size: int = 224,
        sigma: float = 4.0,
        downsample: int = 8,
        train: bool = False,
    ):
        split_dir = Path(split_dir)
        self.images_dir = split_dir / "images"
        self.labels_dir = split_dir / "labels"
        self.img_paths = _list_images(self.images_dir)
        if not self.img_paths:
            raise RuntimeError(f"No images under {self.images_dir}")
        self.img_size = img_size
        self.sigma = sigma
        self.downsample = downsample
        self.train = train
        self.img_transform = _base_transform(img_size, train=False)  # geometry must match map

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        boxes = read_yolo_label(self.labels_dir / f"{img_path.stem}.txt")

        do_flip = self.train and np.random.rand() < 0.5
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if boxes.size:
                boxes = boxes.copy()
                boxes[:, 0] = 1.0 - boxes[:, 0]

        x = self.img_transform(img)
        map_hw = (self.img_size // self.downsample, self.img_size // self.downsample)
        dmap = density_map(boxes, map_hw, sigma=self.sigma / self.downsample)
        return x, torch.from_numpy(dmap).unsqueeze(0)  # (1, H/8, W/8)
