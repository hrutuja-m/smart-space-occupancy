"""Unified single-image occupancy predictor.

    predictor = OccupancyPredictor.load("xgboost")
    predictor.predict("frame.jpg")   -> PredictionResult(count=7, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from smart_space.config import (
    DENSITY_RUN_DIR,
    RESNET_RUN_DIR,
    XGB_RUN_DIR,
    resolve_device,
)
from smart_space.data.dataset import IMAGENET_MEAN, IMAGENET_STD

MODELS = ("yolo", "resnet", "xgboost", "density")


@dataclass
class PredictionResult:
    model: str
    count: float
    boxes: list[list[float]] | None = None       # yolo only
    density_map: np.ndarray | None = None        # density only

    def rounded(self) -> int:
        return int(round(self.count))


def _tensor(image, img_size: int) -> torch.Tensor:
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image[..., ::-1] if image.shape[-1] == 3 else image).convert("RGB")
    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return tfm(image).unsqueeze(0)


class OccupancyPredictor:
    def __init__(self, model_name: str, obj, device: str, img_size: int = 224):
        self.model_name = model_name
        self._obj = obj
        self.device = device
        self.img_size = img_size

    # -- constructors ---------------------------------------------------
    @classmethod
    def load(cls, model_name: str, weights: str | Path | None = None, img_size: int = 224):
        model_name = model_name.lower()
        device = resolve_device()
        if model_name == "yolo":
            from smart_space.models.yolo_counter import YoloCounter

            return cls("yolo", YoloCounter(weights), device, img_size)
        if model_name == "resnet":
            from smart_space.models.resnet_regressor import build_regressor

            net = build_regressor(pretrained=False)
            net.load_state_dict(torch.load(weights or RESNET_RUN_DIR / "resnet18_regressor.pt", map_location=device))
            return cls("resnet", net.to(device).eval(), device, img_size)
        if model_name == "density":
            from smart_space.models.density_net import CSRNetLite

            net = CSRNetLite(pretrained=False)
            net.load_state_dict(torch.load(weights or DENSITY_RUN_DIR / "csrnet_lite.pt", map_location=device))
            return cls("density", net.to(device).eval(), device, img_size)
        if model_name == "xgboost":
            from smart_space.models.resnet_regressor import build_feature_extractor
            from smart_space.models.xgb_regressor import load_xgb

            bundle = {
                "backbone": build_feature_extractor(pretrained=True).to(device).eval(),
                "xgb": load_xgb(weights or XGB_RUN_DIR / "xgb_occupancy.json"),
            }
            return cls("xgboost", bundle, device, img_size)
        raise ValueError(f"unknown model '{model_name}', choose from {MODELS}")

    # -- inference ----------------------------------------------------
    @torch.no_grad()
    def predict(self, image) -> PredictionResult:
        if self.model_name == "yolo":
            boxes = self._obj.detect(image)
            return PredictionResult("yolo", float(len(boxes)), boxes=boxes.tolist())

        x = _tensor(image, self.img_size).to(self.device)
        if self.model_name == "resnet":
            count = float(self._obj(x).squeeze().item())
            return PredictionResult("resnet", max(0.0, count))
        if self.model_name == "density":
            dmap = self._obj(x).squeeze().cpu().numpy()
            return PredictionResult("density", float(max(0.0, dmap.sum())), density_map=dmap)
        if self.model_name == "xgboost":
            feat = self._obj["backbone"](x).flatten(1).cpu().numpy()
            count = float(self._obj["xgb"].predict(feat)[0])
            return PredictionResult("xgboost", max(0.0, count))
        raise RuntimeError(self.model_name)
