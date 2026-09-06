"""Thin wrapper turning a YOLOv8 detector into an occupancy counter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from smart_space.config import YOLO_RUN_DIR


def find_weights(explicit: str | Path | None = None) -> Path:
    """Locate trained YOLO weights: explicit path -> best.pt -> newest *.pt."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    best = YOLO_RUN_DIR / "occupancy" / "weights" / "best.pt"
    if best.exists():
        return best
    candidates = sorted(YOLO_RUN_DIR.glob("**/weights/*.pt"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"No YOLO weights under {YOLO_RUN_DIR}. Train first: `smartspace train yolo`."
    )


class YoloCounter:
    def __init__(self, weights: str | Path | None = None, conf: float = 0.25):
        from ultralytics import YOLO

        self.weights_path = find_weights(weights)
        self.model = YOLO(str(self.weights_path))
        self.conf = conf

    def count(self, image) -> int:
        """image: path / np.ndarray / PIL — return number of person detections."""
        res = self.model(image, conf=self.conf, verbose=False)[0]
        return 0 if res.boxes is None else int(len(res.boxes))

    def detect(self, image) -> np.ndarray:
        """Return an (N, 4) array of xyxy person boxes (for tracking)."""
        res = self.model(image, conf=self.conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return res.boxes.xyxy.cpu().numpy()
