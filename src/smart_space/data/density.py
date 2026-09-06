"""Turn YOLO bounding-box labels into head-point Gaussian density maps.

Density-map regression (CSRNet family) is the standard approach for crowd
counting: instead of predicting a single scalar, the network predicts a 2-D map
whose integral equals the person count. It degrades gracefully under occlusion
because each person contributes mass even when their box overlaps a neighbour.

We approximate a head point as the top-centre of each person box (``cx``,
``y_top``) and place a fixed-sigma Gaussian there. A fixed sigma is adequate for
single-room, roughly fronto-parallel CCTV framing; geometry-adaptive kernels
would be the next refinement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def read_yolo_label(path: Path) -> np.ndarray:
    """Return an (N, 4) array of ``xc, yc, w, h`` in normalised coords (0..1)."""
    if not path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            rows.append([float(x) for x in parts[1:5]])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def head_points(boxes_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map normalised boxes to integer head pixel coordinates (top-centre)."""
    if boxes_norm.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    xc = boxes_norm[:, 0] * width
    y_top = (boxes_norm[:, 1] - boxes_norm[:, 3] / 2.0) * height
    pts = np.stack([xc, y_top], axis=1)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    return np.round(pts).astype(np.int64)


def density_map(
    boxes_norm: np.ndarray,
    out_hw: tuple[int, int],
    sigma: float = 4.0,
) -> np.ndarray:
    """Build a float32 density map of shape ``out_hw`` that sums to ``len(boxes)``.

    The impulse image is blurred with a normalised Gaussian, so the total mass is
    exactly conserved (``map.sum() == N`` up to float error).
    """
    h, w = out_hw
    canvas = np.zeros((h, w), dtype=np.float32)
    pts = head_points(boxes_norm, w, h)
    for x, y in pts:
        canvas[y, x] += 1.0
    if pts.shape[0] == 0:
        return canvas
    blurred = gaussian_filter(canvas, sigma=sigma, mode="constant")
    # Re-normalise to defend against mass lost at the borders.
    total = blurred.sum()
    if total > 0:
        blurred *= pts.shape[0] / total
    return blurred


def count_from_label(path: Path) -> int:
    """Ground-truth person count = number of boxes in the YOLO label file."""
    return int(read_yolo_label(path).shape[0])
