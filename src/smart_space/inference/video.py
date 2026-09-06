"""Video -> occupancy time series.

Two modes:
  * "instant"  : per-sampled-frame YOLO person count (smoothed)
  * "flow"     : YOLO + CentroidTracker + LineCounter -> net occupancy from
                 door crossings (needs a --line argument)

Outputs a pandas DataFrame (t_seconds, count) and optionally a CSV / annotated MP4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from smart_space.models.yolo_counter import YoloCounter
from smart_space.tracking.centroid_tracker import CentroidTracker
from smart_space.tracking.line_counter import LineCounter


@dataclass
class VideoConfig:
    sample_every: int = 15          # process 1 in N frames
    smooth_window: int = 5          # rolling-median window on the count series
    mode: str = "instant"           # "instant" | "flow"
    line: tuple[tuple[float, float], tuple[float, float]] | None = None
    start_count: int = 0
    conf: float = 0.25


@dataclass
class VideoResult:
    series: pd.DataFrame
    entries: int = 0
    exits: int = 0
    peak: int = 0
    mean: float = 0.0
    meta: dict = field(default_factory=dict)


def analyze_video(path: str | Path, cfg: VideoConfig | None = None, weights=None) -> VideoResult:
    import cv2

    cfg = cfg or VideoConfig()
    counter = YoloCounter(weights, conf=cfg.conf)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tracker = CentroidTracker() if cfg.mode == "flow" else None
    line_counter = LineCounter(cfg.line, cfg.start_count) if (cfg.mode == "flow" and cfg.line) else None

    rows: list[tuple[float, float]] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % cfg.sample_every == 0:
            t = frame_idx / fps
            boxes = counter.detect(frame)
            if cfg.mode == "flow" and tracker is not None:
                tracked = tracker.update(boxes)
                if line_counter is not None:
                    line_counter.update(tracked)
                    count = line_counter.occupancy
                else:
                    count = len(tracked)
            else:
                count = len(boxes)
            rows.append((t, float(count)))
        frame_idx += 1
    cap.release()

    df = pd.DataFrame(rows, columns=["t_seconds", "count"])
    if cfg.smooth_window > 1 and len(df) >= cfg.smooth_window:
        df["count"] = df["count"].rolling(cfg.smooth_window, center=True, min_periods=1).median()

    return VideoResult(
        series=df,
        entries=line_counter.entries if line_counter else 0,
        exits=line_counter.exits if line_counter else 0,
        peak=int(df["count"].max()) if len(df) else 0,
        mean=float(df["count"].mean()) if len(df) else 0.0,
        meta={"fps": fps, "frames": frame_idx, "sampled": len(df), "mode": cfg.mode},
    )
