"""Unified evaluation: run every available model on the TEST split, write one
predictions table and one metrics report. Nothing is hard-coded — the numbers in
docs/results.md are produced by this script.

Outputs:
    runs/eval/predictions.parquet   # columns: image, y_true, yolo, resnet, xgboost, density
    runs/eval/metrics.json          # per-model evaluate_counts() output
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from smart_space.config import (
    DENSITY_RUN_DIR,
    EVAL_DIR,
    PROCESSED_DIR,
    RESNET_RUN_DIR,
    XGB_RUN_DIR,
    data_config,
    ensure_dirs,
    resolve_device,
)
from smart_space.data.dataset import OccupancyCountDataset
from smart_space.data.density import count_from_label
from smart_space.evaluation.metrics import evaluate_counts
from smart_space.models.density_net import CSRNetLite
from smart_space.models.resnet_regressor import (
    build_feature_extractor,
    build_regressor,
    extract_features,
)
from smart_space.models.xgb_regressor import load_xgb, predict_counts


def _test_dir():
    return PROCESSED_DIR / "test"


def _resnet_preds(device: str, img_size: int) -> np.ndarray | None:
    ckpt = RESNET_RUN_DIR / "resnet18_regressor.pt"
    if not ckpt.exists():
        return None
    net = build_regressor(pretrained=False)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.to(device).eval()
    ds = OccupancyCountDataset(_test_dir(), img_size=img_size)
    loader = DataLoader(ds, batch_size=16)
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(net(x.to(device)).squeeze(1).cpu().numpy())
    return np.clip(np.concatenate(out), 0, None)


def _xgb_preds(device: str, img_size: int) -> np.ndarray | None:
    model_path = XGB_RUN_DIR / "xgb_occupancy.json"
    if not model_path.exists():
        return None
    ds = OccupancyCountDataset(_test_dir(), img_size=img_size)
    loader = DataLoader(ds, batch_size=16)
    backbone = build_feature_extractor(pretrained=True)
    X, _ = extract_features(backbone, loader, device)
    return predict_counts(load_xgb(model_path), X)


def _density_preds(device: str, img_size: int) -> np.ndarray | None:
    ckpt = DENSITY_RUN_DIR / "csrnet_lite.pt"
    if not ckpt.exists():
        return None
    net = CSRNetLite(pretrained=False)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.to(device).eval()
    ds = OccupancyCountDataset(_test_dir(), img_size=img_size)
    loader = DataLoader(ds, batch_size=8)
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(net.predict_count(x.to(device)).cpu().numpy())
    return np.clip(np.concatenate(out), 0, None)


def _yolo_preds() -> np.ndarray | None:
    try:
        from smart_space.models.yolo_counter import YoloCounter
    except Exception:
        return None
    try:
        counter = YoloCounter()
    except FileNotFoundError:
        return None
    ds = OccupancyCountDataset(_test_dir())
    return np.array([counter.count(str(p)) for p in ds.img_paths], dtype=float)


def run() -> dict:
    ensure_dirs()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    cfg = data_config()
    device = resolve_device()

    base_ds = OccupancyCountDataset(_test_dir(), img_size=cfg.img_size)
    images = [p.name for p in base_ds.img_paths]
    y_true = np.array(
        [count_from_label(base_ds.labels_dir / f"{p.stem}.txt") for p in base_ds.img_paths],
        dtype=float,
    )

    columns: dict[str, np.ndarray] = {"image": images, "y_true": y_true}
    for name, preds in {
        "yolo": _yolo_preds(),
        "resnet": _resnet_preds(device, cfg.img_size),
        "xgboost": _xgb_preds(device, cfg.img_size),
        "density": _density_preds(device, cfg.img_size),
    }.items():
        if preds is not None and len(preds) == len(y_true):
            columns[name] = preds

    df = pd.DataFrame(columns)
    df.to_parquet(EVAL_DIR / "predictions.parquet", index=False)

    metrics = {
        m: evaluate_counts(y_true, df[m].to_numpy())
        for m in df.columns
        if m not in {"image", "y_true"}
    }
    (EVAL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Evaluated {len(y_true)} test images across models: {list(metrics)}")
    for name, res in metrics.items():
        o = res["overall"]
        print(
            f"  {name:9s}  MAE {o['mae']['value']:.3f} "
            f"[{o['mae']['ci_low']:.2f}, {o['mae']['ci_high']:.2f}]  "
            f"RMSE {o['rmse']['value']:.3f}  bias {res['mean_bias']:+.2f}"
        )
    return metrics


if __name__ == "__main__":
    run()
