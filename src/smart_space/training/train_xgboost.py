"""Train XGBoost on frozen ResNet18 embeddings."""

from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader

from smart_space.config import PROCESSED_DIR, XGB_RUN_DIR, data_config, resolve_device, train_config
from smart_space.data.dataset import OccupancyCountDataset
from smart_space.evaluation.metrics import evaluate_counts
from smart_space.models.resnet_regressor import build_feature_extractor, extract_features
from smart_space.models.xgb_regressor import predict_counts, train_xgb
from smart_space.training.common import mlflow_run, seed_everything


def _features(split: str, backbone, device: str, img_size: int):
    ds = OccupancyCountDataset(PROCESSED_DIR / split, img_size=img_size)
    return extract_features(backbone, DataLoader(ds, batch_size=16), device)


def train() -> str:
    dcfg, tcfg = data_config(), train_config("resnet.yaml")
    device = resolve_device()
    seed_everything(dcfg.seed)

    backbone = build_feature_extractor(pretrained=True)
    X_tr, y_tr = _features("train", backbone, device, dcfg.img_size)
    X_va, y_va = _features("val", backbone, device, dcfg.img_size)
    X_te, y_te = _features("test", backbone, device, dcfg.img_size)

    with mlflow_run(tcfg.mlflow_experiment, "xgboost_resnet_features") as run:
        model = train_xgb(X_tr, y_tr, X_va, y_va)
        te_metrics = evaluate_counts(y_te, predict_counts(model, X_te))
        run.log_metrics(
            {
                "test_mae": te_metrics["overall"]["mae"]["value"],
                "test_rmse": te_metrics["overall"]["rmse"]["value"],
                "test_exact": te_metrics["overall"]["exact_match_pct"]["value"],
                "n_estimators_used": int(getattr(model, "best_iteration", 0) or model.n_estimators),
            }
        )

        XGB_RUN_DIR.mkdir(parents=True, exist_ok=True)
        out = XGB_RUN_DIR / "xgb_occupancy.json"
        model.save_model(str(out))
        np.save(XGB_RUN_DIR / "feature_importance.npy", model.feature_importances_)
        run.log_artifact(str(out))

    print(f"Test MAE {te_metrics['overall']['mae']['value']:.3f}  ->  {out}")
    return str(out)


if __name__ == "__main__":
    train()
