"""XGBoost regressor on frozen ResNet18 embeddings.

Rationale: with only a few hundred labelled images, fine-tuning a deep head
overfits. Freezing an ImageNet backbone and fitting a shallow, regularised
gradient-boosted model on its 512-d embeddings is a strong small-data baseline —
and in the v1 experiments it was the most accurate of the three counters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost as xgb

DEFAULT_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    early_stopping_rounds=40,
)


def train_xgb(X_tr, y_tr, X_val, y_val, params: dict | None = None) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**(params or DEFAULT_PARAMS))
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def load_xgb(path: str | Path) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


def predict_counts(model: xgb.XGBRegressor, X: np.ndarray) -> np.ndarray:
    return np.clip(model.predict(X), 0, None)
