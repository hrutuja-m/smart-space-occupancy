"""Diagnostic plots from ``runs/eval/predictions.parquet``.

Generates (into runs/eval/figures/):
  * pred_vs_true.png     — scatter with y=x reference, per model
  * error_by_bucket.png  — MAE per density bucket, per model
  * calibration.png      — binned mean prediction vs mean truth
  * roc_occupied.png     — occupied(count>0) vs empty ROC, per model
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import auc, roc_curve  # noqa: E402

from smart_space.config import EVAL_DIR, bucket_for_count  # noqa: E402

FIG_DIR = EVAL_DIR / "figures"
MODEL_COLS = ("yolo", "resnet", "xgboost", "density")


def _models(df: pd.DataFrame) -> list[str]:
    return [c for c in MODEL_COLS if c in df.columns]


def pred_vs_true(df: pd.DataFrame) -> None:
    models = _models(df)
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), squeeze=False)
    hi = df["y_true"].max()
    for ax, m in zip(axes[0], models):
        ax.scatter(df["y_true"], df[m], s=14, alpha=0.6)
        ax.plot([0, hi], [0, hi], "k--", lw=1)
        ax.set_title(m)
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
    fig.suptitle("Predicted vs true count (test)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pred_vs_true.png", dpi=150)
    plt.close(fig)


def error_by_bucket(df: pd.DataFrame) -> None:
    buckets = df["y_true"].round().astype(int).map(bucket_for_count)
    order = ["empty", "low", "medium", "high"]
    models = _models(df)
    x = np.arange(len(order))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, m in enumerate(models):
        err = (df[m] - df["y_true"]).abs()
        maes = [err[buckets == b].mean() if (buckets == b).any() else 0 for b in order]
        ax.bar(x + i * w, maes, w, label=m)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(order)
    ax.set_ylabel("MAE")
    ax.set_title("Error by density bucket")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_by_bucket.png", dpi=150)
    plt.close(fig)


def calibration(df: pd.DataFrame) -> None:
    models = _models(df)
    fig, ax = plt.subplots(figsize=(6, 5))
    bins = np.linspace(0, df["y_true"].max() + 1, 8)
    centers = (bins[:-1] + bins[1:]) / 2
    idx = np.digitize(df["y_true"], bins) - 1
    for m in models:
        means = [df[m][idx == b].mean() if (idx == b).any() else np.nan for b in range(len(centers))]
        ax.plot(centers, means, marker="o", label=m)
    ax.plot(centers, centers, "k--", lw=1, label="perfect")
    ax.set_xlabel("true count (bin centre)")
    ax.set_ylabel("mean predicted count")
    ax.set_title("Calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration.png", dpi=150)
    plt.close(fig)


def roc_occupied(df: pd.DataFrame) -> None:
    y_bin = (df["y_true"] > 0).astype(int)
    if y_bin.nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in _models(df):
        fpr, tpr, _ = roc_curve(y_bin, df[m])
        ax.plot(fpr, tpr, label=f"{m} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Occupied vs empty ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_occupied.png", dpi=150)
    plt.close(fig)


def generate_all(parquet: Path | None = None) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(parquet or EVAL_DIR / "predictions.parquet")
    pred_vs_true(df)
    error_by_bucket(df)
    calibration(df)
    roc_occupied(df)
    print(f"Wrote figures -> {FIG_DIR}")
    return FIG_DIR


if __name__ == "__main__":
    generate_all()
