"""Counting metrics with uncertainty.

With a test set of only tens–hundreds of images, a bare MAE number is not
trustworthy. Every metric here ships with a bootstrap 95% confidence interval,
and errors are additionally broken down by density bucket so "great on empty
rooms, useless when crowded" cannot hide behind an average.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from smart_space.config import DENSITY_BUCKETS, bucket_for_count


@dataclass
class MetricWithCI:
    value: float
    ci_low: float
    ci_high: float

    def as_dict(self) -> dict:
        return asdict(self)


def _mae(yt, yp):
    return float(np.mean(np.abs(yp - yt)))


def _rmse(yt, yp):
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def _mape(yt, yp):
    mask = yt > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((yp[mask] - yt[mask]) / yt[mask])) * 100.0)


def _exact_match(yt, yp):
    return float(np.mean(np.rint(yp) == np.rint(yt)) * 100.0)


_FUNCS = {"mae": _mae, "rmse": _rmse, "mape": _mape, "exact_match_pct": _exact_match}


def bootstrap_ci(yt, yp, fn, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> MetricWithCI:
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    rng = np.random.default_rng(seed)
    n = len(yt)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = fn(yt[idx], yp[idx])
    return MetricWithCI(
        value=fn(yt, yp),
        ci_low=float(np.nanpercentile(stats, 100 * alpha / 2)),
        ci_high=float(np.nanpercentile(stats, 100 * (1 - alpha / 2))),
    )


def evaluate_counts(y_true, y_pred, n_boot: int = 2000) -> dict:
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    overall = {name: bootstrap_ci(yt, yp, fn, n_boot).as_dict() for name, fn in _FUNCS.items()}

    per_bucket: dict[str, dict] = {}
    buckets = np.array([bucket_for_count(int(round(v))) for v in yt])
    for b in DENSITY_BUCKETS:
        mask = buckets == b
        if mask.sum() == 0:
            continue
        per_bucket[b] = {
            "n": int(mask.sum()),
            "mae": _mae(yt[mask], yp[mask]),
            "rmse": _rmse(yt[mask], yp[mask]),
        }

    # Bias: does the model systematically over- or under-count?
    bias = float(np.mean(yp - yt))
    return {"overall": overall, "per_bucket": per_bucket, "mean_bias": bias, "n": int(len(yt))}
