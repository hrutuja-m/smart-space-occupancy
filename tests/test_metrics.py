import numpy as np

from smart_space.evaluation.metrics import _mae, bootstrap_ci, evaluate_counts


def test_perfect_predictions():
    y = np.array([0, 1, 5, 12, 20], dtype=float)
    res = evaluate_counts(y, y.copy(), n_boot=200)
    assert res["overall"]["mae"]["value"] == 0
    assert res["overall"]["exact_match_pct"]["value"] == 100
    assert res["mean_bias"] == 0


def test_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 20, 200).astype(float)
    yp = yt + rng.normal(0, 2, 200)
    ci = bootstrap_ci(yt, yp, _mae, n_boot=500)
    assert ci.ci_low <= ci.value <= ci.ci_high


def test_bias_sign():
    yt = np.array([5.0, 5.0, 5.0, 5.0])
    yp = np.array([7.0, 7.0, 7.0, 7.0])
    res = evaluate_counts(yt, yp, n_boot=100)
    assert res["mean_bias"] == 2.0


def test_per_bucket_present():
    yt = np.array([0, 0, 2, 8, 15, 25], dtype=float)
    res = evaluate_counts(yt, yt + 1, n_boot=100)
    assert {"empty", "low", "medium", "high"} <= set(res["per_bucket"])
