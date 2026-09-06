"""Shared training helpers: MLflow context + seeding."""

from __future__ import annotations

import contextlib
import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@contextlib.contextmanager
def mlflow_run(experiment: str, run_name: str, params: dict | None = None):
    """Log to MLflow if installed & not disabled; otherwise a no-op context."""
    if os.getenv("SMARTSPACE_NO_MLFLOW"):
        yield _NullRun()
        return
    try:
        import mlflow
    except ImportError:
        yield _NullRun()
        return

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        yield mlflow


class _NullRun:
    def log_metric(self, *_, **__):  # noqa: D401
        pass

    def log_metrics(self, *_, **__):
        pass

    def log_artifact(self, *_, **__):
        pass

    def log_params(self, *_, **__):
        pass
