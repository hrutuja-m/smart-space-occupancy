# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision
pip install -e ".[serve,edge,dev]"
pre-commit install   # optional
```

## Before opening a PR

```bash
ruff check src tests
SMARTSPACE_NO_MLFLOW=1 pytest
```

Both run in CI (`.github/workflows/ci.yml`).

## Conventions

- Source lives under `src/smart_space/`; keep new modules typed and small.
- No hard-coded metrics or paths — add to `configs/*.yaml` and `config.py`.
- New deterministic logic (tracking, metrics, data transforms) needs a unit test.
- Training scripts expose a `train() -> str` returning the checkpoint path and
  wrap the loop in `mlflow_run(...)`.

## Environment variables

| var | effect |
| --- | --- |
| `SMARTSPACE_DEVICE` | force `cpu` / `mps` / `cuda` |
| `SMARTSPACE_NO_MLFLOW` | disable MLflow logging (CI, Docker) |
