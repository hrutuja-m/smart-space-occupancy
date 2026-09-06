# Smart Space Occupancy Monitoring

Estimate **how many people are in a room** from a single fixed camera — and
compare four fundamentally different ML strategies for the same task, end to end:
data pipeline → training → uncertainty-aware evaluation → tracking → REST/UI
serving → ONNX edge benchmark.

<p align="left">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="ci" src="https://img.shields.io/badge/CI-ruff%20%2B%20pytest-informational">
</p>

---

## Why this exists (the real-world logic)

Occupancy sensing drives real building systems:

| Consumer | What it does with a count |
| --- | --- |
| **HVAC / energy** | ventilate & cool to actual load, not a fixed schedule — the largest ROI lever in commercial buildings |
| **Space utilisation** | "which rooms / desks are actually used" for real-estate decisions |
| **Retail** | conversion = transactions ÷ footfall |
| **Safety** | fire-code capacity, crowd-density alerts |

A single camera covers a whole room, gives a *count* (not just presence), and —
done right — runs **on-device and never stores a frame**. `yolov8n` is exactly
the model class you would deploy on a Raspberry Pi / Jetson for that.

## Four models, one interface

| Model | Family | How it counts | Strength |
| --- | --- | --- | --- |
| **YOLOv8n** | object detection | `count = #person boxes` | localises people, edge-tiny, feeds the tracker |
| **ResNet18 regressor** | direct CNN regression | image → scalar | cheap, robust to overlap |
| **XGBoost + ResNet18 features** | frozen backbone + GBM | 512-d embedding → scalar | best small-data accuracy |
| **CSRNet-lite** | density-map estimation | image → density map, `count = Σ` | crowd-counting SOTA family, degrades gracefully |

Plus a **centroid tracker + doorway line-counter** that turns per-frame detection
into a live "people currently in the room" signal from door crossings.

See [`docs/architecture.md`](docs/architecture.md) for the full diagram.

## Quickstart

```bash
# 1. install (CPU/MPS/GPU auto-detected)
pip install torch torchvision
pip install -e ".[serve,edge,dev]"

# 2. get data  (Roboflow People Detection v16, CC BY 4.0)
bash scripts/download_data.sh          # or download the YOLOv8 export manually
smartspace data prepare                # ingest + stratified split + density maps

# 3. train (or skip — checkpoints load if present)
smartspace train all                   # yolo, resnet, xgboost, density
#   ↳ GPU recommended for yolo + density; resnet/xgboost are fine on Apple MPS

# 4. evaluate — one test pass, all models, with bootstrap CIs
smartspace evaluate                    # -> runs/eval/{metrics.json, figures/}

# 5. use it
smartspace predict room.jpg --model xgboost
smartspace video office.mp4 --mode flow --line 0,240,640,240
smartspace export --model resnet       # ONNX + CPU latency benchmark

# 6. serve
uvicorn smart_space.serving.api:app --reload      # REST  :8000
streamlit run src/smart_space/serving/app.py      # demo UI
```

## Results

Test set = 122 images (0–23 people), 800-image local subset, all four models
trained on Apple-Silicon MPS. Regenerate with `smartspace evaluate`; full
breakdown in [`docs/results.md`](docs/results.md).

| Model | MAE | MAE 95% CI | RMSE | Exact % | Bias |
| --- | --- | --- | --- | --- | --- |
| **CSRNet-lite (density)** | **0.76** | [0.62, 0.92] | 1.14 | 48 | +0.06 |
| ResNet18 regressor | 0.92 | [0.73, 1.14] | 1.47 | 50 | +0.07 |
| XGBoost + ResNet feats | 1.43 | [1.18, 1.70] | 2.04 | 33 | +0.13 |
| YOLOv8n detection | 1.79 | [1.38, 2.21] | 2.98 | 34 | **−1.11** |

**The interesting bit** is per-bucket: YOLO is *perfect on empty rooms* (MAE 0.00)
but *worst when crowded* (MAE 2.41, true counts ≥13) — occlusion makes the
detector systematically under-count (negative bias). The density estimator, which
never has to separate individuals, is the most consistent across every bucket
(0.15 → 0.96) and wins overall. This is exactly the detection-vs-regression vs
density trade-off the benchmark was built to expose.

Every metric carries a bootstrap 95% CI and a per-density-bucket breakdown
because the test set is small — a bare MAE would not be trustworthy. Absolute
numbers are optimistic (augmentation leakage — see
[`docs/data_card.md`](docs/data_card.md)); the *ranking* and *per-bucket shape*
are the takeaways.

## Repo layout

```
src/smart_space/
  config.py            single source of truth for paths, device, buckets
  data/                ingest · stratified split · datasets · density maps
  models/              yolo_counter · resnet_regressor · xgb_regressor · density_net (CSRNet-lite)
  tracking/            centroid_tracker · line_counter
  training/            one script per model, MLflow-logged
  evaluation/          metrics (bootstrap CI, per-bucket) · evaluate · plots
  inference/           unified predictor · video pipeline · onnx export + benchmark
  serving/             FastAPI api · Streamlit app
  cli.py               `smartspace` command
tests/                 tracker, line-counter, metrics, density, api
docs/                  architecture · model_card · data_card · results
```

## Engineering

* **Config-driven**, no hard-coded metrics — `configs/*.yaml`, overridable from the CLI.
* **Experiment tracking** — MLflow (`docker compose --profile tracking up mlflow`).
* **Tested** — `pytest` on the deterministic components; **CI** runs ruff + pytest on every push.
* **Containerised** — CPU inference image (`Dockerfile`), `docker-compose` for api + MLflow.
* **Reproducible** — seeded splits, `scripts/run_full_pipeline.sh` for the whole chain.

## Honest limitations

* Single training dataset → **no cross-camera guarantee** without fine-tuning.
* Roboflow ships 2 augmented copies per source image; the split is stratified by
  count but not grouped by source id, so absolute metrics are **optimistic**
  (see [`docs/data_card.md`](docs/data_card.md)).
* CSRNet-lite and YOLO want a GPU for a full run; local defaults use an 800-image
  subset so everything is runnable on a laptop.

## License

MIT (code). Dataset is CC BY 4.0 — attribution in [`docs/data_card.md`](docs/data_card.md).
