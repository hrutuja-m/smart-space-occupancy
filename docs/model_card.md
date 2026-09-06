# Model Card — Smart Space Occupancy counters

## Overview

Four interchangeable models that map a room image to a person count. All are
served behind one `OccupancyPredictor` API and evaluated by one script.

| Model | Params | Train cost (800-img subset) | Artifact |
| --- | --- | --- | --- |
| YOLOv8n (fine-tuned) | 3.2 M | ~40 epochs, GPU strongly preferred | `runs/yolo/occupancy/weights/best.pt` |
| ResNet18 regressor | 11.2 M | ~25 epochs, minutes on Apple MPS | `runs/resnet/resnet18_regressor.pt` |
| XGBoost + ResNet18 features | 400 trees / 512 feats | < 1 min (frozen backbone) | `runs/xgboost/xgb_occupancy.json` |
| CSRNet-lite | 16.3 M | ~40 epochs, GPU recommended | `runs/density/csrnet_lite.pt` |

## Intended use

* **In scope:** aggregate people-counting from a fixed indoor camera for building
  automation, space-utilisation analytics, capacity monitoring.
* **Out of scope:** identifying individuals, tracking identity across sessions,
  any face-recognition or demographic inference, safety-critical headcount
  (evacuation) where a miss is unacceptable, outdoor / wide-area crowd estimation.

## Metrics

Reported by `smart_space/evaluation/evaluate.py` on the held-out test split:

* **MAE**, **RMSE**, **MAPE**, exact-match % — each with a bootstrap 95% CI.
* **Mean bias** (signed) — systematic over/under-count.
* Per-density-bucket MAE/RMSE.

Current numbers: see [`results.md`](results.md) (regenerate with `smartspace evaluate`).

## Training data

Roboflow *People Detection Dataset v16* (CC BY 4.0). See [`data_card.md`](data_card.md),
including the **augmentation-leakage caveat** — absolute metrics are optimistic
until the split is grouped by pre-augmentation source image.

## Ethical considerations & limitations

* Designed for **edge, frame-discarding** deployment; only the integer count is
  emitted.
* Single-dataset training → **no cross-camera guarantee**; treat a new site as
  requiring fine-tuning + recalibration.
* Detection-based counting (YOLO) collapses under heavy occlusion; density and
  regression models are the fallback for crowded scenes.
* Not validated for fairness across appearance; the task avoids person attributes
  but annotation coverage of e.g. seated vs standing, mobility aids, children is
  unknown.

## Maintenance

Retrain when: the camera / mounting changes, the target space type changes
(open-plan vs meeting room), or evaluation bias drifts. CI runs lint + unit tests
on every push; model retraining is manual (`scripts/run_full_pipeline.sh`).
