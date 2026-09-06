# Résumé / portfolio framing

## One-liner

> Built an end-to-end computer-vision system for indoor **occupancy monitoring** —
> benchmarking object detection, CNN regression, gradient boosting on deep
> features, and CSRNet-style density estimation — with a tracking-based
> people-flow counter, uncertainty-aware evaluation, MLflow tracking, a
> FastAPI/Streamlit serving layer, and an ONNX edge-latency benchmark.

## Résumé bullets (pick 2–3)

- Designed a **4-model people-counting benchmark** (YOLOv8, ResNet18 regressor,
  XGBoost-on-CNN-embeddings, CSRNet-lite density net) on a density-stratified
  split; the density estimator won (**MAE 0.76**, 95% CI [0.62, 0.92]) while
  detection-based counting (YOLO) under-counts by ~1.1 people and doubles its
  error in crowded rooms (MAE 2.4) due to occlusion — quantifying the
  detection-vs-regression-vs-density trade-off with bootstrap confidence intervals.
- Built the full ML lifecycle in a single `smartspace` CLI: config-driven data
  ingest + **density-bucket-stratified splitting**, per-model training with
  **MLflow** experiment tracking, and a unified evaluation that reports every
  metric with **bootstrap 95% confidence intervals** and a per-density-bucket
  error breakdown.
- Implemented a real-time **people-flow pipeline** — YOLOv8 detection → centroid
  tracker with occlusion grace period → oriented doorway line-counter — producing
  a live room-occupancy time series from video.
- Shipped the models behind a **FastAPI** inference service and a **Streamlit**
  comparison dashboard; **containerised** with Docker Compose; exported to
  **ONNX** and benchmarked CPU p50/p95 latency to validate edge (Raspberry Pi /
  Jetson) deployment.
- Added a **pytest** suite for the deterministic components and **GitHub Actions
  CI** (ruff + tests); documented model card, data card, and known
  augmentation-leakage limitations.

## Skills this demonstrates

| Area | Evidence in repo |
| --- | --- |
| Deep learning (PyTorch) | ResNet regressor, CSRNet-lite (dilated convs, VGG frontend), transfer learning |
| Classic ML | XGBoost regression on frozen embeddings, early stopping, feature importance |
| Object detection | YOLOv8 fine-tuning from COCO, confidence-thresholded counting |
| Multi-object tracking | centroid association, disappeared-track handling, line-crossing logic |
| Evaluation rigor | bootstrap CIs, stratified splits, calibration curves, bias analysis, ROC |
| MLOps | MLflow, Docker, CI, CLI, config management, ONNX export + benchmarking |
| Software engineering | typed `src/` package, unit tests, reproducible scripts |
| Communication | architecture diagram, model/data cards, honest limitations section |

## Interview talking points

1. **Why four models?** Each represents a real design choice under different
   constraints — interpretability vs robustness vs data budget vs edge cost.
   Detection localises but breaks under occlusion; regression is cheap but blind;
   density estimation is the crowd-counting SOTA compromise; GBM-on-embeddings
   wins when labelled data is scarce.
2. **Why bootstrap CIs?** With ~120 test images, a single MAE number is noise.
   The CI tells you whether "model A beats model B" is real.
3. **Why stratified splitting?** Counts are heavily skewed; a random split can
   leave the test set with no crowded rooms and make the headline metric a lie.
4. **What's the honest weakness?** Roboflow's per-image augmentation means near
   duplicates can straddle train/test — metrics are optimistic until the split is
   grouped by source image. I documented it rather than hiding it.
5. **Edge story:** yolov8n is 3.2M params; the ResNet regressor exports to ONNX
   and runs at ~16 ms p50 / 62 fps on a laptop CPU (`smartspace export` prints
   p50/p95) → comfortably real-time on a Pi-class device, frames discarded in
   memory (privacy by design).
