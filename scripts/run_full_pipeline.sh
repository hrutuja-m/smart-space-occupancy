#!/usr/bin/env bash
# End-to-end: data -> train 4 models -> evaluate -> export ONNX + benchmark.
# Assumes the Roboflow export is already under external_datasets/ (see download_data.sh).
set -euo pipefail

smartspace data prepare
smartspace train yolo
smartspace train resnet
smartspace train xgboost
smartspace train density
smartspace evaluate
smartspace export --model resnet
smartspace export --model density

echo
echo "Done. Artifacts:"
echo "  runs/eval/metrics.json         — headline numbers (with CIs)"
echo "  runs/eval/figures/             — diagnostic plots"
echo "  runs/export/*_benchmark.json   — CPU latency / throughput"
