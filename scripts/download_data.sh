#!/usr/bin/env bash
# Download the "People Detection Dataset v16" (Roboflow Universe, CC BY 4.0).
#
# Roboflow requires a free account + API key for programmatic download. Two options:
#
# 1) Manual (no key):
#    - Visit https://universe.roboflow.com/up-care-thesis/people-detection-dataset/dataset/16
#    - Download -> format "YOLOv8" -> unzip into external_datasets/people_detection/
#
# 2) Roboflow SDK (needs ROBOFLOW_API_KEY in your env):
set -euo pipefail

DEST="external_datasets/people_detection"
mkdir -p "$DEST"

if [[ -n "${ROBOFLOW_API_KEY:-}" ]]; then
  pip install --quiet roboflow
  python - <<'PY'
import os
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("up-care-thesis").project("people-detection-dataset")
project.version(16).download("yolov8", location="external_datasets/people_detection")
PY
  echo "Downloaded to $DEST"
else
  echo "ROBOFLOW_API_KEY not set — follow the manual steps above, then run:"
  echo "  smartspace data prepare"
fi
