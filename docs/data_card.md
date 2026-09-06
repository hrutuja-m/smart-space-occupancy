# Data Card

## Source

**People Detection Dataset — v16 "FinalDataset_640_2"**
Roboflow Universe · workspace `up-care-thesis` · project `people-detection-dataset`
<https://universe.roboflow.com/up-care-thesis/people-detection-dataset/dataset/16>

* **License:** CC BY 4.0 (attribution required; commercial use permitted)
* **Size:** 3,185 images, single class `person`, YOLOv8 format
* **Preprocessing (by Roboflow):** EXIF-orientation stripped, resized to 640×640 (stretch)
* **Augmentation (by Roboflow):** 2 versions per source image — 50% vertical flip,
  brightness ±50%, Gaussian blur 0–2.5 px

## How this project uses it

| Stage | Script | Output |
| --- | --- | --- |
| Download | `scripts/download_data.sh` | `external_datasets/people_detection/` |
| Ingest | `smart_space/data/ingest.py` | `data/raw/` — all splits merged, renamed `000000.jpg`, optionally subsampled (`data.yaml: max_samples`) |
| Split | `smart_space/data/split.py` | `data/processed/{train,val,test}/` (70/15/15), stratified by density bucket; Gaussian density maps under `data/processed/density/` |

Person **count per image = number of label rows** in the YOLO `.txt`.

### Density buckets

| bucket | count range | operational meaning |
| --- | --- | --- |
| `empty` | 0 | room unused |
| `low` | 1–4 | a person or two |
| `medium` | 5–12 | a working group / small meeting |
| `high` | 13+ | crowded / near capacity |

Observed range in the current 800-image subsample: **0–23 people**, mean ≈ 12.

## Known limitations & biases

* **Augmentation leakage risk.** Roboflow produced 2 augmented copies per source
  image. Our split is stratified by count but **not grouped by source image**, so
  an augmented near-duplicate of a training image can land in test and inflate
  scores. *Mitigation for a rigorous run:* split on the pre-augmentation image id
  (not exposed in the export — would require the raw dataset) or deduplicate with
  perceptual hashing before splitting.
* **Single dataset, unknown scene diversity.** All frames come from one thesis
  project; camera height, lens, and environments are not documented. Numbers here
  do **not** transfer to an arbitrary new camera without re-calibration / fine-tuning.
* **Head points are approximated** from box top-centre for density maps; true head
  annotations would be cleaner.
* **No demographic attributes** are present or used. The task is counting, not
  identification — no face recognition, no tracking of identity across sessions.

## Ethical / privacy notes

Occupancy counting is designed to run **on the edge**: frames are processed in
memory and discarded; only the integer count leaves the device. No image storage,
no biometric identifiers. Deployments should still post notice of camera use and
comply with local law (e.g. GDPR Art. 6 legitimate-interest assessment for
workplace monitoring).
