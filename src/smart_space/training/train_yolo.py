"""Fine-tune YOLOv8n on the people dataset (transfer learning from COCO)."""

from __future__ import annotations

from smart_space.config import (
    YOLO_RUN_DIR,
    load_yaml,
    resolve_device,
    write_yolo_data_yaml,
)


def train() -> str:
    from ultralytics import YOLO

    cfg = load_yaml("yolo.yaml")
    data_yaml = write_yolo_data_yaml()
    device = resolve_device()
    # Ultralytics wants "mps"/"cpu"/0 — map "cuda" -> 0.
    yolo_device = 0 if device == "cuda" else device

    model = YOLO(cfg.get("model", "yolov8n.pt"))
    model.train(
        data=str(data_yaml),
        epochs=int(cfg.get("epochs", 40)),
        imgsz=int(cfg.get("imgsz", 640)),
        batch=int(cfg.get("batch", 16)),
        patience=int(cfg.get("patience", 15)),
        device=yolo_device,
        project=str(YOLO_RUN_DIR),
        name="occupancy",
        exist_ok=True,
        verbose=True,
    )
    model.val(data=str(data_yaml), split="test", device=yolo_device)
    weights = YOLO_RUN_DIR / "occupancy" / "weights" / "best.pt"
    print(f"YOLO weights -> {weights}")
    return str(weights)


if __name__ == "__main__":
    train()
