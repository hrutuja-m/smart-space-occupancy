"""FastAPI occupancy inference service.

    uvicorn smart_space.serving.api:app --reload

Endpoints:
    GET  /health
    GET  /models                       -> which model checkpoints are available
    POST /predict?model=xgboost        -> multipart image  -> {count, model, ...}
"""

from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image

from smart_space.config import (
    DENSITY_RUN_DIR,
    RESNET_RUN_DIR,
    XGB_RUN_DIR,
    YOLO_RUN_DIR,
)
from smart_space.inference.predictor import MODELS, OccupancyPredictor

app = FastAPI(title="Smart Space Occupancy API", version="0.2.0")


def _available() -> dict[str, bool]:
    return {
        "yolo": any(YOLO_RUN_DIR.glob("**/weights/*.pt")),
        "resnet": (RESNET_RUN_DIR / "resnet18_regressor.pt").exists(),
        "xgboost": (XGB_RUN_DIR / "xgb_occupancy.json").exists(),
        "density": (DENSITY_RUN_DIR / "csrnet_lite.pt").exists(),
    }


@lru_cache(maxsize=len(MODELS))
def _get_predictor(model: str) -> OccupancyPredictor:
    return OccupancyPredictor.load(model)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def models() -> dict:
    return {"available": _available(), "supported": list(MODELS)}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query("xgboost", enum=list(MODELS)),
) -> dict:
    if not _available().get(model, False):
        raise HTTPException(status_code=409, detail=f"model '{model}' has no trained checkpoint")
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc

    result = _get_predictor(model).predict(np.array(image)[..., ::-1])
    payload = {"model": result.model, "count": result.rounded(), "count_raw": round(result.count, 2)}
    if result.boxes is not None:
        payload["boxes"] = result.boxes
    return payload
