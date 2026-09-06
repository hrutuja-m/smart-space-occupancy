import io

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from smart_space.serving.api import app  # noqa: E402

client = TestClient(app)


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 120, 120)).save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_models_lists_support():
    body = client.get("/models").json()
    assert set(body["supported"]) == {"yolo", "resnet", "xgboost", "density"}


def test_predict_without_checkpoint_returns_409():
    r = client.post("/predict?model=density", files={"file": ("x.png", _png_bytes(), "image/png")})
    # 409 when no checkpoint present (CI has none); 200 if a dev has trained one.
    assert r.status_code in (200, 409)


def test_predict_rejects_non_image():
    r = client.post("/predict?model=xgboost", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code in (400, 409)
