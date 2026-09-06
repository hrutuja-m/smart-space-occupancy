# ── Smart Space Occupancy — inference image ───────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SMARTSPACE_DEVICE=cpu \
    SMARTSPACE_NO_MLFLOW=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.29" "python-multipart>=0.0.9" \
      "onnx>=1.16" "onnxruntime>=1.17"

COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir -e . --no-deps

# Model checkpoints are mounted at runtime:  -v $(pwd)/runs:/app/runs
EXPOSE 8000
CMD ["uvicorn", "smart_space.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
