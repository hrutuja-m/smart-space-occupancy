.PHONY: help install data train evaluate serve app test lint docker clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## editable install with all extras
	pip install torch torchvision && pip install -e ".[serve,edge,dev]"

data:  ## download (manual) then ingest + stratified split
	python -m smart_space.data.ingest && python -m smart_space.data.split

train:  ## train all four models
	smartspace train all

evaluate:  ## unified test-set eval + diagnostic figures
	smartspace evaluate

serve:  ## FastAPI inference service on :8000
	uvicorn smart_space.serving.api:app --reload

app:  ## Streamlit demo
	streamlit run src/smart_space/serving/app.py

test:  ## run unit tests
	SMARTSPACE_NO_MLFLOW=1 pytest

lint:  ## ruff
	ruff check src tests

docker:  ## build the inference image
	docker build -t smart-space-occupancy .

clean:
	rm -rf runs/eval runs/export .pytest_cache .ruff_cache
