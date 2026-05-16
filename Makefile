.PHONY: help install install-dev download-data ingest-manuals train evaluate monitor evaluate-rag serve ui test lint format docker-build docker-up clean

PYTHON := python
PIP := pip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

install: ## Install package + runtime deps
	$(PIP) install -e .

install-dev: ## Install with dev tools (pytest, ruff, mypy)
	$(PIP) install -e ".[dev]"

download-data: ## Download NASA C-MAPSS dataset
	bash scripts/download_data.sh

ingest-manuals: ## Build vector index from PDFs in data/manuals/
	$(PYTHON) -m src.rag.ingestion --manuals-dir data/manuals --persist-dir chroma_db

train: ## Train LSTM autoencoder, log to MLflow
	$(PYTHON) -m scripts.train --subset FD001 --epochs 50

evaluate: ## Run evaluation on test set
	$(PYTHON) -m scripts.evaluate --subset FD001

monitor: ## Run Evidently drift monitoring, saves reports/ to reports/
	$(PYTHON) -m scripts.monitor --subset FD001

evaluate-rag: ## Run RAGAS evaluation on the RAG pipeline (requires ChromaDB + GROQ_API_KEY)
	$(PYTHON) -m scripts.evaluate_rag

serve: ## Run FastAPI server on :8000
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

ui: ## Run Streamlit UI on :8501
	streamlit run src/ui/streamlit_app.py

test: ## Run pytest with coverage
	pytest

lint: ## Run ruff + mypy
	ruff check src tests
	mypy src

format: ## Auto-format with ruff
	ruff check --fix src tests
	ruff format src tests

docker-build: ## Build container
	docker build -t predictive-maintenance-copilot .

docker-up: ## Start full stack via docker-compose
	docker compose up --build

clean: ## Remove caches and build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
