# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.0.0] - 2026-05-17

### Added

**Core ML pipeline**
- LSTM autoencoder for multivariate time-series anomaly detection on NASA C-MAPSS data
- Reconstruction-error scoring with F1-optimised threshold
- MLflow experiment tracking for training runs
- Sliding-window preprocessing with per-unit StandardScaler

**RAG pipeline**
- PDF ingestion into ChromaDB via LlamaIndex (273k chunks)
- Hybrid retriever: BM25 + dense embeddings (sentence-transformers) + Reciprocal Rank Fusion
- Structured LLM prompt with typed JSON output (pydantic AnomalyAlert)
- Multi-provider LLM support: Ollama (default), Groq, OpenAI, HuggingFace

**API and UI**
- FastAPI endpoint (POST /predict) returning anomaly score + LLM-generated alert
- Streamlit dashboard for interactive sensor window inspection

**MLOps**
- Evidently drift monitoring for sensor and anomaly-score drift
- RAGAS evaluation on golden Q&A set (RAG quality metrics)
- Docker and Docker Compose support
- Deploy to Hugging Face Spaces

**Quality and CI**
- Test suite: 86 tests, 88% coverage (excluding UI and CLI entry points)
- Synthetic C-MAPSS fixtures and fake model checkpoints for offline testing
- GIVEN/WHEN/THEN docstrings on all tests
- pre-commit hook: ruff lint + format check + mypy + pytest
- GitHub Actions CI on Python 3.12
- Versioned Docker image tagging on git tag push (v*.*.*)
