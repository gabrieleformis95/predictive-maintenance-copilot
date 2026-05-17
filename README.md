---
title: Predictive Maintenance Copilot
emoji: ⚙
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Predictive Maintenance Copilot

> End-to-end pipeline that detects equipment degradation from sensor data and **explains it in natural language** with citations to maintenance manuals. Built on NASA C-MAPSS turbofan engine data, with a Retrieval-Augmented Generation layer over public industrial maintenance documentation.

[![CI](https://github.com/gabrieleformis95/predictive-maintenance-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/gabrieleformis95/predictive-maintenance-copilot/actions)
[![Coverage](https://img.shields.io/badge/coverage-88%25-brightgreen.svg)](https://github.com/gabrieleformis95/predictive-maintenance-copilot/actions)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live demo](https://img.shields.io/badge/demo-HF%20Spaces-orange.svg)](https://huggingface.co/spaces/gabrieleformis/predictive-maintenance-copilot)

---

## What it does

Industrial sensor data is noisy and high-dimensional. Operators see anomaly alerts but rarely understand *why* a machine is degrading or *what to do about it*. This system closes that gap.

1. **Anomaly detection** — an LSTM autoencoder learns the normal sensor signature of a turbofan engine and flags deviations from that baseline. Validated on the NASA C-MAPSS benchmark.
2. **Contextual retrieval** — when an anomaly fires, the system extracts which sensors deviated and how, and retrieves the most relevant pages from indexed maintenance manuals (hybrid BM25 + dense retrieval with RRF fusion).
3. **Structured explanation** — an LLM produces a JSON alert with severity, probable cause, recommended action, and **citations** back to the source manual.
4. **Operator UI** — Streamlit dashboard with sensor trajectories, anomaly timeline, and alert cards.

## Architecture

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ C-MAPSS CSV  │───▶│ Preprocessing & │───▶│ LSTM Autoencoder │
│  (sensors)   │    │  windowing       │    │  (PyTorch)        │
└──────────────┘    └─────────────────┘    └────────┬─────────┘
                                                     │ reconstruction
                                                     │ error → score
                                                     ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ Maintenance  │───▶│ Chunking +       │───▶│ Hybrid retriever │
│  manuals     │    │  embeddings      │    │  (BM25 + dense)  │
│  (PDFs)      │    │  → ChromaDB      │    └────────┬─────────┘
└──────────────┘    └─────────────────┘             │
                                                     ▼
                                            ┌──────────────────┐
                                            │  LLM             │
                                            │  Ollama (local)  │
                                            │  Groq (fallback) │
                                            └────────┬─────────┘
                                                     ▼
                                  { severity, cause, action, citations[] }
                                                     │
                                                     ▼
                                            ┌──────────────────┐
                                            │  FastAPI /predict │
                                            │  Streamlit UI    │
                                            └──────────────────┘
```

## Results

| Metric | Value |
|--------|------:|
| ROC-AUC (C-MAPSS FD001) | **0.906** |
| PR-AUC | **0.789** |
| F1-optimal threshold | 0.344 |
| Anomaly horizon | last 30 cycles before failure |
| RAG faithfulness (RAGAS) | **0.917** |

Model: LSTM autoencoder, hidden\_dim=128, latent\_dim=32, window\_size=30, trained 200 epochs on FD001 healthy cycles.

### PR Curve

![PR Curve](https://raw.githubusercontent.com/gabrieleformis95/predictive-maintenance-copilot/main/docs/images/pr_curve.png)

### Dashboard

![Dashboard - sensor trajectories and anomaly score](https://raw.githubusercontent.com/gabrieleformis95/predictive-maintenance-copilot/main/docs/images/dashboard_1.png)

![Dashboard - anomaly detail](https://raw.githubusercontent.com/gabrieleformis95/predictive-maintenance-copilot/main/docs/images/dashboard_2.png)

## Quick start

```bash
# 1. Setup
make install-dev
make download-data          # NASA C-MAPSS dataset (requires Kaggle credentials)
make ingest-manuals         # build vector index from PDFs in data/manuals/

# 2. Train & evaluate
make train                  # LSTM autoencoder + MLflow tracking
make evaluate               # ROC-AUC, PR-AUC, PR curve, saves F1-optimal threshold

# 3. Run
make serve                  # FastAPI on :8000
make ui                     # Streamlit on :8501
```

Open `http://localhost:8501` in your browser.

### LLM configuration

The pipeline routes to Ollama by default with Groq as cloud fallback. Set in `.env`:

```env
LLM_PROVIDER=ollama          # ollama | groq | openai
LLM_MODEL=qwen2.5:7b
GROQ_API_KEY=your_key_here   # optional fallback
```

Pull the Ollama model before running:

```bash
ollama pull qwen2.5:7b
```

## Tech stack

- **ML**: PyTorch (LSTM autoencoder), scikit-learn (threshold calibration)
- **RAG**: LlamaIndex, ChromaDB, sentence-transformers, BM25 + RRF fusion
- **LLM**: Ollama (local), Groq (cloud fallback), OpenAI (optional)
- **Serving**: FastAPI, Streamlit
- **MLOps**: MLflow, Evidently (drift monitoring), RAGAS (RAG evaluation), GitHub Actions CI

## Project layout

```
.
├── src/
│   ├── config.py             # Pydantic settings
│   ├── data/                 # C-MAPSS loaders, preprocessing
│   ├── models/               # LSTM autoencoder
│   ├── rag/                  # ingestion, retrieval, prompts
│   ├── llm/                  # LLM backends (Ollama, Groq, OpenAI, HF)
│   ├── pipeline.py           # end-to-end orchestration
│   ├── api/                  # FastAPI app
│   └── ui/                   # Streamlit dashboard
├── scripts/                  # train, evaluate CLIs
├── tests/                    # pytest
├── data/                     # raw/, processed/, manuals/ (gitignored)
├── pyproject.toml
├── Makefile
└── .github/workflows/ci.yml
```

## Pitfalls encountered

- **Scaler data leakage** - initially fit the scaler on the full dataset including test cycles. Metrics looked better but the model had seen future data. Fixed by fitting only on healthy training cycles (RUL > threshold).
- **Threshold sensitivity to anomaly horizon** - the F1-optimal threshold shifts significantly depending on how many cycles before failure you label as anomalous (h=30 vs h=50). Reported results use h=30; h=50 inflates recall at the cost of precision.
- **ChromaDB persistence with embedding model changes** - upgrading the embedding model invalidates the existing vector store silently. Queries return results but recall drops. Requires full re-ingest after any model change.
- **BM25 index rebuild on corpus change** - the BM25 index is not updated incrementally. Any change to the corpus (adding/removing documents) requires a full rebuild and deletion of the old index.
- **ChromaDB size** - 612k chunks from ~250MB of PDFs produced an 8GB SQLite file. The binary embedding index is only ~1GB; the rest is stored document text. This made cloud deployment impractical without reducing the corpus or switching to a leaner vector store.

## Hugging Face Space

The [live demo](https://huggingface.co/spaces/gabrieleformis/predictive-maintenance-copilot) runs the full anomaly detection pipeline but disables the RAG explanation layer. The ChromaDB vector store built from the maintenance manuals is ~8GB — too large for HF Spaces free tier. Anomaly detection, sensor trajectories, and score timelines work normally. For the full pipeline including LLM explanations with manual citations, run locally following the Quick start above.

## Roadmap

- [x] C-MAPSS data loader and preprocessing
- [x] LSTM autoencoder training with MLflow tracking
- [x] Anomaly evaluation: ROC-AUC 0.906, PR-AUC 0.789
- [x] PDF ingestion pipeline → ChromaDB (612k chunks)
- [x] Hybrid retriever (BM25 + dense + RRF)
- [x] Structured LLM prompt with output parser
- [x] FastAPI endpoint with end-to-end pipeline
- [x] Streamlit dashboard
- [x] Drift monitoring with Evidently
- [x] Deploy to Hugging Face Spaces
- [x] RAGAS evaluation on golden Q/A set

## License

MIT. Datasets and manuals retain their original licenses; please consult each source.

## About

Built by [Gabriele Formis](https://github.com/gabrieleformis95).
