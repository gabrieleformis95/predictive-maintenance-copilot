"""Centralized configuration via pydantic-settings.

Values come from environment variables and `.env`. Defaults are reasonable
for local development.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings. All env vars override these defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: str = Field(default="ollama", description="ollama | groq | huggingface | openai")
    llm_model: str = Field(
        default="qwen2.5:7b",
        description=(
            "Model name. For Ollama use tags like 'qwen2.5:7b'. "
            "For Groq use IDs like 'llama-3.3-70b-versatile'. "
            "For HF use repo IDs like 'mistralai/Mistral-7B-Instruct-v0.3'."
        ),
    )
    hf_token: str | None = None
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"

    # --- Embeddings & vector store ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_persist_dir: Path = PROJECT_ROOT / "chroma_db"
    collection_name: str = "industrial_manuals"
    bm25_index_dir: Path = PROJECT_ROOT / "data" / "bm25_index"

    # --- Paths ---
    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    manuals_dir: Path = PROJECT_ROOT / "data" / "manuals"

    # --- MLflow ---
    mlflow_tracking_uri: str = f"file:{PROJECT_ROOT / 'mlruns'}"
    mlflow_experiment_name: str = "predictive-maintenance-copilot"

    # --- Anomaly detector ---
    window_size: int = 30
    anomaly_threshold_percentile: float = 95.0
    checkpoint_path: Path = PROJECT_ROOT / "checkpoints" / "autoencoder_FD001.pt"


settings = Settings()
