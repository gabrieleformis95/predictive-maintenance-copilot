"""Ingest PDFs from data/manuals/ into a ChromaDB collection.

Uses LlamaIndex for chunking + embedding, and persists locally so the
RAG side of the project doesn't depend on any cloud service.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.logging import RichHandler

from src.config import settings

app = typer.Typer(help="Ingest maintenance manuals into the vector store.")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("rag.ingestion")


def build_index(
    manuals_dir: Path,
    persist_dir: Path,
    collection_name: str,
    embedding_model: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> None:
    """Read PDFs from `manuals_dir` and build a persistent Chroma collection.

    Lazy-imports LlamaIndex / Chroma so this module imports quickly when the
    user just wants to run other parts of the codebase.
    """
    import chromadb
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.settings import Settings as LISettings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    manuals_dir = Path(manuals_dir)
    persist_dir = Path(persist_dir)

    if not manuals_dir.exists() or not any(manuals_dir.iterdir()):
        raise FileNotFoundError(
            f"No documents found in {manuals_dir}. "
            "Drop a few maintenance PDFs there before running ingestion."
        )

    persist_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading documents from %s", manuals_dir)
    documents = SimpleDirectoryReader(input_dir=str(manuals_dir), recursive=True).load_data()
    logger.info("Loaded %d documents", len(documents))

    logger.info("Initializing embeddings model: %s", embedding_model)
    LISettings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
    LISettings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    logger.info("Initializing Chroma at %s (collection=%s)", persist_dir, collection_name)
    client = chromadb.PersistentClient(path=str(persist_dir))
    chroma_collection = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logger.info("Building index (this may take a few minutes)…")
    VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    logger.info("Indexing complete. Collection size: %d", chroma_collection.count())


@app.command()
def ingest(
    manuals_dir: Annotated[Path | None, typer.Option("--manuals-dir")] = None,
    persist_dir: Annotated[Path | None, typer.Option("--persist-dir")] = None,
    collection_name: Annotated[str | None, typer.Option("--collection")] = None,
    embedding_model: Annotated[str | None, typer.Option("--embed-model")] = None,
) -> None:
    """Ingest PDFs and build the vector store."""
    build_index(
        manuals_dir or settings.manuals_dir,
        persist_dir or settings.chroma_persist_dir,
        collection_name or settings.collection_name,
        embedding_model or settings.embedding_model,
    )


def ingest_cli() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    app()
