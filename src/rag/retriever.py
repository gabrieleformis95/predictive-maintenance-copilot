"""Hybrid retriever: BM25 + dense embeddings + Reciprocal Rank Fusion.

All heavy state (Chroma collection, sentence-transformer, BM25 index) is
loaded once and cached for the lifetime of the process. Call
`clear_retriever_cache()` after re-ingesting manuals.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

_FETCH_BATCH = 4_000  # ChromaDB rows per get() call


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int | None
    score: float
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class _RetrieverState:
    collection: Any
    embed_model: Any
    bm25: Any
    all_ids: list[str]


def _fetch_all(collection) -> tuple[list[str], list[str], list[dict]]:
    """Paginate through a ChromaDB collection to avoid SQLite variable limits."""
    all_ids: list[str] = []
    all_texts: list[str] = []
    all_meta: list[dict] = []
    offset = 0
    while True:
        batch = collection.get(
            limit=_FETCH_BATCH,
            offset=offset,
            include=["documents", "metadatas"],
        )
        if not batch["ids"]:
            break
        all_ids.extend(batch["ids"])
        all_texts.extend(batch["documents"])
        all_meta.extend([m or {} for m in batch["metadatas"]])
        offset += len(batch["ids"])
        if len(batch["ids"]) < _FETCH_BATCH:
            break
    return all_ids, all_texts, all_meta


@lru_cache(maxsize=1)
def _get_state() -> _RetrieverState | None:
    """Load and cache everything needed for retrieval. Returns None if unavailable."""
    try:
        import bm25s
        import chromadb
        from sentence_transformers import SentenceTransformer

        from src.config import settings

        client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        collection = client.get_collection(settings.collection_name)

        bm25_dir = settings.bm25_index_dir
        ids_file = bm25_dir / "ids.json"

        if bm25_dir.exists() and ids_file.exists():
            logger.info("Loading BM25 index from disk (%s)...", bm25_dir)
            bm25_index = bm25s.BM25.load(str(bm25_dir), mmap=True)
            with open(ids_file) as f:
                all_ids: list[str] = json.load(f)
        else:
            logger.info("Fetching corpus from ChromaDB (paginated)...")
            all_ids, all_texts, _ = _fetch_all(collection)
            if not all_ids:
                logger.warning("Vector store collection is empty — run `make ingest-manuals`.")
                return None
            logger.info("Building BM25 index over %d chunks...", len(all_ids))
            corpus_tokens = bm25s.tokenize(all_texts, stopwords="en", show_progress=False)
            bm25_index = bm25s.BM25()
            bm25_index.index(corpus_tokens, show_progress=False)
            bm25_dir.mkdir(parents=True, exist_ok=True)
            bm25_index.save(str(bm25_dir))
            with open(ids_file, "w") as f:
                json.dump(all_ids, f)
            logger.info("BM25 index saved to %s.", bm25_dir)

        if not all_ids:
            logger.warning("Vector store collection is empty — run `make ingest-manuals`.")
            return None

        embed_model = SentenceTransformer(settings.embedding_model)

        logger.info(
            "Retriever ready: %d chunks in BM25 index, collection '%s'.",
            len(all_ids),
            settings.collection_name,
        )
        return _RetrieverState(
            collection=collection,
            embed_model=embed_model,
            bm25=bm25_index,
            all_ids=all_ids,
        )
    except Exception as e:
        logger.warning("Could not initialise retriever: %s", e)
        return None


def clear_retriever_cache() -> None:
    """Invalidate the retriever cache and delete the on-disk BM25 index."""
    import shutil

    from src.config import settings

    _get_state.cache_clear()
    bm25_dir = settings.bm25_index_dir
    if bm25_dir.exists():
        shutil.rmtree(bm25_dir)
        logger.info("BM25 index cleared from %s.", bm25_dir)


def _rrf(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion over multiple ranked ID lists."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    """Return top-k chunks for `query` using BM25 + dense RRF fusion.

    Returns an empty list (never raises) when the vector store is unavailable
    or empty — the pipeline handles the no-context case gracefully.
    """
    state = _get_state()
    if state is None:
        return []

    n_candidates = min(k * 4, len(state.all_ids))

    # Dense retrieval.
    query_vec = state.embed_model.encode(query).tolist()
    dense_res = state.collection.query(
        query_embeddings=[query_vec],
        n_results=n_candidates,
        include=[],
    )
    dense_ranked: list[str] = dense_res["ids"][0]

    # BM25 retrieval.
    import bm25s as _bm25s

    query_tokens = _bm25s.tokenize([query], stopwords="en", show_progress=False)
    bm25_results, _ = state.bm25.retrieve(query_tokens, k=n_candidates)
    bm25_ranked: list[str] = [state.all_ids[i] for i in bm25_results[0].tolist()]

    # RRF fusion.
    fused = _rrf([dense_ranked, bm25_ranked])
    top_ids = sorted(fused, key=lambda d: fused[d], reverse=True)[:k]

    if not top_ids:
        return []

    # Fetch text and metadata for top-k from ChromaDB.
    fetched = state.collection.get(ids=top_ids, include=["documents", "metadatas"])
    id_to_doc: dict[str, tuple[str, dict]] = {
        doc_id: (doc, meta or {})
        for doc_id, doc, meta in zip(
            fetched["ids"], fetched["documents"], fetched["metadatas"], strict=False
        )
    }

    chunks: list[RetrievedChunk] = []
    for doc_id in top_ids:
        if doc_id not in id_to_doc:
            continue
        text, meta = id_to_doc[doc_id]
        chunks.append(
            RetrievedChunk(
                text=text,
                source=meta.get("file_name", meta.get("source", "unknown")),
                page=int(meta["page_label"]) if "page_label" in meta else None,
                score=fused[doc_id],
                metadata={str(k_): str(v) for k_, v in meta.items()},
            )
        )

    return chunks
