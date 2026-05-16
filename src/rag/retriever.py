"""Hybrid retriever: BM25 + dense embeddings + Reciprocal Rank Fusion.

All heavy state (Chroma collection, sentence-transformer, BM25 index) is
loaded once and cached for the lifetime of the process. Call
`clear_retriever_cache()` after re-ingesting manuals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


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
    all_texts: list[str]
    all_meta: list[dict]


_BM25_CAP = 30_000  # max docs loaded into BM25 index
_FETCH_BATCH = 4_000  # ChromaDB rows per get() call


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
        import chromadb
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        from src.config import settings

        client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        collection = client.get_collection(settings.collection_name)

        logger.info("Fetching corpus from ChromaDB (paginated)...")
        all_ids, all_texts, all_meta = _fetch_all(collection)

        if not all_ids:
            logger.warning("Vector store collection is empty — run `make ingest-manuals`.")
            return None

        # Cap BM25 corpus to avoid excessive memory use on large collections.
        if len(all_ids) > _BM25_CAP:
            logger.info(
                "Collection has %d chunks; capping BM25 index at %d.",
                len(all_ids),
                _BM25_CAP,
            )
            import random

            indices = random.sample(range(len(all_ids)), _BM25_CAP)
            bm25_ids = [all_ids[i] for i in indices]
            bm25_texts = [all_texts[i] for i in indices]
            bm25_meta = [all_meta[i] for i in indices]
        else:
            bm25_ids, bm25_texts, bm25_meta = all_ids, all_texts, all_meta

        tokenized = [t.lower().split() for t in bm25_texts]
        bm25 = BM25Okapi(tokenized)
        embed_model = SentenceTransformer(settings.embedding_model)

        logger.info(
            "Retriever ready: %d total chunks, %d in BM25 index, collection '%s'.",
            len(all_ids),
            len(bm25_ids),
            settings.collection_name,
        )
        return _RetrieverState(
            collection=collection,
            embed_model=embed_model,
            bm25=bm25,
            all_ids=bm25_ids,
            all_texts=bm25_texts,
            all_meta=bm25_meta,
        )
    except Exception as e:
        logger.warning("Could not initialise retriever: %s", e)
        return None


def clear_retriever_cache() -> None:
    """Invalidate the retriever cache (call after re-ingesting manuals)."""
    _get_state.cache_clear()


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
    bm25_scores = state.bm25.get_scores(query.lower().split())
    bm25_ranked: list[str] = [
        state.all_ids[i]
        for i in sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[
            :n_candidates
        ]
    ]

    # RRF fusion.
    fused = _rrf([dense_ranked, bm25_ranked])
    top_ids = sorted(fused, key=lambda d: fused[d], reverse=True)[:k]

    id_to_idx = {doc_id: i for i, doc_id in enumerate(state.all_ids)}
    chunks: list[RetrievedChunk] = []
    for doc_id in top_ids:
        if doc_id not in id_to_idx:
            continue
        idx = id_to_idx[doc_id]
        meta = state.all_meta[idx]
        chunks.append(
            RetrievedChunk(
                text=state.all_texts[idx],
                source=meta.get("file_name", meta.get("source", "unknown")),
                page=int(meta["page_label"]) if "page_label" in meta else None,
                score=fused[doc_id],
                metadata={str(k_): str(v) for k_, v in meta.items()},
            )
        )

    return chunks
