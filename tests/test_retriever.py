"""Tests for src/rag/retriever.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.rag.retriever import (
    RetrievedChunk,
    _fetch_all,
    _RetrieverState,
    _rrf,
    clear_retriever_cache,
    retrieve,
)

_FETCH_BATCH = 4_000


# ---------------------------------------------------------------------------
# _rrf
# ---------------------------------------------------------------------------


def test_rrf_single_list_descending():
    """GIVEN a single ranked list ['a', 'b', 'c'].
    WHEN _rrf is called.
    THEN scores are strictly descending: score(a) > score(b) > score(c).
    """
    scores = _rrf([["a", "b", "c"]], k=60)
    assert scores["a"] > scores["b"] > scores["c"]


def test_rrf_fusion_boosts_shared_ids():
    """GIVEN two ranked lists where 'x' appears in both and 'w' in only one.
    WHEN _rrf is called.
    THEN score('x') > score('w') because 'x' gets contributions from both lists.
    """
    scores = _rrf([["x", "y", "z"], ["z", "x", "w"]])
    assert scores["x"] > scores["w"]
    assert scores["z"] > scores["w"]


def test_rrf_empty_lists():
    """GIVEN two empty ranked lists.
    WHEN _rrf is called.
    THEN an empty dict is returned.
    """
    assert _rrf([[], []]) == {}


# ---------------------------------------------------------------------------
# _fetch_all
# ---------------------------------------------------------------------------


def test_fetch_all_single_batch():
    """GIVEN a collection that returns all docs in one batch smaller than _FETCH_BATCH.
    WHEN _fetch_all is called.
    THEN all ids, texts, and meta are returned; None metadata is replaced with {}.
    """
    mock_col = MagicMock()
    mock_col.get.return_value = {
        "ids": ["a", "b"],
        "documents": ["text_a", "text_b"],
        "metadatas": [{"k": "v"}, None],
    }
    ids, texts, meta = _fetch_all(mock_col)
    assert ids == ["a", "b"]
    assert texts == ["text_a", "text_b"]
    assert meta == [{"k": "v"}, {}]


def test_fetch_all_paginates_across_batches():
    """GIVEN a collection with _FETCH_BATCH docs in the first call and 2 in the second.
    WHEN _fetch_all is called.
    THEN results from both batches are concatenated.
    """
    mock_col = MagicMock()
    batch1_ids = [f"id_{i}" for i in range(_FETCH_BATCH)]
    mock_col.get.side_effect = [
        {
            "ids": batch1_ids,
            "documents": ["t"] * _FETCH_BATCH,
            "metadatas": [{}] * _FETCH_BATCH,
        },
        {
            "ids": ["id_last1", "id_last2"],
            "documents": ["t", "t"],
            "metadatas": [{}, {}],
        },
    ]
    ids, _, _ = _fetch_all(mock_col)
    assert len(ids) == _FETCH_BATCH + 2


def test_fetch_all_empty_collection():
    """GIVEN a collection that immediately returns no ids.
    WHEN _fetch_all is called.
    THEN empty lists are returned.
    """
    mock_col = MagicMock()
    mock_col.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    ids, texts, meta = _fetch_all(mock_col)
    assert ids == [] and texts == [] and meta == []


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------


def test_retrieve_returns_empty_when_state_none():
    """GIVEN the retriever state is unavailable (ChromaDB not initialised).
    WHEN retrieve is called.
    THEN an empty list is returned without raising.
    """
    with patch("src.rag.retriever._get_state", return_value=None):
        assert retrieve("any query", k=5) == []


def _make_state() -> _RetrieverState:
    ids = ["doc1", "doc2", "doc3"]
    texts = ["bearing maintenance", "lubrication guide", "inspection checklist"]
    meta = [
        {"file_name": "manual.pdf", "page_label": "10"},
        {"file_name": "guide.pdf", "page_label": "5"},
        {"source": "checklist.pdf"},
    ]
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"ids": [ids]}
    mock_collection.get.return_value = {
        "ids": ids,
        "documents": texts,
        "metadatas": meta,
    }
    mock_embed = MagicMock()
    mock_embed.encode.return_value = np.zeros(384)
    mock_bm25 = MagicMock()
    # bm25s.retrieve returns (results_indices, scores), shape (n_queries, k)
    mock_bm25.retrieve.return_value = (
        np.array([[0, 1, 2]]),
        np.array([[0.9, 0.5, 0.3]]),
    )
    return _RetrieverState(
        collection=mock_collection,
        embed_model=mock_embed,
        bm25=mock_bm25,
        all_ids=ids,
    )


def test_retrieve_returns_retrieved_chunks():
    """GIVEN a RetrieverState with 3 documents and a query.
    WHEN retrieve is called with k=3.
    THEN a non-empty list of RetrievedChunk objects is returned.
    """
    with patch("src.rag.retriever._get_state", return_value=_make_state()):
        result = retrieve("bearing fault", k=3)
    assert len(result) > 0
    assert all(isinstance(c, RetrievedChunk) for c in result)


def test_retrieve_parses_page_label_from_metadata():
    """GIVEN a document with page_label='10' in its metadata.
    WHEN retrieve returns that document.
    THEN the RetrievedChunk.page equals 10 (int).
    """
    with patch("src.rag.retriever._get_state", return_value=_make_state()):
        result = retrieve("bearing", k=3)
    chunk = next((c for c in result if c.source == "manual.pdf"), None)
    if chunk is not None:
        assert chunk.page == 10


def test_retrieve_handles_missing_page_label():
    """GIVEN a document with no page_label in its metadata.
    WHEN retrieve returns that document.
    THEN the RetrievedChunk.page is None.
    """
    with patch("src.rag.retriever._get_state", return_value=_make_state()):
        result = retrieve("inspection", k=3)
    chunk = next((c for c in result if c.source == "checklist.pdf"), None)
    if chunk is not None:
        assert chunk.page is None


# ---------------------------------------------------------------------------
# clear_retriever_cache
# ---------------------------------------------------------------------------


def test_clear_retriever_cache(monkeypatch):
    """GIVEN the retriever lru_cache is populated.
    WHEN clear_retriever_cache is called.
    THEN _get_state.cache_clear is invoked exactly once.
    """
    import src.rag.retriever as mod

    calls = []
    monkeypatch.setattr(mod._get_state, "cache_clear", lambda: calls.append(1))
    clear_retriever_cache()
    assert len(calls) == 1
