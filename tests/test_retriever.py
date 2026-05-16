"""Tests for src/rag/retriever.py."""

from __future__ import annotations

from unittest.mock import patch

from src.rag.retriever import _rrf, clear_retriever_cache, retrieve


def test_rrf_single_list_descending():
    scores = _rrf([["a", "b", "c"]], k=60)
    assert scores["a"] > scores["b"] > scores["c"]


def test_rrf_fusion_boosts_shared_ids():
    # "x" appears rank-0 in list1 and rank-1 in list2 → high combined score.
    # "w" appears only in list2 rank-2 → lower score.
    scores = _rrf([["x", "y", "z"], ["z", "x", "w"]])
    assert scores["x"] > scores["w"]
    assert scores["z"] > scores["w"]


def test_rrf_empty_lists():
    assert _rrf([[], []]) == {}


def test_retrieve_returns_empty_when_state_none():
    with patch("src.rag.retriever._get_state", return_value=None):
        result = retrieve("any query", k=5)
    assert result == []


def test_clear_retriever_cache(monkeypatch):
    import src.rag.retriever as mod

    calls = []
    monkeypatch.setattr(mod._get_state, "cache_clear", lambda: calls.append(1))
    clear_retriever_cache()
    assert len(calls) == 1
