from typing import Any, Dict, List, Optional, Tuple

import pytest

from search.cached_vector_search import CachedVectorSearch


class DummyCache:
    def __init__(self, *, hit: bool, cached_results: Optional[List[Dict[str, Any]]] = None):
        self.hit = hit
        self.cached_results = cached_results or [{"chunk_id": "cached1", "content": "c", "score": 0.9, "metadata": {}}]
        self.store_calls: int = 0
        self.lookup_calls: int = 0

    def embed_query_unit(self, _query: str):
        return [1.0, 0.0]

    def lookup(self, _query: str, *, query_embedding_unit=None):
        self.lookup_calls += 1
        if self.hit:
            return True, 0.99, "prev_query", 10.0, list(self.cached_results)
        return False, 0.0, None, None, None

    def store(self, *args, **kwargs):
        self.store_calls += 1


class DummyReranker:
    def __init__(self):
        self.calls = 0

    def rerank(self, query: str, docs: List[Dict[str, Any]], *, rerank_k=None, score_key="score"):
        self.calls += 1
        # Reverse ordering to prove it ran
        return list(reversed(docs)), {"rerank_ms": 1.23}


class DummyVectorStore:
    def __init__(self, candidates: Optional[List[Dict[str, Any]]] = None):
        self.candidates = candidates or [
            {"chunk_id": "c1", "content": "a", "score": 0.8, "metadata": {}},
            {"chunk_id": "c2", "content": "b", "score": 0.7, "metadata": {}},
        ]
        self.search_with_timings_calls = 0

    def search_with_timings(self, *, query: str, limit: int, filters=None, score_threshold: float = 0.0):
        self.search_with_timings_calls += 1
        return list(self.candidates)[:limit], {"openai_embed_ms": 10.0, "qdrant_search_ms": 5.0}

    def search(self, *, query: str, limit: int, filters=None, score_threshold: float = 0.0):
        return list(self.candidates)[:limit]


def test_cache_hit_bypasses_vector_store_and_reranker():
    vs = DummyVectorStore()
    cache = DummyCache(hit=True)
    reranker = DummyReranker()

    cvs = CachedVectorSearch(vs, cache=cache, reranker=reranker)  # type: ignore[arg-type]
    results = cvs.search("q", limit=1, cache_mode="enabled")

    assert vs.search_with_timings_calls == 0
    assert reranker.calls == 0
    assert cache.lookup_calls == 1
    assert results[0]["chunk_id"] == "cached1"
    assert results[0]["middleware"]["cache_hit"] is True


def test_cache_miss_calls_vector_store_reranks_and_stores():
    vs = DummyVectorStore(
        candidates=[
            {"chunk_id": "c1", "content": "a", "score": 0.8, "metadata": {}},
            {"chunk_id": "c2", "content": "b", "score": 0.7, "metadata": {}},
        ]
    )
    cache = DummyCache(hit=False)
    reranker = DummyReranker()

    cvs = CachedVectorSearch(vs, cache=cache, reranker=reranker, candidate_limit=2, rerank_k=1)  # type: ignore[arg-type]
    results = cvs.search("q", limit=2, cache_mode="enabled")

    assert vs.search_with_timings_calls == 1
    assert reranker.calls == 1
    assert cache.store_calls == 1
    assert results[0]["chunk_id"] == "c2"  # reversed by reranker dummy
    assert results[0]["middleware"]["cache_hit"] is False
    assert "latency_breakdown_ms" in results[0]["middleware"]


def test_cache_mode_only_never_hits_vector_store_on_miss():
    vs = DummyVectorStore()
    cache = DummyCache(hit=False)
    reranker = DummyReranker()

    cvs = CachedVectorSearch(vs, cache=cache, reranker=reranker)  # type: ignore[arg-type]
    results = cvs.search("q", limit=2, cache_mode="only")

    assert results == []
    assert vs.search_with_timings_calls == 0
    assert reranker.calls == 0


def test_cache_mode_bypass_forces_cold_path_and_stores():
    vs = DummyVectorStore()
    cache = DummyCache(hit=True)  # even if it could hit, bypass should force cold
    reranker = DummyReranker()

    cvs = CachedVectorSearch(vs, cache=cache, reranker=reranker, candidate_limit=2)  # type: ignore[arg-type]
    results = cvs.search("q", limit=1, cache_mode="bypass")

    assert vs.search_with_timings_calls == 1
    assert cache.lookup_calls == 0
    assert cache.store_calls == 1
    assert results[0]["middleware"]["cache_hit"] is False

