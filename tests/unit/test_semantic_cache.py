import pytest


from middleware.semantic_cache import SemanticCache, SemanticCacheConfig


def test_store_and_semantic_lookup_hit(monkeypatch):
    cache = SemanticCache(
        SemanticCacheConfig(ttl_seconds=300, max_entries=100, similarity_threshold=0.8)
    )

    # Deterministic embeddings for test
    def fake_embed(text: str):
        if text == "hello":
            return [1.0, 0.0]
        if text == "hi":
            return [1.0, 0.0]
        return [0.0, 1.0]

    monkeypatch.setattr(cache._embedder, "embed_query", fake_embed)

    cache.store("hello", results=[{"chunk_id": "c1"}])
    hit, sim, matched_query, age_ms, results = cache.lookup("hi")

    assert hit is True
    assert sim >= 0.99
    assert matched_query == "hello"
    assert age_ms is not None
    assert results == [{"chunk_id": "c1"}]


def test_ttl_expiry(monkeypatch):
    cache = SemanticCache(
        SemanticCacheConfig(ttl_seconds=1, max_entries=100, similarity_threshold=0.9)
    )
    monkeypatch.setattr(cache._embedder, "embed_query", lambda _t: [1.0, 0.0])

    # Freeze time for store, then advance past TTL for lookup.
    times = iter([1000.0, 1002.0])
    monkeypatch.setattr("middleware.semantic_cache.time.time", lambda: next(times))

    cache.store("q1", results=[{"chunk_id": "c1"}])
    hit, *_ = cache.lookup("q1")

    assert hit is False
    stats = cache.get_stats()
    assert stats["expired"] >= 1


def test_lru_eviction(monkeypatch):
    cache = SemanticCache(
        SemanticCacheConfig(ttl_seconds=300, max_entries=2, similarity_threshold=0.99)
    )

    def fake_embed(text: str):
        mapping = {"q1": [1.0, 0.0], "q2": [0.0, 1.0], "q3": [-1.0, 0.0]}
        return mapping[text]

    monkeypatch.setattr(cache._embedder, "embed_query", fake_embed)

    cache.store("q1", results=[{"chunk_id": "c1"}])
    cache.store("q2", results=[{"chunk_id": "c2"}])
    cache.store("q3", results=[{"chunk_id": "c3"}])  # should evict q1 (LRU)

    assert cache.size() == 2

    hit1, *_ = cache.lookup("q1")
    hit2, *_ = cache.lookup("q2")
    hit3, *_ = cache.lookup("q3")

    assert hit1 is False
    assert hit2 is True
    assert hit3 is True

