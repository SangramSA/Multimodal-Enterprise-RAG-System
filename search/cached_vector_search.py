"""Vector search with semantic cache + lightweight local reranking."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from loguru import logger

from middleware.reranker import CrossEncoderReranker, RerankerConfig
from middleware.semantic_cache import SemanticCache, SemanticCacheConfig
from vector.vector_store import VectorStore


class CachedVectorSearch:
    """
    Drop-in replacement for `VectorSearch` with:
    - semantic cache (local embeddings; avoids vector DB call on hit)
    - rerank top 5 among 50 candidates (local cross-encoder)
    - per-stage millisecond breakdown
    """

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        cache: Optional[SemanticCache] = None,
        cache_config: Optional[SemanticCacheConfig] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        reranker_config: Optional[RerankerConfig] = None,
        candidate_limit: int = 50,
        rerank_k: int = 5,
    ):
        self.vector_store = vector_store
        self.cache = cache or SemanticCache(cache_config)
        self.reranker = reranker or CrossEncoderReranker(reranker_config)
        self.candidate_limit = candidate_limit
        self.rerank_k = rerank_k

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
        *,
        cache_mode: str = "enabled",
    ) -> List[Dict[str, Any]]:
        """
        Args:
            cache_mode:
              - "enabled": use cache; on miss hit vector DB and store
              - "bypass": force vector DB call; still store (so the next call can hit cache)
              - "only": only consult cache; never hit vector DB
        """
        overall_t0 = time.perf_counter()
        timings_ms: Dict[str, float] = {}

        logger.info(
            "CachedVectorSearch.search | mode={} | limit={} | candidate_limit={} | score_threshold={}",
            cache_mode,
            limit,
            self.candidate_limit,
            score_threshold,
        )

        cache_mode = (cache_mode or "enabled").lower()
        if cache_mode not in {"enabled", "bypass", "only"}:
            cache_mode = "enabled"

        # Pre-embed once for lookup/store (local, fast compared to OpenAI embeddings)
        t0 = time.perf_counter()
        query_embedding_unit = self.cache.embed_query_unit(query)
        timings_ms["cache_embed_ms"] = (time.perf_counter() - t0) * 1000.0

        # 1) Semantic cache lookup
        if cache_mode != "bypass":
            t1 = time.perf_counter()
            hit, sim, matched_query, age_ms, cached_results = self.cache.lookup(
                query, query_embedding_unit=query_embedding_unit
            )
            timings_ms["cache_lookup_ms"] = (time.perf_counter() - t1) * 1000.0

            if hit and cached_results is not None:
                logger.info(
                    "CachedVectorSearch semantic cache HIT | query_prefix='{}' | sim={:.3f} | age_ms={:.1f}",
                    query[:80],
                    sim,
                    age_ms or 0.0,
                )
                results = cached_results[:limit]
                self._attach_middleware_metadata(
                    results,
                    cache_hit=True,
                    cache_similarity=sim,
                    cache_matched_query=matched_query,
                    cache_age_ms=age_ms,
                    timings_ms=timings_ms,
                    cache_mode=cache_mode,
                )
                timings_ms["total_ms"] = (time.perf_counter() - overall_t0) * 1000.0
                return results

            if cache_mode == "only":
                logger.info(
                    "CachedVectorSearch semantic cache ONLY mode MISS | query_prefix='{}'",
                    query[:80],
                )
                timings_ms["total_ms"] = (time.perf_counter() - overall_t0) * 1000.0
                return []

        # 2) Cold path: vector DB + rerank
        try:
            candidates, vs_timings = self.vector_store.search_with_timings(
                query=query,
                limit=self.candidate_limit,
                filters=filters,
                score_threshold=score_threshold,
            )
            timings_ms.update(vs_timings)
        except Exception as e:
            logger.warning("CachedVectorSearch vector_store search_with_timings failed: {}", e)
            candidates = self.vector_store.search(
                query=query, limit=self.candidate_limit, filters=filters, score_threshold=score_threshold
            )

        # 3) Rerank a small prefix locally
        try:
            reranked, rerank_timings = self.reranker.rerank(
                query=query,
                docs=candidates,
                rerank_k=self.rerank_k,
                score_key="score",
            )
            timings_ms.update(rerank_timings)
        except Exception as e:
            logger.warning("CachedVectorSearch reranker failed, returning original ordering: {}", e)
            reranked = candidates
            timings_ms["rerank_ms"] = 0.0

        results = reranked[:limit]

        logger.info(
            "CachedVectorSearch COLD path | query_prefix='{}' | candidates={} | returned={} | timings_ms={}",
            query[:80],
            len(candidates),
            len(results),
            {k: round(v, 2) for k, v in timings_ms.items()},
        )

        # 4) Store into semantic cache for next time (enabled + bypass)
        if cache_mode in {"enabled", "bypass"}:
            t_store = time.perf_counter()
            self.cache.store(
                query=query,
                results=reranked,
                query_embedding_unit=query_embedding_unit,
                metadata={"filters": filters, "score_threshold": score_threshold},
            )
            timings_ms["cache_store_ms"] = (time.perf_counter() - t_store) * 1000.0

        timings_ms["total_ms"] = (time.perf_counter() - overall_t0) * 1000.0
        self._attach_middleware_metadata(
            results,
            cache_hit=False,
            cache_similarity=None,
            cache_matched_query=None,
            cache_age_ms=None,
            timings_ms=timings_ms,
            cache_mode=cache_mode,
        )
        return results

    def _attach_middleware_metadata(
        self,
        results: List[Dict[str, Any]],
        *,
        cache_hit: bool,
        cache_similarity: Optional[float],
        cache_matched_query: Optional[str],
        cache_age_ms: Optional[float],
        timings_ms: Dict[str, float],
        cache_mode: str,
    ) -> None:
        meta = {
            "cache_hit": cache_hit,
            "cache_similarity": cache_similarity,
            "cache_matched_query": cache_matched_query,
            "cache_age_ms": cache_age_ms,
            "cache_mode": cache_mode,
            "latency_breakdown_ms": dict(timings_ms),
        }
        for r in results:
            r["middleware"] = meta

