"""Semantic cache for retrieval results (LRU + TTL) using local embeddings.

This is a POC cache designed to sit between a query and a vector DB:
- On semantic cache hit: return cached results without calling the vector DB.
- On miss: caller fetches from vector DB and stores results.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    inv = 1.0 / norm
    return [x * inv for x in vec]


def _cosine_sim_unit(a_unit: List[float], b_unit: List[float]) -> float:
    # Assumes both are already normalized.
    if len(a_unit) != len(b_unit):
        return 0.0
    return float(sum(x * y for x, y in zip(a_unit, b_unit)))


class LocalBiEncoder:
    """Small local model for query embeddings (lazy import)."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "sentence-transformers is required for semantic cache embeddings. "
                    "Install it (and its torch dependency) to enable this feature."
                ) from e
            self._model = SentenceTransformer(self.model_name)
            return self._model

    def embed_query(self, text: str) -> List[float]:
        model = self._get_model()
        emb = model.encode(text, normalize_embeddings=True)
        # sentence-transformers returns numpy array by default; convert to plain list.
        return [float(x) for x in emb.tolist()]


@dataclass(frozen=True)
class SemanticCacheConfig:
    ttl_seconds: int = 300
    max_entries: int = 1000
    similarity_threshold: float = 0.90
    bi_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class SemanticCacheEntry:
    query: str
    embedding_unit: List[float]
    results: List[Dict[str, Any]]
    created_at_s: float
    metadata: Dict[str, Any]

    def age_ms(self, now_s: Optional[float] = None) -> float:
        now = now_s if now_s is not None else time.time()
        return (now - self.created_at_s) * 1000.0


class SemanticCache:
    """In-memory semantic cache keyed by embedding similarity (not exact string)."""

    def __init__(self, config: Optional[SemanticCacheConfig] = None):
        self.config = config or SemanticCacheConfig()
        self._embedder = LocalBiEncoder(self.config.bi_encoder_model)
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, SemanticCacheEntry]" = OrderedDict()

        # Stats
        self._lookups = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expired = 0

    def _purge_expired_locked(self, now_s: float):
        if not self._entries:
            return
        ttl = self.config.ttl_seconds
        if ttl <= 0:
            return

        expired_keys = []
        for key, entry in self._entries.items():
            if (now_s - entry.created_at_s) > ttl:
                expired_keys.append(key)

        for key in expired_keys:
            self._entries.pop(key, None)
            self._expired += 1

    def _evict_lru_locked(self):
        while len(self._entries) > self.config.max_entries:
            self._entries.popitem(last=False)
            self._evictions += 1

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            hit_rate = (self._hits / self._lookups) if self._lookups else 0.0
            return {
                "entries": len(self._entries),
                "lookups": self._lookups,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "expired": self._expired,
                "ttl_seconds": self.config.ttl_seconds,
                "max_entries": self.config.max_entries,
                "similarity_threshold": self.config.similarity_threshold,
                "bi_encoder_model": self.config.bi_encoder_model,
            }

    def lookup(
        self, query: str, *, query_embedding_unit: Optional[List[float]] = None
    ) -> Tuple[bool, float, Optional[str], Optional[float], Optional[List[Dict[str, Any]]]]:
        """
        Lookup semantically similar cached results.

        Returns:
            (hit, similarity, matched_query, age_ms, cached_results)
        """
        now_s = time.time()
        embedding_unit = query_embedding_unit or self.embed_query_unit(query)
        
        with self._lock:
            self._lookups += 1
            self._purge_expired_locked(now_s)
            
            best_key = None
            best_sim = -1.0
            
            # Brute-force scan for POC; for large cache sizes use ANN.
            for key, entry in self._entries.items():
                sim = _cosine_sim_unit(embedding_unit, entry.embedding_unit)
                if sim > best_sim:
                    best_sim = sim
                    best_key = key
            
            if best_key is None or best_sim < self.config.similarity_threshold:
                self._misses += 1
                logger.debug(
                    "SemanticCache MISS | query_prefix='{}' | best_sim={:.3f} | threshold={:.3f} | size={}",
                    query[:80],
                    best_sim if best_sim >= 0 else 0.0,
                    self.config.similarity_threshold,
                    len(self._entries),
                )
                return False, float(best_sim if best_sim >= 0 else 0.0), None, None, None
            
            # LRU update
            entry = self._entries.pop(best_key)
            self._entries[best_key] = entry
            self._hits += 1
            age_ms = entry.age_ms(now_s)
            logger.debug(
                "SemanticCache HIT  | query_prefix='{}' | matched_query='{}' | sim={:.3f} | age_ms={:.1f} | size={}",
                query[:80],
                entry.query[:80],
                best_sim,
                age_ms,
                len(self._entries),
            )
            
            return True, float(best_sim), entry.query, age_ms, entry.results

    def store(
        self,
        query: str,
        results: List[Dict[str, Any]],
        *,
        query_embedding_unit: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        now_s = time.time()
        embedding_unit = query_embedding_unit or self.embed_query_unit(query)

        entry = SemanticCacheEntry(
            query=query,
            embedding_unit=embedding_unit,
            results=results,
            created_at_s=now_s,
            metadata=metadata or {},
        )

        with self._lock:
            # Use the raw query as the key (LRU order), but similarity matching is embedding-based.
            # If identical query string exists, overwrite and refresh position.
            if query in self._entries:
                self._entries.pop(query, None)
            self._entries[query] = entry
            self._purge_expired_locked(now_s)
            before_evict = len(self._entries)
            self._evict_lru_locked()
            after_evict = len(self._entries)

        logger.debug(
            "SemanticCache STORE | query_prefix='{}' | results={} | ttl_s={} | max_entries={} | size_before={} | size_after={}",
            query[:80],
            len(results),
            self.config.ttl_seconds,
            self.config.max_entries,
            before_evict,
            after_evict,
        )

    def embed_query_unit(self, query: str) -> List[float]:
        """Embed and normalize a query using the local bi-encoder."""
        return _normalize(self._embedder.embed_query(query))

