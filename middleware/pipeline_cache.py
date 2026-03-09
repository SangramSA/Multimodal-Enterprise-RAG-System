"""Pipeline-level semantic cache for full query responses.

This cache operates above retrieval:
- Keys: semantic embedding of the sanitized user query
- Values: full pipeline response dicts (answer + sources + metadata)
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from vector.embedding_service import EmbeddingService


def _cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    inv = 1.0 / norm
    return [x * inv for x in vec]


@dataclass(frozen=True)
class PipelineCacheConfig:
    ttl_seconds: int = 300
    max_entries: int = 512
    similarity_threshold: float = 0.92


@dataclass
class PipelineCacheEntry:
    query: str
    embedding_unit: List[float]
    response: Dict[str, Any]
    created_at_s: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def age_ms(self, now_s: Optional[float] = None) -> float:
        now = now_s if now_s is not None else time.time()
        return (now - self.created_at_s) * 1000.0


class PipelineCache:
    """In-memory pipeline response cache (semantic, LRU + TTL)."""

    def __init__(self, config: Optional[PipelineCacheConfig] = None):
        self.config = config or PipelineCacheConfig()
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, PipelineCacheEntry]" = OrderedDict()
        self._embedder = EmbeddingService()

        # Stats
        self._lookups = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expired = 0

    def embed_query_unit(self, query: str) -> List[float]:
        """Embed and normalize query text using the configured embedding service."""
        return _normalize(self._embedder.embed_text(query))

    def _purge_expired_locked(self, now_s: float) -> None:
        if not self._entries or self.config.ttl_seconds <= 0:
            return
        expired_keys = []
        for key, entry in self._entries.items():
            if (now_s - entry.created_at_s) > self.config.ttl_seconds:
                expired_keys.append(key)
        for key in expired_keys:
            self._entries.pop(key, None)
            self._expired += 1

    def _evict_lru_locked(self) -> None:
        while len(self._entries) > self.config.max_entries:
            self._entries.popitem(last=False)
            self._evictions += 1

    def lookup(
        self,
        query: str,
        *,
        query_embedding_unit: Optional[List[float]] = None,
    ) -> Tuple[bool, Optional[str], Optional[float], Optional[Dict[str, Any]]]:
        """
        Lookup a semantically similar cached pipeline response.

        Returns:
            (hit, matched_query, age_ms, cached_response)
        """
        now_s = time.time()
        embedding_unit = query_embedding_unit or self.embed_query_unit(query)

        with self._lock:
            self._lookups += 1
            self._purge_expired_locked(now_s)

            best_key = None
            best_sim = -1.0

            for key, entry in self._entries.items():
                sim = _cosine_sim(embedding_unit, entry.embedding_unit)
                if sim > best_sim:
                    best_sim = sim
                    best_key = key

            if best_key is None or best_sim < self.config.similarity_threshold:
                self._misses += 1
                return False, None, None, None

            entry = self._entries.pop(best_key)
            self._entries[best_key] = entry  # mark as most recently used
            self._hits += 1
            return True, entry.query, entry.age_ms(now_s), entry.response

    def store(
        self,
        query: str,
        response: Dict[str, Any],
        *,
        query_embedding_unit: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        now_s = time.time()
        embedding_unit = query_embedding_unit or self.embed_query_unit(query)

        entry = PipelineCacheEntry(
            query=query,
            embedding_unit=embedding_unit,
            response=response,
            created_at_s=now_s,
            metadata=metadata or {},
        )

        with self._lock:
            if query in self._entries:
                self._entries.pop(query, None)
            self._entries[query] = entry
            self._purge_expired_locked(now_s)
            self._evict_lru_locked()

    def stats(self) -> Dict[str, Any]:
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
                "config": {
                    "ttl_seconds": self.config.ttl_seconds,
                    "max_entries": self.config.max_entries,
                    "similarity_threshold": self.config.similarity_threshold,
                },
            }

