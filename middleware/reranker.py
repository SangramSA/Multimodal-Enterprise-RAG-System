"""Lightweight local reranker for retrieval results (Cross-Encoder).

POC behavior:
- Caller fetches a larger candidate set from the vector DB (e.g., 50 docs).
- We rerank only a small prefix (e.g., top 5 by initial score) with a local model.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class LocalCrossEncoder:
    """Local cross-encoder reranker (lazy import)."""

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
                from sentence_transformers import CrossEncoder  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "sentence-transformers is required for reranking. "
                    "Install it (and its torch dependency) to enable this feature."
                ) from e
            self._model = CrossEncoder(self.model_name)
            return self._model

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        model = self._get_model()
        scores = model.predict(pairs)
        # sentence-transformers returns numpy array or list depending on backend
        return [float(x) for x in list(scores)]


@dataclass(frozen=True)
class RerankerConfig:
    # Small, commonly used model; can be overridden via config/env later.
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_k: int = 5
    content_max_chars: int = 800


class CrossEncoderReranker:
    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()
        self._model = LocalCrossEncoder(self.config.model_name)

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        *,
        rerank_k: Optional[int] = None,
        score_key: str = "score",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Rerank only the top-K docs by initial score.

        Returns:
            (reranked_docs, timings_ms)
        """
        k = rerank_k if rerank_k is not None else self.config.rerank_k
        if not docs or k <= 0:
            return docs, {"rerank_ms": 0.0}

        # Keep stable ordering for docs outside the reranked prefix.
        sorted_docs = sorted(docs, key=lambda d: float(d.get(score_key, 0.0)), reverse=True)
        prefix = sorted_docs[:k]
        suffix = sorted_docs[k:]

        pairs: List[Tuple[str, str]] = []
        for d in prefix:
            content = d.get("content") or ""
            content = content[: self.config.content_max_chars]
            pairs.append((query, content))

        t0 = time.perf_counter()
        scores = self._model.score_pairs(pairs)
        rerank_ms = (time.perf_counter() - t0) * 1000.0

        rescored: List[Dict[str, Any]] = []
        for d, s in zip(prefix, scores):
            d2 = dict(d)
            d2["rerank_score"] = float(s)
            rescored.append(d2)

        rescored.sort(key=lambda d: float(d.get("rerank_score", 0.0)), reverse=True)

        return rescored + suffix, {"rerank_ms": rerank_ms}

