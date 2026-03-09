"""Run cold vs cached semantic-cache experiments.

This script measures:
- Retrieval latency breakdown (ms) for cold (forced DB) vs cached semantic hit
- Retrieval overlap@K between cold and cached results
- DeepEval answer-quality metrics delta (optional; requires OpenAI + DeepEval)

Example:
  python scripts/cache_experiment.py --n 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# Ensure project imports work when running from any CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from search.cached_vector_search import CachedVectorSearch
from utils.config import LOGS_DIR, OPENAI_API_KEY, OPENAI_MODEL
from vector.embedding_service import EmbeddingService
from vector.qdrant_client import QdrantClientWrapper
from vector.vector_store import VectorStore


@dataclass
class ExperimentCase:
    query: str
    expected_answer: Optional[str] = None
    paraphrase_query: Optional[str] = None


def _load_default_cases(n: int) -> List[ExperimentCase]:
    # Prefer already-generated test suite (created by evals/TestSuite).
    candidate = Path(__file__).parent.parent / "evals" / "test_data" / "test_cases.json"
    if candidate.exists():
        try:
            data = json.loads(candidate.read_text())
            cases: List[ExperimentCase] = []
            for obj in data:
                if not obj.get("query"):
                    continue
                cases.append(
                    ExperimentCase(
                        query=str(obj["query"]),
                        expected_answer=str(obj.get("expected_answer") or "") or None,
                    )
                )
                if len(cases) >= n:
                    break
            if cases:
                logger.info(f"Loaded {len(cases)} cases from {candidate}")
                return cases
        except Exception as e:
            logger.warning(f"Failed to load default test cases: {e}")

    # Fallback: minimal examples (no expected answers → DeepEval answer relevancy may be N/A).
    return [
        ExperimentCase(query="What is Retrieval-Augmented Generation (RAG)?"),
        ExperimentCase(query="How does Qdrant store vectors and payload metadata?"),
        ExperimentCase(query="What is Reciprocal Rank Fusion and why is it useful?"),
    ][:n]


def _extract_latency_breakdown(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        return {}
    mw = results[0].get("middleware") or {}
    return dict(mw.get("latency_breakdown_ms") or {})


def _extract_cache_info(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"cache_hit": False}
    mw = results[0].get("middleware") or {}
    return {
        "cache_hit": bool(mw.get("cache_hit")),
        "cache_similarity": mw.get("cache_similarity"),
        "cache_matched_query": mw.get("cache_matched_query"),
        "cache_age_ms": mw.get("cache_age_ms"),
        "cache_mode": mw.get("cache_mode"),
    }


def _overlap_at_k(a: List[Dict[str, Any]], b: List[Dict[str, Any]], k: int) -> float:
    a_ids = {d.get("chunk_id") for d in a[:k] if d.get("chunk_id")}
    b_ids = {d.get("chunk_id") for d in b[:k] if d.get("chunk_id")}
    if not a_ids and not b_ids:
        return 1.0
    if not a_ids or not b_ids:
        return 0.0
    return len(a_ids.intersection(b_ids)) / float(len(a_ids.union(b_ids)))


def _build_context(docs: List[Dict[str, Any]], top_k: int = 5) -> str:
    parts = []
    for i, d in enumerate(docs[:top_k], 1):
        content = (d.get("content") or "")[:500]
        chunk_id = d.get("chunk_id") or f"doc_{i}"
        modality = (d.get("metadata") or {}).get("modality", "unknown")
        parts.append(f"[Source {i} - {modality} - {chunk_id}]\n{content}\n")
    return "\n\n".join(parts)


def _generate_answer_openai(query: str, context: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for answer generation")

    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "You are a helpful assistant that answers questions based on provided context.\n"
        "Use only the information from the context to answer. If the context doesn't contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that provides accurate answers based on the given context.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return (resp.choices[0].message.content or "").strip()


def _generate_paraphrase_openai(query: str) -> str:
    """Generate a semantically similar paraphrase for a query."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for paraphrase generation")

    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "Rephrase the following question using different words but preserving its meaning. "
        "Return only the rephrased question.\n\n"
        f"Original question: {query}"
    )
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that generates paraphrased questions preserving meaning.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=100,
    )
    return (resp.choices[0].message.content or "").strip()


def _cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


def main() -> int:
    parser = argparse.ArgumentParser(description="Cold vs cached semantic-cache experiment")
    parser.add_argument("--n", type=int, default=10, help="Number of queries to run")
    parser.add_argument(
        "--skip-deepeval",
        action="store_true",
        help="Skip DeepEval metrics (still measures retrieval latency + overlap@K)",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=50,
        help="How many candidates to fetch from vector DB on cold path",
    )
    parser.add_argument(
        "--rerank-k",
        type=int,
        default=5,
        help="How many of the top candidates to rerank locally",
    )

    args = parser.parse_args()

    # Initialize vector store + cached vector search
    qdrant = QdrantClientWrapper()
    embeddings = EmbeddingService()
    vector_store = VectorStore(qdrant, embeddings)
    cached_vs = CachedVectorSearch(
        vector_store,
        candidate_limit=args.candidate_limit,
        rerank_k=args.rerank_k,
    )

    cases = _load_default_cases(args.n)
    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n": len(cases),
        "candidate_limit": args.candidate_limit,
        "rerank_k": args.rerank_k,
        "results": [],
        "summary": {},
    }

    for idx, case in enumerate(cases, 1):
        logger.info(f"[{idx}/{len(cases)}] {case.query}")

        # Determine paraphrased query for semantic cache test
        q1 = case.query
        paraphrase = case.paraphrase_query
        paraphrase_similarity: Optional[float] = None

        if paraphrase is None:
            try:
                candidate = _generate_paraphrase_openai(q1)
                emb_q1 = embeddings.embed_text(q1)
                emb_q2 = embeddings.embed_text(candidate)
                paraphrase_similarity = _cosine_sim(emb_q1, emb_q2)
                # Accept paraphrase only if sufficiently similar
                if paraphrase_similarity >= 0.9:
                    paraphrase = candidate
                    logger.info(
                        f"Paraphrase accepted for experiment case {idx}: sim={paraphrase_similarity:.3f}"
                    )
                else:
                    logger.warning(
                        f"Paraphrase too dissimilar for case {idx} (sim={paraphrase_similarity:.3f}), using original query for cached run."
                    )
            except Exception as e:
                logger.warning(f"Paraphrase generation failed for case {idx}: {e}")

        if paraphrase is None:
            paraphrase = q1  # Fallback: identical string

        # Cold (forced DB) — also stores into cache for the next run
        cold_results = cached_vs.search(q1, limit=10, cache_mode="bypass")
        cold_latency = _extract_latency_breakdown(cold_results)
        cold_cache_info = _extract_cache_info(cold_results)

        # Cached (semantic) run using paraphrased query when available
        cached_results = cached_vs.search(paraphrase, limit=10, cache_mode="enabled")
        cached_latency = _extract_latency_breakdown(cached_results)
        cached_cache_info = _extract_cache_info(cached_results)

        overlap5 = _overlap_at_k(cold_results, cached_results, 5)
        overlap10 = _overlap_at_k(cold_results, cached_results, 10)

        case_result: Dict[str, Any] = {
            "case": asdict(case),
            "paraphrase": paraphrase,
            "paraphrase_similarity": paraphrase_similarity,
            "cold": {
                "cache": cold_cache_info,
                "latency_breakdown_ms": cold_latency,
                "top_chunk_ids": [d.get("chunk_id") for d in cold_results[:10]],
            },
            "cached": {
                "cache": cached_cache_info,
                "latency_breakdown_ms": cached_latency,
                "top_chunk_ids": [d.get("chunk_id") for d in cached_results[:10]],
            },
            "overlap": {"jaccard@5": overlap5, "jaccard@10": overlap10},
            "deepeval": None,
        }

        if not args.skip_deepeval:
            from evals.metrics import evaluate_with_deepeval

            # Generate answers from the retrieved context and evaluate with DeepEval.
            cold_context = _build_context(cold_results, top_k=5)
            cached_context = _build_context(cached_results, top_k=5)

            cold_answer = _generate_answer_openai(case.query, cold_context)
            cached_answer = _generate_answer_openai(case.query, cached_context)

            retrieval_context_cold = [(d.get("content") or "")[:500] for d in cold_results[:3]]
            retrieval_context_cached = [(d.get("content") or "")[:500] for d in cached_results[:3]]

            cold_eval = evaluate_with_deepeval(
                input_text=case.query,
                actual_output=cold_answer,
                expected_output=case.expected_answer,
                retrieval_context=retrieval_context_cold if any(retrieval_context_cold) else None,
                ground_truths=None,
            )
            cached_eval = evaluate_with_deepeval(
                input_text=case.query,
                actual_output=cached_answer,
                expected_output=case.expected_answer,
                retrieval_context=retrieval_context_cached if any(retrieval_context_cached) else None,
                ground_truths=None,
            )

            case_result["deepeval"] = {
                "cold": {"answer": cold_answer, "metrics": cold_eval},
                "cached": {"answer": cached_answer, "metrics": cached_eval},
            }

        report["results"].append(case_result)

    # Summaries
    def avg(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    cold_totals = [
        float(r["cold"]["latency_breakdown_ms"].get("total_ms", 0.0)) for r in report["results"]
    ]
    cached_totals = [
        float(r["cached"]["latency_breakdown_ms"].get("total_ms", 0.0)) for r in report["results"]
    ]
    overlaps5 = [float(r["overlap"]["jaccard@5"]) for r in report["results"]]

    report["summary"] = {
        "avg_cold_total_ms": avg(cold_totals),
        "avg_cached_total_ms": avg(cached_totals),
        "avg_speedup_x": (avg(cold_totals) / avg(cached_totals)) if avg(cached_totals) else None,
        "avg_jaccard@5": avg(overlaps5),
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOGS_DIR / f"cache_experiment_{int(time.time())}.json"
    out_path.write_text(json.dumps(report, indent=2))
    logger.success(f"Wrote report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

