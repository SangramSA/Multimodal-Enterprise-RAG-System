"""Custom evaluation metrics and DeepEval integration."""

from typing import List, Dict, Any, Optional
import time
from loguru import logger

try:
    from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    logger.warning("DeepEval not available. Install with: pip install deepeval")


def calculate_precision_at_k(retrieved: List[Dict[str, Any]], relevant: List[str], k: int = 10) -> float:
    """Calculate precision@k."""
    if not retrieved or k == 0:
        return 0.0
    
    top_k = retrieved[:k]
    relevant_ids = set(relevant)
    
    relevant_retrieved = sum(1 for item in top_k if item.get("chunk_id") in relevant_ids)
    return relevant_retrieved / min(len(top_k), k)


def calculate_recall_at_k(retrieved: List[Dict[str, Any]], relevant: List[str], k: int = 10) -> float:
    """Calculate recall@k."""
    if not relevant:
        return 0.0
    
    top_k = retrieved[:k]
    relevant_ids = set(relevant)
    
    relevant_retrieved = sum(1 for item in top_k if item.get("chunk_id") in relevant_ids)
    return relevant_retrieved / len(relevant_ids)


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_exact_match(predicted: str, expected: str) -> bool:
    """Check if predicted answer exactly matches expected."""
    return predicted.strip().lower() == expected.strip().lower()


def calculate_semantic_similarity(predicted: str, expected: str) -> float:
    """Calculate semantic similarity (simplified - would use embeddings in production)."""
    # Simple word overlap for now
    pred_words = set(predicted.lower().split())
    exp_words = set(expected.lower().split())
    
    if not pred_words or not exp_words:
        return 0.0
    
    intersection = pred_words & exp_words
    union = pred_words | exp_words
    
    return len(intersection) / len(union) if union else 0.0


def measure_latency(timings: List[float]) -> Dict[str, float]:
    """Calculate latency percentiles."""
    if not timings:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    
    sorted_timings = sorted(timings)
    n = len(sorted_timings)
    
    return {
        "p50": sorted_timings[n // 2],
        "p95": sorted_timings[int(n * 0.95)] if n > 1 else sorted_timings[0],
        "p99": sorted_timings[int(n * 0.99)] if n > 1 else sorted_timings[0],
        "mean": sum(timings) / n
    }


def evaluate_with_deepeval(
    input_text: str,
    actual_output: str,
    expected_output: Optional[str] = None,
    context: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate using DeepEval metrics (hallucination, relevancy, faithfulness).
    
    Args:
        input_text: The input query
        actual_output: The generated answer
        expected_output: Expected answer (optional)
        context: List of context strings used for retrieval (optional)
    
    Returns:
        Dictionary with DeepEval metric scores
    """
    if not DEEPEVAL_AVAILABLE:
        return {
            "hallucination_score": None,
            "answer_relevancy_score": None,
            "faithfulness_score": None,
            "error": "DeepEval not available"
        }
    
    results = {
        "hallucination_score": None,
        "answer_relevancy_score": None,
        "faithfulness_score": None
    }
    
    try:
        # Hallucination detection (requires context)
        if context:
            try:
                hallucination_metric = HallucinationMetric(threshold=0.5)
                test_case = LLMTestCase(
                    input=input_text,
                    actual_output=actual_output,
                    context=" ".join(context[:3])  # Use top 3 contexts
                )
                hallucination_metric.measure(test_case)
                results["hallucination_score"] = hallucination_metric.score
                results["hallucination_passed"] = hallucination_metric.success
            except Exception as e:
                logger.warning(f"Hallucination metric failed: {e}")
                results["hallucination_error"] = str(e)
        
        # Answer Relevancy (requires expected output)
        if expected_output:
            try:
                relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
                test_case = LLMTestCase(
                    input=input_text,
                    actual_output=actual_output,
                    expected_output=expected_output
                )
                relevancy_metric.measure(test_case)
                results["answer_relevancy_score"] = relevancy_metric.score
                results["answer_relevancy_passed"] = relevancy_metric.success
            except Exception as e:
                logger.warning(f"Answer relevancy metric failed: {e}")
                results["answer_relevancy_error"] = str(e)
        
        # Faithfulness (requires context)
        if context:
            try:
                faithfulness_metric = FaithfulnessMetric(threshold=0.7)
                test_case = LLMTestCase(
                    input=input_text,
                    actual_output=actual_output,
                    context=" ".join(context[:3])
                )
                faithfulness_metric.measure(test_case)
                results["faithfulness_score"] = faithfulness_metric.score
                results["faithfulness_passed"] = faithfulness_metric.success
            except Exception as e:
                logger.warning(f"Faithfulness metric failed: {e}")
                results["faithfulness_error"] = str(e)
    
    except Exception as e:
        logger.error(f"DeepEval evaluation failed: {e}")
        results["error"] = str(e)
    
    return results

