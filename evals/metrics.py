"""Custom evaluation metrics and DeepEval integration."""

from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from deepeval.metrics import (
        HallucinationMetric, 
        AnswerRelevancyMetric, 
        FaithfulnessMetric
    )
    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    logger.warning("DeepEval not available. Install with: pip install deepeval")

from evals.deepeval_cache import get_cache


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
    retrieval_context: Optional[List[str]] = None,
    ground_truths: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate RAG pipeline using DeepEval metrics.
    
    Evaluates generator (answer quality) metrics: AnswerRelevancy, Faithfulness, Hallucination.
    Uses caching to avoid redundant API calls for identical inputs.
    
    Args:
        input_text: The input query
        actual_output: The generated answer from the RAG pipeline
        expected_output: Expected answer (optional, for AnswerRelevancy)
        retrieval_context: List of retrieved document chunks (required for Hallucination and Faithfulness)
        ground_truths: List of ground truth relevant documents (unused, kept for compatibility)
    
    Returns:
        Dictionary with DeepEval metric scores for generator metrics
    """
    if not DEEPEVAL_AVAILABLE:
        return {
            # Generator metrics
            "hallucination_score": None,
            "answer_relevancy_score": None,
            "faithfulness_score": None,
            "error": "DeepEval not available"
        }
    
    # Check cache first
    cache = get_cache()
    cached_result = cache.get(input_text, actual_output, expected_output, retrieval_context, ground_truths)
    if cached_result:
        logger.debug("Using cached DeepEval result")
        return cached_result
    
    results = {
        # Generator metrics
        "hallucination_score": None,
        "answer_relevancy_score": None,
        "faithfulness_score": None
    }
    
    try:
        # Format retrieval context (required for RAG metrics)
        formatted_retrieval_context = None
        if retrieval_context:
            formatted_retrieval_context = [
                ctx if isinstance(ctx, str) else str(ctx)
                for ctx in retrieval_context
            ]
        
        # ===== GENERATOR METRICS =====
        # These evaluate the quality of the generated answer
        
        # Hallucination: Does the answer contain information not in the retrieval context?
        if formatted_retrieval_context:
            try:
                hallucination_metric = HallucinationMetric(threshold=0.7)
                test_case = LLMTestCase(
                    input=input_text,
                    actual_output=actual_output,
                    context=formatted_retrieval_context,  # HallucinationMetric uses 'context' parameter
                    retrieval_context=formatted_retrieval_context
                )
                hallucination_metric.measure(test_case)
                results["hallucination_score"] = hallucination_metric.score
                results["hallucination_passed"] = hallucination_metric.success
            except Exception as e:
                logger.warning(f"Hallucination metric failed: {e}")
                results["hallucination_error"] = str(e)
        
        # Answer Relevancy: How relevant is the answer to the query?
        # Requires expected_output
        if expected_output:
            try:
                relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
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
        
        # Faithfulness: Is the answer faithful to the retrieval context?
        if formatted_retrieval_context:
            try:
                faithfulness_metric = FaithfulnessMetric(threshold=0.5)
                test_case = LLMTestCase(
                    input=input_text,
                    actual_output=actual_output,
                    context=formatted_retrieval_context,  # FaithfulnessMetric uses 'context' parameter
                    retrieval_context=formatted_retrieval_context
                )
                
                # Measure faithfulness (DeepEval makes multiple API calls which can timeout)
                faithfulness_metric.measure(test_case)
                results["faithfulness_score"] = faithfulness_metric.score
                results["faithfulness_passed"] = faithfulness_metric.success
                
            except Exception as e:
                error_str = str(e)
                # Check for timeout-related errors
                if "TimeoutError" in error_str or "RetryError" in error_str or "timeout" in error_str.lower():
                    logger.warning(
                        f"Faithfulness metric timed out (likely due to long answer/context or API delays): {error_str}"
                    )
                    results["faithfulness_error"] = "Timeout - answer or context too long, or API delay"
                elif "429" in error_str or "quota" in error_str.lower():
                    logger.warning(f"Faithfulness metric failed due to rate limiting: {error_str}")
                    results["faithfulness_error"] = "Rate limit - API quota exceeded"
                else:
                    logger.warning(f"Faithfulness metric failed: {e}")
                    results["faithfulness_error"] = str(e)
    
    except Exception as e:
        logger.error(f"DeepEval evaluation failed: {e}")
        results["error"] = str(e)
    
    # Cache the result (even if it has errors, to avoid retrying failed cases)
    cache.set(input_text, actual_output, results, expected_output, retrieval_context, ground_truths)
    
    return results


def evaluate_with_deepeval_automatic_upload(
    test_cases: List[Dict[str, Any]],
    metrics: Optional[List] = None
) -> Dict[str, Any]:
    """
    Evaluate test cases using DeepEval's evaluate() function for automatic Confident AI uploads.
    
    This function uses DeepEval's native evaluate() function which automatically uploads
    results to Confident AI when CONFIDENT_API_KEY is set.
    
    Args:
        test_cases: List of LLMTestCase objects from DeepEval
        metrics: List of DeepEval metric instances (optional, uses defaults if None)
    
    Returns:
        Dictionary with evaluation results and Confident AI report URL
    """
    if not DEEPEVAL_AVAILABLE:
        logger.error("DeepEval not available. Cannot use automatic upload.")
        return {"error": "DeepEval not available"}
    
    try:
        # Default metrics if not provided
        
        metrics = [
                HallucinationMetric(threshold=0.6),
                AnswerRelevancyMetric(threshold=0.6),
                FaithfulnessMetric(threshold=0.6)
            ]
        
        # Run evaluation with automatic Confident AI upload
        logger.info("Running DeepEval evaluation with automatic Confident AI upload...")
        results = evaluate(test_cases, metrics=metrics)
        
        logger.success("Evaluation completed and uploaded to Confident AI")
        return {
            "success": True,
            "results": results,
            "confident_ai_uploaded": True
        }
        
    except Exception as e:
        logger.error(f"DeepEval automatic upload evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "confident_ai_uploaded": False
        }

