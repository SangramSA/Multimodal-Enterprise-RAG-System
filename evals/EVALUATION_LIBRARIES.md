# Evaluation Libraries

## Current Implementation

The evaluation framework uses a **hybrid approach** combining:

### Custom Metrics (`evals/metrics.py`)
- Precision@K
- Recall@K
- F1 Score
- Exact Match
- Semantic Similarity (word overlap)
- Latency metrics (mean, P50, P95, P99)

### DeepEval Metrics (Integrated)
- **Hallucination Detection**: Detects if the answer contains information not present in the context
- **Answer Relevancy**: Measures how relevant the answer is to the query
- **Faithfulness**: Measures how faithful the answer is to the provided context

## DeepEval Integration

DeepEval is now fully integrated into the evaluation framework. The `evaluate_with_deepeval()` function in `evals/metrics.py` provides:

- Hallucination detection (requires context)
- Answer relevancy scoring (requires expected output)
- Faithfulness metrics (requires context)

All metrics are automatically calculated during evaluation and included in the results.

## Usage

DeepEval metrics are automatically used when:
- Context is available (for hallucination and faithfulness)
- Expected output is available (for answer relevancy)

The evaluation results include both custom and DeepEval metrics for comprehensive assessment.

## Other Evaluation Libraries

- **Arize Phoenix**: LLM observability and evaluation
- **LangSmith**: LangChain's evaluation platform
- **TruLens**: LLM evaluation framework

These could be integrated if needed for production use.

