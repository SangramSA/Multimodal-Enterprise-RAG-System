# Confident AI Integration Guide

## Issue: 404 Error on Custom Upload

The error `404 Client Error: Not Found for url: https://api.confident-ai.com/v1/evaluations/test-runs` occurs because:

1. **Custom endpoint doesn't exist**: We were trying to use a custom API endpoint that doesn't exist
2. **DeepEval handles uploads automatically**: DeepEval has built-in Confident AI integration that works differently

## Solution: Use DeepEval's Native Integration

DeepEval automatically uploads results to Confident AI when you use the `evaluate()` function. However, our current implementation uses `measure()` directly for more control.

### Option 1: Use DeepEval's `evaluate()` Function (Recommended for Automatic Uploads)

To enable automatic Confident AI uploads, you would need to refactor to use DeepEval's `evaluate()` function:

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    HallucinationMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric
)

# Create test cases
test_cases = []
for result in per_test_results:
    test_case = LLMTestCase(
        input=result["query"],
        actual_output=result["answer"],
        expected_output=result.get("expected_answer"),
        retrieval_context=[doc["content"] for doc in result["retrieved_documents"][:3]]
    )
    test_cases.append(test_case)

# Define metrics
metrics = [
    HallucinationMetric(threshold=0.5),
    AnswerRelevancyMetric(threshold=0.7),
    FaithfulnessMetric(threshold=0.7),
    ContextualRelevancyMetric(threshold=0.7)
]

# Evaluate with automatic Confident AI upload
evaluate(test_cases, metrics=metrics)
```

**Setup**:
```bash
# Set CONFIDENT_API_KEY (DeepEval's expected name)
export CONFIDENT_API_KEY=your_api_key
```

### Option 2: Keep Current Implementation (No Automatic Uploads)

Our current implementation:
- ✅ Uses `measure()` directly for fine-grained control
- ✅ Supports caching
- ✅ Supports parallel execution
- ❌ No automatic Confident AI uploads
- ✅ Results saved to `logs/eval_results.json`

This is fine if you don't need automatic uploads. You can manually view results in the JSON file.

### Option 3: Manual Upload (Future Enhancement)

If you need to upload results manually, you would need to:
1. Use Confident AI's actual API endpoint (not the one we tried)
2. Format data according to Confident AI's API specification
3. Handle authentication properly

However, this is not recommended as DeepEval's native integration is the supported approach.

## Current Status

- ✅ Custom upload is deprecated (no more 404 errors)
- ✅ Evaluation framework works correctly
- ✅ Results are saved locally to `logs/eval_results.json`
- ⚠️ Automatic Confident AI uploads require using `evaluate()` function

## Recommendation

For now, **keep using the current implementation** (Option 2). The evaluation framework works correctly and saves all results locally. If you need Confident AI dashboards, consider refactoring to use DeepEval's `evaluate()` function in the future.

## References

- [DeepEval Confident AI Integration](https://deepeval.com/docs/getting-started-rag)
- [DeepEval evaluate() function](https://deepeval.com/docs/llm-evals/introduction)

