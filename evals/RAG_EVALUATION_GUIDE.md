# RAG Evaluation Guide

This document explains how the evaluation framework follows [DeepEval's RAG evaluation best practices](https://deepeval.com/docs/getting-started-rag).

## Overview

According to DeepEval's documentation, RAG evaluation should treat the **retriever** and **generator** as separate components. This is because in a RAG pipeline, the final output is only as good as the context you've fed into your LLM.

## Evaluation Architecture

### Component Separation

The framework evaluates two components separately:

1. **Retriever**: Evaluates the quality of retrieved documents
2. **Generator**: Evaluates the quality of the generated answer

### Metrics Used

#### Retriever Metrics (RAG Triad)

These metrics evaluate how well the retriever finds relevant documents:

1. **ContextualRelevancyMetric**
   - Measures: How relevant are the retrieved contexts to the query?
   - Requires: `retrieval_context` (the actual retrieved document chunks)
   - Threshold: 0.7 (default)

2. **ContextualPrecisionMetric**
   - Measures: Of the retrieved contexts, how many are actually relevant?
   - Requires: `retrieval_context` + `ground_truths` (list of relevant documents)
   - Threshold: 0.7 (default)
   - **Note**: Currently not calculated as we don't have ground truth relevant documents

3. **ContextualRecallMetric**
   - Measures: Of all relevant contexts, how many were retrieved?
   - Requires: `retrieval_context` + `ground_truths` (list of relevant documents)
   - Threshold: 0.7 (default)
   - **Note**: Currently not calculated as we don't have ground truth relevant documents

#### Generator Metrics (Answer Quality)

These metrics evaluate how well the generator creates answers:

1. **HallucinationMetric**
   - Measures: Does the answer contain information not in the retrieval context?
   - Requires: `retrieval_context` (to check against)
   - Threshold: 0.5 (default)
   - Lower is better (0 = no hallucination)

2. **AnswerRelevancyMetric**
   - Measures: How relevant is the answer to the query?
   - Requires: `expected_output` (ground truth answer)
   - Threshold: 0.7 (default)
   - Higher is better

3. **FaithfulnessMetric**
   - Measures: Is the answer faithful to the retrieval context?
   - Requires: `retrieval_context` (to verify faithfulness)
   - Threshold: 0.7 (default)
   - Higher is better

## Implementation Details

### Test Case Structure

Following DeepEval's recommendations, each test case uses `LLMTestCase` with:

```python
LLMTestCase(
    input=input_text,                    # The query
    actual_output=actual_output,         # Generated answer
    expected_output=expected_output,     # Ground truth (optional)
    retrieval_context=retrieval_context,  # Retrieved document chunks
    context=retrieval_context           # Also used for some metrics
)
```

### Key Implementation Points

1. **`retrieval_context` Parameter**: 
   - This is the actual retrieved document chunks from the RAG pipeline
   - Extracted from `response.get("retrieved_documents")`
   - Truncated to 500 chars per document to prevent timeouts

2. **Component Separation**:
   - Retriever metrics evaluate `retrieval_context` quality
   - Generator metrics evaluate `actual_output` quality
   - Both use the same `retrieval_context` but evaluate different aspects

3. **Ground Truths**:
   - Currently set to `None` as test cases don't include ground truth relevant documents
   - To enable ContextualPrecision/Recall, add `ground_truths` to test cases

## Usage

### Running Evaluations

```bash
# Standard evaluation
python evals/run_evaluation.py --skip-ingestion

# With parallel execution
python evals/run_evaluation.py --skip-ingestion --parallel 4
```

### Output

The evaluation results include:

**Generator Metrics**:
- `avg_hallucination_score`: Average hallucination score (lower is better)
- `avg_answer_relevancy`: Average answer relevancy (higher is better)
- `avg_faithfulness`: Average faithfulness (higher is better)

**Retriever Metrics**:
- `avg_contextual_relevancy`: Average contextual relevancy (higher is better)
- `avg_contextual_precision`: Average contextual precision (if ground truths provided)
- `avg_contextual_recall`: Average contextual recall (if ground truths provided)

## Future Enhancements

To fully implement DeepEval's RAG evaluation:

1. **Add Ground Truths**: Include ground truth relevant documents in test cases to enable ContextualPrecision and ContextualRecall
2. **Component-Level Tracing**: Use `@observe` decorator to trace retriever and generator separately
3. **Dataset Structure**: Consider using DeepEval's `EvaluationDataset` and `Golden` classes

## References

- [DeepEval RAG Evaluation Guide](https://deepeval.com/docs/getting-started-rag)
- [DeepEval RAG Evaluation Guides](https://deepeval.com/guides/guides-rag-evaluation)
- [DeepEval RAG Triad](https://deepeval.com/guides/guides-rag-triad)

