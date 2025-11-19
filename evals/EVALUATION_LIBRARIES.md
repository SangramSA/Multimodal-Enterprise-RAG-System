# Evaluation Framework

## Overview

The evaluation framework uses **DeepEval** as the primary metrics engine and **Confident AI** for hosted evaluation reports. All evaluations run locally, but results are automatically uploaded to Confident AI for visualization and tracking.

The framework follows [DeepEval's RAG evaluation best practices](https://deepeval.com/docs/getting-started-rag), evaluating both **retriever** and **generator** components separately.

## DeepEval Metrics

The framework uses DeepEval's RAG evaluation metrics, following the recommended approach:

### Retriever Metrics (RAG Triad)

These metrics evaluate the quality of retrieved documents:

#### 1. Contextual Relevancy
- **Purpose**: Measures how relevant the retrieved contexts are to the query
- **Requires**: `retrieval_context` (retrieved document chunks)
- **Output**: Score between 0-1 (higher is better)
- **Metric**: `ContextualRelevancyMetric`

#### 2. Contextual Precision
- **Purpose**: Of the retrieved contexts, how many are actually relevant?
- **Requires**: `retrieval_context` and `ground_truths` (list of relevant documents)
- **Output**: Score between 0-1 (higher is better)
- **Metric**: `ContextualPrecisionMetric`

#### 3. Contextual Recall
- **Purpose**: Of all relevant contexts, how many were retrieved?
- **Requires**: `retrieval_context` and `ground_truths` (list of relevant documents)
- **Output**: Score between 0-1 (higher is better)
- **Metric**: `ContextualRecallMetric`

### Generator Metrics (Answer Quality)

These metrics evaluate the quality of the generated answer:

#### 1. Hallucination Detection
- **Purpose**: Detects if the answer contains information not present in the retrieval context
- **Requires**: `retrieval_context` (retrieved documents)
- **Output**: Score between 0-1 (lower is better, 0 = no hallucination)
- **Metric**: `HallucinationMetric`

#### 2. Answer Relevancy
- **Purpose**: Measures how relevant the answer is to the query
- **Requires**: `expected_output` (ground truth answer)
- **Output**: Score between 0-1 (higher is better)
- **Metric**: `AnswerRelevancyMetric`

#### 3. Faithfulness
- **Purpose**: Evaluates if the answer is faithful to the retrieved context
- **Requires**: `retrieval_context` (retrieved documents)
- **Output**: Score between 0-1 (higher is better)
- **Metric**: `FaithfulnessMetric`

### Additional Metrics
- **Latency**: Mean, P95 latency for query processing
- **Per-Test Results**: Individual test case results with all metrics

## Confident AI Integration

### Important Note

**DeepEval automatically uploads results to Confident AI** when using the `evaluate()` function. However, our current implementation uses `measure()` directly for more granular control, which means automatic uploads are not available.

### Setup for DeepEval's Native Integration

To enable DeepEval's automatic Confident AI uploads (when using `evaluate()`):

```bash
# DeepEval expects CONFIDENT_API_KEY (not CONFIDENT_AI_API_KEY)
CONFIDENT_API_KEY=your_api_key
```

### Current Implementation

Our evaluation framework:
1. **Uses `measure()` directly**: For fine-grained control over individual metrics
2. **Custom upload disabled**: The custom endpoint doesn't exist (404 error expected)
3. **Local results**: All results are saved to `logs/eval_results.json`

### Future Enhancement

To enable automatic Confident AI uploads, consider refactoring to use DeepEval's `evaluate()` function:

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase

# Create test cases
test_cases = [LLMTestCase(...) for ...]

# Evaluate with automatic Confident AI upload
evaluate(test_cases, metrics=[...])
```

### Benefits (when using evaluate())

- **Hosted Reports**: Access evaluation reports from anywhere
- **Historical Tracking**: Compare results across different runs
- **Team Collaboration**: Share reports with team members
- **Visual Analytics**: Interactive dashboards for metric analysis

## Usage

### Running Evaluations

```bash
# Full evaluation (ingest + evaluate)
python evals/run_evaluation.py

# Skip ingestion (if data already ingested)
python evals/run_evaluation.py --skip-ingestion

# Custom sample sizes
python evals/run_evaluation.py --squad-samples 50 --docvqa-samples 25 --fleurs-samples 25

# Parallel execution (faster for large test suites)
python evals/run_evaluation.py --skip-ingestion --parallel 4
```

### Output

Evaluation results are saved to:
- **Local**: `logs/eval_results.json`
- **Confident AI**: Dashboard URL (if enabled)

The console output includes:
- Summary of all metrics
- Link to Confident AI dashboard (if enabled)
- Performance metrics (latency)

### Caching

DeepEval metric results are automatically cached to avoid redundant API calls. The cache uses MD5 hashes of inputs (query, answer, expected answer, context) as keys.

**Cache Management**:
```bash
# View cache statistics
python scripts/manage_deepeval_cache.py stats

# Clear cache (useful when test cases change)
python scripts/manage_deepeval_cache.py clear

# View cache contents
python scripts/manage_deepeval_cache.py show
```

**Disable Caching**:
```bash
# In .env file
DEEPEVAL_CACHE_ENABLED=false
```

**Benefits**:
- Faster re-runs: Identical test cases use cached results
- Cost savings: Avoids duplicate OpenAI API calls for same inputs
- Resume capability: Cached results preserved if evaluation fails mid-run

## Architecture

```
Test Suite (test_suite.py)
    ↓
DeepEval Metrics (metrics.py)
    ↓
Evaluation Results (aggregated)
    ↓
Confident AI Client (confident_ai_client.py)
    ↓
Confident AI Dashboard
```

## Future Enhancements

- Multi-turn conversation evaluation
- Prompt versioning and A/B testing
- Custom metric definitions
- CI/CD integration for automated evaluations

