# Evaluation Framework

This directory contains the evaluation framework for the Multimodal Enterprise RAG system.

## Overview

The evaluation framework uses **DeepEval** as the primary metrics engine and **Confident AI** for hosted evaluation reports. All evaluations run locally, but results are automatically uploaded to Confident AI for visualization and tracking.

The evaluation process is split into two modular steps:

1. **Data Ingestion** (`ingest_test_data.py`) - Ingests test data from SQuAD v2 into Neo4j and Qdrant
2. **Evaluation** (`test_suite.py`) - Runs queries against the ingested data and measures performance using DeepEval metrics

## Files

- `run_evaluation.py` - Main script that orchestrates ingestion and evaluation
- `ingest_test_data.py` - Script to ingest test datasets into the system
- `test_suite.py` - Test suite that loads test cases and runs evaluations with DeepEval
- `metrics.py` - DeepEval metrics integration (Hallucination, Answer Relevancy, Faithfulness)
- `confident_ai_client.py` - Client for uploading results to Confident AI
- `deepeval_cache.py` - Caching system for DeepEval results to reduce API calls
- `evaluation_report_template.md` - Template for evaluation reports
- `EVALUATION_LIBRARIES.md` - Detailed documentation on DeepEval and Confident AI

## Quick Start

### Basic Usage

Run the complete evaluation pipeline (ingestion + evaluation):

```bash
python evals/run_evaluation.py --test-cases 10
```

This will:
1. Ingest 10 test cases from SQuAD v2 into the system
2. Run evaluation using DeepEval metrics
3. Save results to `logs/eval_results.json`
4. Upload results to Confident AI (if configured)

### Skip Ingestion

If test data is already ingested:

```bash
python evals/run_evaluation.py --test-cases 10 --skip-ingestion
```

### Parallel Execution

Run evaluations in parallel for faster execution:

```bash
python evals/run_evaluation.py --test-cases 20 --parallel 3
```

This uses 3 parallel workers to evaluate test cases concurrently.

### Automatic Confident AI Upload

Use DeepEval's automatic upload feature:

```bash
python evals/run_evaluation.py --test-cases 10 --use-automatic-upload
```

This requires `CONFIDENT_API_KEY` to be set in your `.env` file.

## Command Line Arguments

- `--test-cases N`: Number of SQuAD v2 test cases to evaluate (default: 10)
- `--skip-ingestion`: Skip data ingestion (assumes data is already ingested)
- `--parallel N`: Number of parallel workers for evaluation (default: 1)
- `--use-automatic-upload`: Use DeepEval's automatic Confident AI upload

## Test Data Sources

Currently, the evaluation framework uses:

- **SQuAD v2**: Text-based question answering (validation split)
  - Primary dataset for evaluation
  - Provides questions, contexts, and expected answers
  - Supports factual lookup queries

**Note**: DocVQA and FLEURS datasets are supported in the codebase but are currently disabled (set to 0 samples) in the default configuration. They can be enabled by modifying `run_evaluation.py` if needed.

## Evaluation Metrics

The framework uses **DeepEval** metrics following RAG evaluation best practices:

### Generator Metrics (Answer Quality)

1. **Hallucination Detection** (`HallucinationMetric`)
   - Detects if the answer contains information not present in the retrieval context
   - Score: 0-1 (lower is better, 0 = no hallucination)
   - Threshold: 0.2 (configurable)

2. **Answer Relevancy** (`AnswerRelevancyMetric`)
   - Measures how relevant the answer is to the query
   - Score: 0-1 (higher is better)
   - Threshold: 0.7 (configurable)

3. **Faithfulness** (`FaithfulnessMetric`)
   - Verifies that the answer is grounded in the retrieved context
   - Score: 0-1 (higher is better)
   - Threshold: 0.7 (configurable)

### Performance Metrics

- **Latency**: Query processing time (mean, P50, P95, P99)
- **Success Rate**: Percentage of queries that return valid answers
- **Error Rate**: Percentage of queries that fail

Results are saved to `logs/eval_results.json` with detailed metrics for each test case.

## Confident AI Integration

The framework automatically uploads evaluation results to Confident AI for hosted reporting:

### Setup

1. Sign up at https://www.confident-ai.com/
2. Get your API key from the dashboard
3. Add to `.env`:
   ```
   CONFIDENT_API_KEY=your_api_key_here
   CONFIDENT_AI_PROJECT=your_project_name
   ```

### Automatic Upload

DeepEval automatically uploads results when `CONFIDENT_API_KEY` is set. Use the `--use-automatic-upload` flag to enable this feature.

### Viewing Results

After evaluation, you'll receive:
- A link to the Confident AI dashboard
- Historical tracking of evaluation runs
- Performance trends over time
- Comparison of different model versions

## Caching

DeepEval results are cached to reduce redundant API calls:

- Cache location: `logs/deepeval_cache.json`
- Cache is enabled by default (`DEEPEVAL_CACHE_ENABLED=true` in `.env`)
- Cache can be managed using `scripts/manage_deepeval_cache.py`

## Workflow

### 1. Ingestion Phase

When `--skip-ingestion` is not used:

1. Loads test datasets from HuggingFace
2. Creates temporary files from contexts
3. Processes through ingestion pipeline:
   - Extracts text content
   - Chunks content intelligently
   - Extracts entities and relationships using GPT-4o
   - Classifies content into domains
4. Builds knowledge graph in Neo4j
5. Creates embeddings and indexes in Qdrant vector store

### 2. Evaluation Phase

1. Loads test queries and expected answers from test cases
2. For each test case:
   - Runs query through query pipeline
   - Retrieves context using hybrid search
   - Generates answer using GPT-4o
   - Evaluates using DeepEval metrics:
     * Hallucination detection
     * Answer relevancy
     * Faithfulness
   - Measures latency
3. Aggregates metrics across all test cases
4. Uploads results to Confident AI (if configured)
5. Saves results to `logs/eval_results.json`

## Configuration

### Environment Variables

Required:
- `OPENAI_API_KEY`: OpenAI API key for LLM services

Optional:
- `CONFIDENT_API_KEY`: Confident AI API key for result uploads
- `CONFIDENT_AI_PROJECT`: Confident AI project name
- `DEEPEVAL_CACHE_ENABLED`: Enable/disable caching (default: true)
- `EVAL_LOG_PATH`: Path to evaluation results file (default: `logs/eval_results.json`)

### Metric Thresholds

Thresholds can be configured in `evals/metrics.py`:
- `HALLUCINATION_THRESHOLD`: Default 0.2
- `ANSWER_RELEVANCY_THRESHOLD`: Default 0.7
- `FAITHFULNESS_THRESHOLD`: Default 0.7

## Example Output

```
================================================================================
STEP 1: Ingesting test data into the system
================================================================================
Loading 10 samples from SQuAD v2...
Ingesting test data...
Test data ingestion completed

================================================================================
STEP 2: Building test suite
================================================================================
Test suite built with 10 test cases

================================================================================
STEP 3: Running evaluation
================================================================================
Running evaluation with 1 workers...
Evaluating test case 1/10...
Evaluating test case 2/10...
...
Evaluation completed

================================================================================
STEP 4: Saving evaluation results
================================================================================
Evaluation results saved to logs/eval_results.json

================================================================================
EVALUATION SUMMARY
================================================================================
Total Test Cases: 10
Success Rate: 100.0%
Average Latency: 2.5s

Hallucination Metric:
  Average Score: 0.15 (lower is better)
  Pass Rate: 90.0%

Answer Relevancy Metric:
  Average Score: 0.85 (higher is better)
  Pass Rate: 100.0%

Faithfulness Metric:
  Average Score: 0.88 (higher is better)
  Pass Rate: 100.0%
```

## Troubleshooting

### Common Issues

1. **OpenAI API Errors**
   - Verify `OPENAI_API_KEY` is set in `.env`
   - Check your OpenAI account has credits
   - Ensure access to GPT-4 models

2. **Database Connection Errors**
   - Ensure Docker services (Neo4j, Qdrant) are running
   - Check database credentials in `.env`

3. **Confident AI Upload Failures**
   - Verify `CONFIDENT_API_KEY` is correct
   - Check network connectivity
   - Review logs for detailed error messages

4. **Slow Evaluation**
   - Use `--parallel N` to enable parallel execution
   - Enable caching (`DEEPEVAL_CACHE_ENABLED=true`)
   - Reduce number of test cases for initial testing

5. **Memory Issues**
   - Reduce `--parallel` workers if running out of memory
   - Process fewer test cases at a time

## Notes

- Ingestion can take a while depending on sample sizes (each test case requires entity extraction)
- Make sure Docker services (Neo4j, Qdrant) are running before ingestion
- Ensure OpenAI API key is configured for entity extraction and embeddings
- Test data is ingested into the same databases used by the main system
- DeepEval caching significantly reduces API calls and costs during development
- Parallel execution speeds up evaluation but increases API rate limit usage

## Additional Resources

- `EVALUATION_LIBRARIES.md`: Detailed documentation on DeepEval and Confident AI
- DeepEval Documentation: https://deepeval.com/docs/getting-started-rag
- Confident AI Documentation: https://www.confident-ai.com/docs

