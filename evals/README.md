# Evaluation Framework

This directory contains the evaluation framework for the Multimodal Enterprise RAG system.

## Overview

The evaluation process is split into two modular steps:

1. **Data Ingestion** (`ingest_test_data.py`) - Ingests test data from SQuAD v2, DocVQA, and FLEURS into Neo4j and Qdrant
2. **Evaluation** (`test_suite.py`) - Runs queries against the ingested data and measures performance

## Files

- `ingest_test_data.py` - Script to ingest test datasets into the system
- `run_evaluation.py` - Main script that orchestrates ingestion and evaluation
- `test_suite.py` - Test suite that loads test cases and runs evaluations
- `metrics.py` - Custom evaluation metrics (precision, recall, F1, latency, etc.)
- `evaluation_report_template.md` - Template for evaluation reports

## Usage

### Option 1: Run Complete Pipeline

Run both ingestion and evaluation together:

```bash
python evals/run_evaluation.py
```

### Option 2: Run Separately

1. First, ingest test data:
```bash
python evals/ingest_test_data.py
```

2. Then run evaluation (skip ingestion):
```bash
python evals/run_evaluation.py --skip-ingestion
```

### Customizing Sample Sizes

```bash
python evals/run_evaluation.py --squad-samples 50 --docvqa-samples 25 --fleurs-samples 25
```

## Test Data Sources

- **SQuAD v2**: Text-based question answering (validation split)
- **DocVQA**: Visual document question answering
- **FLEURS**: Audio transcription and question answering

## Evaluation Metrics

The system measures:

- **Retrieval Quality**: Precision@k, Recall@k, F1 Score
- **Answer Quality**: Exact Match Rate, Semantic Similarity
- **Performance**: Latency (mean, P50, P95, P99)

Results are saved to `logs/eval_results.json`.

## Workflow

1. **Ingestion Phase**:
   - Loads test datasets
   - Creates temporary files from contexts/images/audio
   - Processes through ingestion pipeline
   - Extracts entities and relationships
   - Builds knowledge graph in Neo4j
   - Indexes in Qdrant vector store

2. **Evaluation Phase**:
   - Loads test queries and expected answers
   - Runs queries through query pipeline
   - Measures retrieval and answer quality
   - Calculates metrics
   - Generates evaluation report

## Notes

- Ingestion can take a while depending on sample sizes
- Make sure Docker services (Neo4j, Qdrant) are running before ingestion
- Ensure OpenAI API key is configured for entity extraction and embeddings
- Test data is ingested into the same databases used by the main system

