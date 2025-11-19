# Multimodal Enterprise RAG System

A modular, evaluation-first multimodal Retrieval-Augmented Generation (RAG) system that supports text, image, and audio ingestion, builds a searchable knowledge graph, and enables hybrid search using graph traversal, keyword filtering, and semantic vector retrieval.

## Features

- **Multimodal Ingestion**: Support for PDF, TXT, JPG, PNG, MP3 files
- **Knowledge Graph**: Neo4j-based graph construction with entity and relationship extraction
- **Vector Database**: Qdrant for semantic search
- **Hybrid Search**: Combines graph traversal, keyword search, and vector similarity
- **Agent Orchestration**: LangChain-based retrieval agents
- **Evaluation Framework**: DeepEval-based test suite with metrics
- **Domain Classification**: Automatic domain tagging for documents
- **Streamlit UI**: Interactive web interface for file upload and querying

## Architecture

The system follows a modular pipeline architecture:

1. **Data Ingestion**: Multi-modal processors for text, image, and audio
2. **Entity Extraction**: LLM-based extraction with cross-modal linking
3. **Knowledge Graph**: Neo4j for structured relationships
4. **Vector Database**: Qdrant for semantic search
5. **Hybrid Search**: Graph + Keyword + Vector retrieval
6. **Agent Orchestration**: LangChain agents for retrieval
7. **Query Pipeline**: End-to-end query processing

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- OpenAI API key

## Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Multimodal-Enterprise-RAG
```

2. **Create environment file**:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

3. **Start Docker services**:
```bash
docker compose up -d
```

**Note:** Use `docker compose` (space) for Docker Compose V2, or `docker-compose` (hyphen) for older versions.

This will start:
- Neo4j on ports 7474 (HTTP) and 7687 (Bolt)
- Qdrant on ports 6333 (HTTP) and 6334 (gRPC)

4. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

5. **Initialize databases**:
```bash
python setup/init_databases.py
```

6. **Run the Streamlit UI**:
```bash
streamlit run ui/app.py
```

## Usage

### File Upload

1. Open the Streamlit UI (default: http://localhost:8501)
2. Upload files (PDF, TXT, JPG, PNG, MP3)
3. Files will be processed and indexed automatically

### Querying

1. Enter a natural language query in the query interface
2. Select query type (factual lookup, visual QA, audio QA, etc.)
3. View results with citations and source documents
4. Explore the knowledge graph visualization

### Evaluation

The evaluation process has two steps:

1. **Ingest test data** (first time only):
```bash
python evals/ingest_test_data.py
```

2. **Run evaluation**:
```bash
python evals/run_evaluation.py
```

Or run both steps together:
```bash
python evals/run_evaluation.py
```

To skip ingestion if data is already ingested:
```bash
python evals/run_evaluation.py --skip-ingestion
```

You can also customize the number of samples:
```bash
python evals/run_evaluation.py --squad-samples 50 --docvqa-samples 25 --fleurs-samples 25
```

## Project Structure

```
multimodal-enterprise-rag/
├── docker-compose.yml          # Neo4j, Qdrant services
├── requirements.txt
├── .env                        # API keys, configs
├── setup/                      # Database initialization
├── evals/                      # Evaluation framework
├── ingestion/                  # Multi-modal processors
├── extraction/                 # Entity/relationship extraction
├── graph/                      # Neo4j operations
├── vector/                     # Qdrant operations
├── search/                     # Hybrid search
├── agents/                     # LangChain agents
├── pipeline/                   # End-to-end pipelines
├── ui/                         # Streamlit interface
├── utils/                      # Utilities
└── tests/                      # Tests
```

## Evaluation

The system includes comprehensive evaluation metrics:
- Retrieval quality (precision@k, recall@k)
- Hallucination rate
- Latency (p50, p95, p99)
- Answer relevance
- Exact match and F1 scores

Test cases are extracted from:
- SQuAD v2 (text QA)
- DocVQA (visual document QA)
- FLEURS (audio QA)

## Error Handling

The system includes comprehensive error handling for:
- API failures (retry with exponential backoff)
- Database connection issues
- File processing errors
- Resource limitations
- User input validation

## License

[Specify license]

## Contributing

[Contributing guidelines]

