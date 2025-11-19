# Multimodal Enterprise RAG System

A modular, evaluation-first multimodal Retrieval-Augmented Generation (RAG) system that supports text, image, and audio ingestion, builds a searchable knowledge graph, and enables hybrid search using graph traversal, keyword filtering, and semantic vector retrieval.

## Features

- **Multimodal Ingestion**: Support for PDF, TXT, JPG, PNG, MP3 files
- **Knowledge Graph**: Neo4j-based graph construction with entity and relationship extraction
- **Vector Database**: Qdrant for semantic search
- **Hybrid Search**: Combines graph traversal, keyword search, and vector similarity
- **Agent Orchestration**: LangChain-based retrieval agents, CrewAI multi-agent framework (optional)
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
6. **Agent Orchestration**: LangChain agents for retrieval, CrewAI for multi-agent orchestration (optional)
7. **Query Pipeline**: End-to-end query processing with agentic pipeline support

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

The evaluation framework uses **DeepEval** for metrics calculation and **Confident AI** for hosted evaluation reports.

#### Metrics

The system evaluates using DeepEval RAG metrics, following [DeepEval's RAG evaluation best practices](https://deepeval.com/docs/getting-started-rag):

**Retriever Metrics (RAG Triad)**:
- **Contextual Relevancy**: How relevant are the retrieved contexts to the query?
- **Contextual Precision**: Of the retrieved contexts, how many are relevant? (requires ground truths)
- **Contextual Recall**: Of all relevant contexts, how many were retrieved? (requires ground truths)

**Generator Metrics (Answer Quality)**:
- **Hallucination Score**: Detects if the answer contains information not present in the retrieval context
- **Answer Relevancy**: Measures how relevant the answer is to the query
- **Faithfulness**: Evaluates if the answer is faithful to the retrieved context

#### Running Evaluations

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

**Parallel Execution**:
```bash
# Run with 4 parallel workers (faster for large test suites)
python evals/run_evaluation.py --skip-ingestion --parallel 4

# Combine with custom sample sizes
python evals/run_evaluation.py --squad-samples 100 --parallel 4
```

**Note**: Parallel execution is recommended for large test suites. Start with 2-4 workers and adjust based on your system resources and API rate limits.

#### Confident AI Integration

**Important**: DeepEval automatically uploads results to Confident AI when using the `evaluate()` function. Since we use `measure()` directly for more control, automatic uploads are not available.

To enable Confident AI reporting with DeepEval's native integration:

1. **Set the API key** (DeepEval expects `CONFIDENT_API_KEY`):
```bash
CONFIDENT_API_KEY=your_api_key
```

2. **Optional legacy support** (for custom client):
```bash
CONFIDENT_AI_API_KEY=your_api_key  # Legacy, also sets CONFIDENT_API_KEY
CONFIDENT_AI_PROJECT=your_project_name
CONFIDENT_AI_ENABLED=true
```

**Note**: 
- DeepEval's automatic uploads work when using `evaluate()` function
- Our current implementation uses `measure()` directly, so automatic uploads are not available
- Custom upload endpoint is deprecated (404 error expected)
- To use automatic uploads, consider refactoring to use DeepEval's `evaluate()` function

#### DeepEval Caching

To avoid redundant API calls and speed up evaluations, DeepEval results are automatically cached. The cache stores metric results based on a hash of the inputs (query, answer, expected answer, context).

**Cache Location**: `logs/deepeval_cache.json`

**Enable/Disable Caching**:
```bash
# In .env file
DEEPEVAL_CACHE_ENABLED=true   # Enable (default)
DEEPEVAL_CACHE_ENABLED=false  # Disable
```

**Manage Cache**:
```bash
# View cache statistics
python scripts/manage_deepeval_cache.py stats

# Clear cache
python scripts/manage_deepeval_cache.py clear

# Show cache contents
python scripts/manage_deepeval_cache.py show
```

**Benefits**:
- **Faster re-runs**: Identical test cases use cached results
- **Cost savings**: Avoids duplicate OpenAI API calls
- **Resume capability**: If evaluation fails, cached results are preserved

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

