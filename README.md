# Multimodal Enterprise RAG System

A modular, evaluation-first multimodal Retrieval-Augmented Generation (RAG) system that supports text, image, and audio ingestion, builds a searchable knowledge graph, and enables hybrid search using graph traversal, keyword filtering, and semantic vector retrieval.

## Features

- **Multimodal Ingestion**: Support for PDF, TXT, JPG, PNG, MP3 files with intelligent chunking
- **Knowledge Graph**: Neo4j-based graph construction with entity and relationship extraction
- **Vector Database**: Qdrant for semantic search with rich metadata
- **Hybrid Search**: Combines graph traversal, keyword search, and vector similarity using RRF
- **Agentic Query Pipeline**: 5-stage agent orchestration (validation, triage, retrieval, generation, post-processing)
- **CrewAI Integration**: Optional multi-agent framework for sophisticated orchestration
- **Multi-Step Reasoning**: Advanced reasoning capabilities with transparent step-by-step explanations
- **Evaluation Framework**: DeepEval-based test suite with automatic Confident AI reporting
- **Domain Classification**: Automatic domain tagging for documents
- **Interactive Graph Explorer**: Visual graph exploration using pyvis (similar to Neo4j browser)
- **Telemetry & Observability**: Comprehensive metrics tracking with LangSmith integration
- **Streamlit UI**: Interactive web interface for file upload, querying, graph exploration, and evaluation

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
2. Choose between standard pipeline or CrewAI orchestration (optional)
3. View results with citations and source documents
4. Review reasoning steps for complex queries (if using agentic pipeline)
5. Explore the knowledge graph interactively using the Graph Explorer

#### Agentic Query Pipeline

The system supports an advanced agentic query pipeline with 5 specialized agents:

1. **Query Validation Agent**: Validates queries with security checks and complexity assessment
2. **Query Triage Agent**: Classifies queries and selects optimal search strategy
3. **Retrieval Orchestration Agent**: Orchestrates multiple search methods with direct data store access
4. **Answer Generation Agent**: Generates answers with multi-step reasoning when needed
5. **Post-Processing Agent**: Validates answers, detects hallucinations, and verifies citations

Enable CrewAI orchestration in the UI for role-based multi-agent coordination.

#### Graph Explorer

The Graph Explorer provides interactive visualization of the knowledge graph:

- **Entity Search**: Find entities and their connections
- **Full Graph View**: View entire graph structure (with node limits)
- **Subgraph Exploration**: Explore neighborhood around specific entities
- **Interactive Features**: Drag nodes, zoom, pan, and view relationship details
- **Statistics**: View node type distribution and relationship counts

### Evaluation

The evaluation framework uses **DeepEval** for metrics calculation and **Confident AI** for hosted evaluation reports.

#### Metrics

The system evaluates using DeepEval generator metrics, following [DeepEval's RAG evaluation best practices](https://deepeval.com/docs/getting-started-rag):

**Generator Metrics (Answer Quality)**:
- **Hallucination Score**: Detects if the answer contains information not present in the retrieval context (lower is better, threshold: 0.5)
- **Answer Relevancy**: Measures how relevant the answer is to the query (higher is better, threshold: 0.7)
- **Faithfulness**: Evaluates if the answer is faithful to the retrieved context (higher is better, threshold: 0.7)

These metrics focus on answer quality and are automatically calculated for each test case. Results are cached to avoid redundant API calls.

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

You can customize the number of test cases:
```bash
# Run with specific number of SQuAD v2 test cases (default: 10)
python evals/run_evaluation.py --test-cases 20

# Note: DocVQA and FLEURS are available but disabled by default
# To enable them, modify the code or use the UI evaluation page
```

**Parallel Execution**:
```bash
# Run with 3 parallel workers (faster for large test suites)
python evals/run_evaluation.py --test-cases 20 --parallel 3

# Combine with automatic Confident AI upload
python evals/run_evaluation.py --test-cases 10 --parallel 3 --use-automatic-upload
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
- Use `--use-automatic-upload` flag to enable DeepEval's automatic Confident AI uploads
- Requires `CONFIDENT_API_KEY` environment variable to be set
- Results are automatically uploaded to Confident AI dashboard for visualization

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
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── .env                        # API keys, configs
├── setup/                      # Database initialization
├── evals/                      # Evaluation framework (DeepEval)
├── ingestion/                  # Multi-modal processors
├── extraction/                 # Entity/relationship extraction
├── graph/                      # Neo4j operations
├── vector/                     # Qdrant operations
├── search/                     # Hybrid search (keyword, vector, graph)
├── agents/                     # Agent classes (validation, triage, retrieval, generation, post-processing)
├── pipeline/                   # End-to-end pipelines (ingestion, query, agentic, CrewAI)
├── ui/                         # Streamlit interface
├── utils/                      # Utilities (config, errors, logging, telemetry)
├── docs/                       # Documentation
├── scripts/                    # Utility scripts (telemetry viewer, cache management, SSL fix)
└── tests/                      # Unit and integration tests
```

## Error Handling

The system includes comprehensive error handling for:
- API failures (retry with exponential backoff)
- Database connection issues
- File processing errors
- Resource limitations
- User input validation

## Telemetry and Observability

The system includes comprehensive telemetry for monitoring agent operations:

- **Operation Tracking**: Tracks all agent operations with timing, success/error rates
- **LangSmith Integration**: Automatic tracing for LangChain agent operations
- **Metrics Collection**: Structured logging with operation-level metrics
- **Export Capabilities**: Telemetry data can be exported for analysis

View telemetry in the Streamlit UI or use the CLI tool:
```bash
python scripts/view_telemetry.py
```

## Architecture Decisions

For detailed explanations of architectural choices, see [ARCHITECTURE.md](ARCHITECTURE.md). Key decisions include:

- **Agentic Pipeline**: Provides sophisticated query understanding and multi-step reasoning
- **CrewAI Integration**: Optional framework for role-based multi-agent orchestration
- **Hybrid Search (RRF)**: Combines keyword, vector, and graph search for comprehensive retrieval
- **Dual Storage**: Neo4j for structure, Qdrant for semantics - each optimized for its purpose
- **Evaluation-First**: DeepEval integration ensures quality and continuous improvement
- **Telemetry**: Comprehensive observability for production readiness

## License

MIT License

Copyright (c) 2025 Sangram Shinde

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

