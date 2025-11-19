# CrewAI Usage Guide

This guide explains how to use the CrewAI multi-agent orchestration framework in the Multimodal Enterprise RAG system.

## Overview

CrewAI is an optional framework for multi-agent orchestration that provides:
- **Role-based agents**: Each agent has a specific role, goal, and backstory
- **Task management**: Structured task definitions with expected outputs
- **Crew orchestration**: Automatic coordination between agents
- **Enhanced observability**: Built-in logging and debugging

The system supports two orchestration modes:
1. **Custom Orchestration** (default): Manual agent coordination
2. **CrewAI Orchestration** (optional): Framework-based multi-agent system

Both modes provide the same functionality, but CrewAI offers better structure and observability for complex multi-agent workflows.

## Prerequisites

1. **Install CrewAI** (if not already installed):
   ```bash
   pip install crewai crewai-tools
   ```

2. **Verify installation**:
   ```bash
   python -c "import crewai; print('CrewAI version:', crewai.__version__)"
   ```

## Usage Methods

### Method 1: Streamlit UI (Easiest)

1. **Start the Streamlit app**:
   ```bash
   streamlit run ui/app.py
   ```

2. **Navigate to the Query page**

3. **Enable CrewAI (optional)**:
   - Check the "Use CrewAI Orchestration (Experimental)" checkbox

4. **Enter your query and submit**

The system will use CrewAI to orchestrate the multi-agent pipeline.

### Method 2: Programmatic Usage

#### Basic Usage

```python
from pipeline.agentic_query_pipeline import AgenticQueryPipeline
from agents.retrieval_orchestration_agent import RetrievalOrchestrationAgent
from search.hybrid_search import HybridSearch
from search.graph_search import GraphSearch
from search.keyword_search import KeywordSearch
from search.vector_search import VectorSearch

# Initialize search components
hybrid_search = HybridSearch(...)
graph_search = GraphSearch(...)
keyword_search = KeywordSearch(...)
vector_search = VectorSearch(...)

# Create retrieval agent
retrieval_agent = RetrievalOrchestrationAgent(
    hybrid_search=hybrid_search,
    graph_search=graph_search,
    keyword_search=keyword_search,
    vector_search=vector_search
)

# Initialize pipeline with CrewAI enabled
pipeline = AgenticQueryPipeline(
    retrieval_agent=retrieval_agent,
    use_crewai=True  # Enable CrewAI orchestration
)

# Process a query
result = pipeline.process(
    query="What is the relationship between Project Apollo and Aparavi?",
    max_iterations=3
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Sources: {len(result['sources'])} documents")
```

### Method 3: Direct CrewAI Orchestrator

For advanced usage, you can use the CrewAI orchestrator directly:

```python
from pipeline.crewai_orchestrator import CrewAIOrchestrator
from agents.retrieval_orchestration_agent import RetrievalOrchestrationAgent

# Initialize orchestrator
orchestrator = CrewAIOrchestrator(retrieval_agent)

# Execute pipeline
result = orchestrator.execute_pipeline(
    query="What documents mention NASA?",
    max_iterations=3
)
```

## CrewAI Architecture

### Agents

The system uses 6 CrewAI agents:

1. **Query Security Validator** (`QueryValidationAgent`)
   - Role: Validates and sanitizes queries
   - Tools: `validate_query_tool`

2. **Query Classifier** (`QueryTriageAgent`)
   - Role: Classifies queries and selects search strategy
   - Tools: `triage_query_tool`

3. **Information Retrieval Specialist** (`RetrievalOrchestrationAgent`)
   - Role: Retrieves relevant documents
   - Tools: `retrieve_documents_tool`

4. **Answer Synthesis Expert** (`AnswerGenerationAgent`)
   - Role: Generates comprehensive answers
   - Tools: `generate_answer_tool`

5. **Answer Quality Validator** (`PostProcessingAgent`)
   - Role: Validates answer quality
   - Tools: `validate_answer_tool`

### Tasks

Each agent has corresponding tasks defined in `pipeline/crewai_tasks.py`:

- `create_validation_task()`: Query validation
- `create_triage_task()`: Query classification
- `create_retrieval_task()`: Document retrieval
- `create_generation_task()`: Answer generation
- `create_qa_task()`: Quality assurance

### Tools

Agent methods are wrapped as CrewAI tools in `agents/crewai_tools.py`:

- `validate_query_tool`: Wraps `QueryValidationAgent.validate()`
- `triage_query_tool`: Wraps `QueryTriageAgent.triage()`
- `retrieve_documents_tool`: Wraps `RetrievalOrchestrationAgent.retrieve()`
- `generate_answer_tool`: Wraps `AnswerGenerationAgent.generate()`
- `validate_answer_tool`: Wraps `PostProcessingAgent.process()`

## Current Implementation Status

**Note**: The current implementation uses a **hybrid approach**:

- **Agent Creation**: CrewAI agents are created with roles, goals, and backstories
- **Tool Integration**: Agent methods are wrapped as CrewAI tools
- **Execution**: Currently uses direct agent calls for better performance and reliability
- **Future Enhancement**: Full CrewAI task execution can be enabled for more structured workflows

This hybrid approach provides:
- ✅ CrewAI structure and organization
- ✅ Better observability and debugging
- ✅ Reliable execution with direct agent calls
- ✅ Easy migration to full CrewAI execution later

## Configuration

### Environment Variables

No special environment variables are required for CrewAI. The system will:
- Automatically detect if CrewAI is installed
- Fall back to custom orchestration if CrewAI is unavailable
- Log warnings if CrewAI is requested but not installed

### LLM Configuration

CrewAI agents use the same LLM configuration as the rest of the system:
- Model: `OPENAI_MODEL` (default: `gpt-4o`)
- API Key: `OPENAI_API_KEY`
- Temperature: 0.1 (for deterministic responses)

## Differences: CrewAI vs Custom Orchestration

| Feature | Custom Orchestration | CrewAI Orchestration |
|---------|---------------------|---------------------|
| **Structure** | Manual coordination | Role-based agents |
| **Observability** | Custom telemetry | CrewAI + telemetry |
| **Debugging** | Log-based | Enhanced with CrewAI tools |
| **Flexibility** | Full control | Framework-guided |
| **Performance** | Direct calls | Slightly more overhead |
| **Complexity** | Lower | Higher (but more organized) |

## Troubleshooting

### CrewAI Not Available

If you see: `"CrewAI not available. Using custom orchestration."`

**Solution**: Install CrewAI:
```bash
pip install crewai crewai-tools
```

### Import Errors

If you see import errors for CrewAI modules:

**Solution**: Ensure CrewAI is installed in the correct virtual environment:
```bash
source .venv/bin/activate  # or your venv path
pip install crewai crewai-tools
```

### Performance Issues

If CrewAI seems slower:

**Solution**: The current implementation uses direct agent calls for performance. If you need full CrewAI execution, you can modify `pipeline/crewai_orchestrator.py` to use CrewAI tasks instead of direct calls.

## Example Workflow

Here's a complete example:

```python
from pipeline.agentic_query_pipeline import AgenticQueryPipeline
from agents.retrieval_orchestration_agent import RetrievalOrchestrationAgent
# ... other imports ...

# 1. Initialize components
retrieval_agent = RetrievalOrchestrationAgent(...)

# 2. Create pipeline with CrewAI
pipeline = AgenticQueryPipeline(
    retrieval_agent=retrieval_agent,
    use_crewai=True
)

# 3. Process query
result = pipeline.process(
    query="What is the relationship between Project Apollo and Aparavi?",
    max_iterations=3
)

# 4. Access results
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Sources: {len(result['sources'])} documents")
print(f"Iterations: {result['metadata']['iterations']}")
print(f"Methods used: {result['metadata']['methods_used']}")
```

## Next Steps

1. **Try it in the UI**: Enable CrewAI in the Streamlit interface
2. **Monitor telemetry**: Check the Telemetry page to see agent performance
3. **Experiment**: Compare CrewAI vs custom orchestration for your use cases
4. **Customize**: Modify agent roles/goals in `agents/crewai_agents.py` if needed

## Additional Resources

- [CrewAI Documentation](https://docs.crewai.com/)
- [CrewAI GitHub](https://github.com/joaomdmoura/crewAI)
- System Architecture: See `ARCHITECTURE.md`
- Telemetry: See `docs/TELEMETRY.md`

