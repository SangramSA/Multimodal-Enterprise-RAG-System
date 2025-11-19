# Telemetry and Observability

This document describes the telemetry and observability features for the agentic query pipeline.

## Framework Support

The system supports two orchestration modes:
- **Custom Orchestration**: Manual agent coordination (default)
- **CrewAI Orchestration**: Framework-based multi-agent orchestration (optional)

Both modes are fully instrumented with telemetry.

## Overview

The system includes comprehensive telemetry to track agent operations, performance metrics, and debugging information. This helps with:

- **Performance Monitoring**: Track execution times and identify bottlenecks
- **Error Debugging**: See detailed error information and stack traces
- **Usage Analytics**: Understand which agents are used most and how
- **Quality Assurance**: Monitor success rates and confidence scores

## Libraries Used

### 1. LangSmith (LangChain Observability)

**LangSmith** is LangChain's official observability platform. It provides:

- Automatic tracing of LangChain agent operations
- Token usage tracking
- Cost estimation
- Performance metrics
- Debugging tools

**Setup:**

1. Get a LangSmith API key from [https://smith.langchain.com](https://smith.langchain.com)
2. Set environment variable:
   ```bash
   export LANGSMITH_API_KEY=your_api_key_here
   # OR
   export LANGCHAIN_API_KEY=your_api_key_here
   ```
3. Optionally set project name:
   ```bash
   export LANGCHAIN_PROJECT=multimodal-rag
   ```

**Features:**
- Automatic tracing of `RetrievalOrchestrationAgent` operations
- Tool usage tracking
- Token consumption monitoring
- Cost analysis

### 2. Custom Telemetry Module

A custom telemetry system (`utils/telemetry.py`) tracks:

- Agent operation start/end times
- Input/output data (with size limits for privacy)
- Error information
- Metadata (confidence scores, iteration counts, etc.)

**Features:**
- Structured logging to JSONL files
- Real-time metrics collection
- Agent statistics aggregation
- Export capabilities

## Telemetry Data Collected

### Per-Agent Metrics

Each agent operation tracks:

- **Agent Name**: Which agent performed the operation
- **Operation**: What operation was performed (e.g., "validate", "triage", "retrieve")
- **Duration**: Execution time in milliseconds
- **Input Data**: Query, parameters, etc. (truncated for privacy)
- **Output Data**: Results, confidence scores, etc.
- **Error Information**: Error messages if operation failed
- **Metadata**: Additional context (iterations, tool usage, etc.)

### Pipeline-Level Metrics

The `AgenticQueryPipeline` tracks:

- Total execution time
- Number of iterations
- Search strategies used
- Methods used (keyword, vector, graph, hybrid)
- Confidence scores
- Hallucination scores
- Citation verification results

## Viewing Telemetry

### 1. Streamlit UI

Navigate to the **Telemetry** page in the Streamlit UI to see:

- Overall statistics (total operations, success rate, errors)
- Per-agent performance metrics
- Recent operations with details
- Export functionality

### 2. Command Line

Use the `view_telemetry.py` script:

```bash
# View live metrics from current session
python scripts/view_telemetry.py --live

# View metrics from log file
python scripts/view_telemetry.py --log-file logs/telemetry.jsonl

# Export metrics to JSON
python scripts/view_telemetry.py --live --export metrics.json
```

### 3. Log Files

Telemetry is automatically written to:

- `logs/telemetry.jsonl` - JSONL format, one metric per line
- `logs/app.log` - Standard application logs with telemetry markers

## LangSmith Dashboard

Access the LangSmith dashboard at [https://smith.langchain.com](https://smith.langchain.com) to see:

- Traces of all LangChain agent operations
- Token usage and costs
- Performance metrics
- Error analysis
- Tool usage patterns

## Privacy and Data Limits

The telemetry system includes privacy protections:

- **Input Data**: Only first argument if it's a short string (< 500 chars)
- **Output Data**: Only keys and small values (< 200 chars)
- **Large Objects**: Converted to string representation
- **Sensitive Data**: Not logged (API keys, tokens, etc.)

## Configuration

### Enable/Disable LangSmith

```python
from utils.langsmith_config import configure_langsmith

# Enable
configure_langsmith(enabled=True)

# Disable
configure_langsmith(enabled=False)
```

### Custom Telemetry Settings

The telemetry collector can be configured:

```python
from utils.telemetry import TelemetryCollector

collector = TelemetryCollector(log_dir=Path("custom_logs"))
```

## Example Metrics

### Query Validation Agent

```json
{
  "agent_name": "QueryValidationAgent",
  "operation": "validate",
  "duration_ms": 45.2,
  "input_data": {
    "query_length": 25
  },
  "output_data": {
    "is_valid": true,
    "complexity": "simple",
    "intent": "factual"
  },
  "metadata": {
    "function": "validate"
  }
}
```

### Retrieval Orchestration Agent

```json
{
  "agent_name": "RetrievalOrchestrationAgent",
  "operation": "retrieve",
  "duration_ms": 1234.5,
  "input_data": {
    "query_length": 30
  },
  "output_data": {
    "methods_used": ["hybrid", "vector"],
    "confidence": 0.85
  },
  "metadata": {
    "num_results": 10
  }
}
```

## Best Practices

1. **Monitor Regularly**: Check telemetry dashboard for anomalies
2. **Set Alerts**: Configure alerts for high error rates or slow operations
3. **Export Periodically**: Export metrics for long-term analysis
4. **Review LangSmith**: Use LangSmith for detailed LangChain agent debugging
5. **Privacy**: Be mindful of what data is logged (already limited by default)

## Troubleshooting

### LangSmith Not Working

1. Check API key is set: `echo $LANGSMITH_API_KEY`
2. Verify project name: `echo $LANGCHAIN_PROJECT`
3. Check logs for LangSmith errors
4. Ensure `langsmith` package is installed: `pip install langsmith`

### No Telemetry Data

1. Check log directory exists: `ls logs/`
2. Verify file permissions
3. Check if agents are actually being called
4. Review application logs for errors

### High Memory Usage

- Telemetry data is written to disk immediately
- Old metrics are not kept in memory indefinitely
- Use export and clear if needed: `collector.clear_metrics()`

## Future Enhancements

Potential improvements:

- **OpenTelemetry Integration**: Full OpenTelemetry support for industry-standard observability
- **Prometheus Metrics**: Export metrics in Prometheus format
- **Grafana Dashboards**: Pre-built dashboards for visualization
- **Real-time Streaming**: Stream metrics to external systems
- **Anomaly Detection**: Automatic detection of performance issues

