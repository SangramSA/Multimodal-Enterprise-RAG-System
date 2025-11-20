"""End-to-end pipelines for ingestion and query processing.

This module contains pipeline orchestrators:
- IngestionPipeline: Multi-modal file ingestion pipeline
- QueryPipeline: Standard query processing pipeline
- AgenticQueryPipeline: Agentic query processing with multi-agent orchestration
- CrewAIOrchestrator: CrewAI-based multi-agent orchestration
- InputValidator: Input validation utilities
"""

__all__ = [
    "IngestionPipeline",
    "QueryPipeline",
    "AgenticQueryPipeline",
    "CrewAIOrchestrator",
    "InputValidator",
]

# Lazy imports to avoid circular dependencies
# Import directly from submodules when needed:
# from pipeline.ingestion_pipeline import IngestionPipeline
