"""LangSmith configuration for LangChain agent observability."""

import os
from typing import Optional
from loguru import logger

try:
    from langsmith import Client, traceable
    from langsmith.run_helpers import tracing_context
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    logger.warning("LangSmith not installed. Install with: pip install langsmith")


def configure_langsmith(api_key: Optional[str] = None, 
                       project_name: Optional[str] = None,
                       enabled: bool = True):
    """
    Configure LangSmith for observability.
    
    Args:
        api_key: LangSmith API key (defaults to LANGSMITH_API_KEY env var)
        project_name: Project name for traces (defaults to "multimodal-rag")
        enabled: Whether to enable LangSmith tracing
    """
    if not LANGSMITH_AVAILABLE:
        logger.warning("LangSmith not available. Skipping configuration.")
        return False
    
    if not enabled:
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        os.environ.pop("LANGCHAIN_API_KEY", None)
        os.environ.pop("LANGCHAIN_ENDPOINT", None)
        os.environ.pop("LANGCHAIN_PROJECT", None)
        logger.info("LangSmith tracing disabled")
        return False
    
    # Get API key
    final_api_key = api_key or os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    
    if not final_api_key or len(final_api_key) < 10:
        logger.warning(
            "LangSmith API key not found or invalid. Set LANGSMITH_API_KEY or LANGCHAIN_API_KEY env var. "
            "LangSmith tracing will be disabled."
        )
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        os.environ.pop("LANGCHAIN_API_KEY", None)
        os.environ.pop("LANGCHAIN_ENDPOINT", None)
        os.environ.pop("LANGCHAIN_PROJECT", None)
        return False
    
    # Set environment variables for LangChain
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = final_api_key
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    
    if project_name:
        os.environ["LANGCHAIN_PROJECT"] = project_name
    elif "LANGCHAIN_PROJECT" not in os.environ:
        os.environ["LANGCHAIN_PROJECT"] = "multimodal-rag"
    
    logger.info(f"✅ LangSmith configured | Project: {os.environ.get('LANGCHAIN_PROJECT')}")
    return True


def get_langsmith_client() -> Optional[Client]:
    """Get LangSmith client if available."""
    if not LANGSMITH_AVAILABLE:
        return None
    
    try:
        return Client()
    except Exception as e:
        logger.warning(f"Failed to create LangSmith client: {e}")
        return None


def trace_agent_operation(operation_name: str):
    """
    Decorator to trace agent operations with LangSmith.
    
    Usage:
        @trace_agent_operation("query_validation")
        def validate(self, query: str):
            ...
    """
    if not LANGSMITH_AVAILABLE:
        # Return no-op decorator
        def noop_decorator(func):
            return func
        return noop_decorator
    
    return traceable(name=operation_name)

