"""Base class for all agents in the agentic query pipeline."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from loguru import logger
from contextlib import contextmanager

from utils.errors import APIError, ProcessingError
from utils.telemetry import get_telemetry_collector
from utils.langsmith_config import trace_agent_operation


class BaseAgent(ABC):
    """Base class for all agents with common functionality."""
    
    def __init__(self, name: str):
        """
        Initialize base agent.
        
        Args:
            name: Name of the agent for logging
        """
        self.name = name
        self.logger = logger.bind(agent=name)
        self.telemetry = get_telemetry_collector()
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Main processing method to be implemented by subclasses.
        
        Returns:
            Dictionary with processing results
        """
        pass
    
    def log_info(self, message: str):
        """Log info message with agent context."""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log warning message with agent context."""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message with agent context."""
        self.logger.error(message)
    
    def log_debug(self, message: str):
        """Log debug message with agent context."""
        self.logger.debug(message)
    
    def handle_error(self, error: Exception, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Handle errors consistently across agents.
        
        Args:
            error: Exception that occurred
            context: Optional context about where error occurred
        
        Returns:
            Error result dictionary
        """
        error_msg = f"{self.name} error"
        if context:
            error_msg += f" in {context}"
        error_msg += f": {str(error)}"
        
        self.log_error(error_msg)
        
        return {
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "context": context
        }
    
    def validate_input(self, input_data: Any, input_type: type, input_name: str) -> bool:
        """
        Validate input data type.
        
        Args:
            input_data: Data to validate
            input_type: Expected type
            input_name: Name of input for error messages
        
        Returns:
            True if valid, raises ValueError if not
        """
        if not isinstance(input_data, input_type):
            raise ValueError(f"{input_name} must be of type {input_type.__name__}, got {type(input_data).__name__}")
        return True
    
    @contextmanager
    def track_operation(self, operation: str, input_data: Optional[Dict[str, Any]] = None,
                      metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager to track an operation with telemetry.
        
        Usage:
            with self.track_operation("validate", input_data={"query": query}):
                result = self.validate(query)
        """
        operation_id = self.telemetry.start_operation(
            agent_name=self.name,
            operation=operation,
            input_data=input_data,
            metadata=metadata
        )
        
        try:
            yield operation_id
            self.telemetry.end_operation(operation_id)
        except Exception as e:
            self.telemetry.end_operation(operation_id, error=str(e))
            raise

