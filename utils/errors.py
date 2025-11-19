"""Custom error classes and error handling utilities."""

from typing import Optional, Dict, Any
from loguru import logger


class RAGError(Exception):
    """Base exception for RAG system errors."""
    pass


class APIError(RAGError):
    """Exception for API-related errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, retry_after: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class DatabaseError(RAGError):
    """Exception for database-related errors."""
    pass


class ProcessingError(RAGError):
    """Exception for data processing errors."""
    pass


class ValidationError(RAGError):
    """Exception for input validation errors."""
    pass


class FileError(RAGError):
    """Exception for file-related errors."""
    pass


def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Handle errors gracefully and return error information."""
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context or {}
    }
    
    # Log the error
    logger.error(f"Error occurred: {error_info}")
    
    # Add specific handling based on error type
    if isinstance(error, APIError):
        error_info["retryable"] = error.status_code in [429, 500, 502, 503, 504]
        if error.retry_after:
            error_info["retry_after"] = error.retry_after
    elif isinstance(error, DatabaseError):
        error_info["retryable"] = True
    elif isinstance(error, ValidationError):
        error_info["retryable"] = False
    
    return error_info


def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """Retry a function with exponential backoff."""
    import time
    import random
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
            time.sleep(delay)
    
    raise Exception("Max retries exceeded")

