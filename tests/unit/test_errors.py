"""Unit tests for error handling."""

import pytest
import time
from unittest.mock import patch, Mock
from utils.errors import (
    RAGError,
    APIError,
    DatabaseError,
    ProcessingError,
    ValidationError,
    FileError,
    handle_error,
    retry_with_backoff
)


class TestErrorClasses:
    """Test suite for error classes."""
    
    def test_rag_error_base(self):
        """Test base RAGError class."""
        error = RAGError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_api_error_with_status_code(self):
        """Test APIError with status code."""
        error = APIError("API failed", status_code=500)
        assert error.status_code == 500
        assert str(error) == "API failed"
    
    def test_api_error_with_retry_after(self):
        """Test APIError with retry_after."""
        error = APIError("Rate limited", status_code=429, retry_after=60)
        assert error.retry_after == 60
    
    def test_database_error(self):
        """Test DatabaseError."""
        error = DatabaseError("Database connection failed")
        assert isinstance(error, RAGError)
        assert str(error) == "Database connection failed"
    
    def test_processing_error(self):
        """Test ProcessingError."""
        error = ProcessingError("Processing failed")
        assert isinstance(error, RAGError)
    
    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Invalid input")
        assert isinstance(error, RAGError)
    
    def test_file_error(self):
        """Test FileError."""
        error = FileError("File not found")
        assert isinstance(error, RAGError)


class TestErrorHandling:
    """Test suite for error handling functions."""
    
    def test_handle_error_api_error_retryable(self):
        """Test handling APIError with retryable status code."""
        error = APIError("Server error", status_code=500)
        error_info = handle_error(error)
        
        assert error_info["error_type"] == "APIError"
        assert error_info["retryable"] is True
        assert error_info["error_message"] == "Server error"
    
    def test_handle_error_api_error_not_retryable(self):
        """Test handling APIError with non-retryable status code."""
        error = APIError("Bad request", status_code=400)
        error_info = handle_error(error)
        
        assert error_info["retryable"] is False
    
    def test_handle_error_database_error(self):
        """Test handling DatabaseError."""
        error = DatabaseError("Connection failed")
        error_info = handle_error(error)
        
        assert error_info["error_type"] == "DatabaseError"
        assert error_info["retryable"] is True
    
    def test_handle_error_validation_error(self):
        """Test handling ValidationError."""
        error = ValidationError("Invalid input")
        error_info = handle_error(error)
        
        assert error_info["error_type"] == "ValidationError"
        assert error_info["retryable"] is False
    
    def test_handle_error_with_context(self):
        """Test error handling with context."""
        error = APIError("API failed")
        context = {"file_id": "file1", "operation": "extract"}
        error_info = handle_error(error, context)
        
        assert error_info["context"] == context


class TestRetryWithBackoff:
    """Test suite for retry_with_backoff."""
    
    def test_retry_success_first_attempt(self):
        """Test retry succeeds on first attempt."""
        func = Mock(return_value="success")
        
        result = retry_with_backoff(func, max_retries=3)
        
        assert result == "success"
        assert func.call_count == 1
    
    @patch('time.sleep')
    def test_retry_succeeds_after_failure(self, mock_sleep):
        """Test retry succeeds after initial failure."""
        func = Mock(side_effect=[Exception("Fail"), "success"])
        
        result = retry_with_backoff(func, max_retries=3)
        
        assert result == "success"
        assert func.call_count == 2
        assert mock_sleep.called
    
    @patch('time.sleep')
    def test_retry_exhausts_attempts(self, mock_sleep):
        """Test retry exhausts all attempts."""
        func = Mock(side_effect=Exception("Always fails"))
        
        with pytest.raises(Exception, match="Always fails"):
            retry_with_backoff(func, max_retries=3)
        
        assert func.call_count == 3
        assert mock_sleep.call_count == 2  # Sleep between attempts
    
    @patch('time.sleep')
    def test_retry_exponential_backoff(self, mock_sleep):
        """Test that retry uses exponential backoff."""
        func = Mock(side_effect=[Exception("Fail"), Exception("Fail"), "success"])
        
        retry_with_backoff(func, max_retries=3, base_delay=1.0)
        
        # Verify sleep was called with increasing delays
        assert mock_sleep.call_count == 2
        # First delay should be ~1s, second ~2s (with some randomness)
        call_args = [call[0][0] for call in mock_sleep.call_args_list]
        assert call_args[1] > call_args[0]  # Second delay > first

