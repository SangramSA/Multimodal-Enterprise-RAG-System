"""Unit tests for validation module."""

import pytest
from pathlib import Path
from pipeline.validation import InputValidator
from utils.errors import ValidationError


class TestInputValidator:
    """Test suite for InputValidator."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = InputValidator()
        assert validator.MAX_QUERY_LENGTH == 1000
        assert validator.MIN_QUERY_LENGTH == 1
        assert len(validator.ALLOWED_FILE_TYPES) > 0
    
    def test_validate_query_success(self):
        """Test successful query validation."""
        validator = InputValidator()
        result = validator.validate_query("What is machine learning?")
        
        assert result["is_valid"] is True
        assert "sanitized_query" in result
    
    def test_validate_query_empty(self):
        """Test validation of empty query."""
        validator = InputValidator()
        with pytest.raises(ValidationError, match="Query must be a non-empty string"):
            validator.validate_query("")
    
    def test_validate_query_whitespace_only(self):
        """Test validation of whitespace-only query."""
        validator = InputValidator()
        with pytest.raises(ValidationError, match="Query too short"):
            validator.validate_query("   \n\t  ")
    
    def test_validate_query_too_long(self):
        """Test validation of query that's too long."""
        validator = InputValidator()
        long_query = "a" * 1001
        with pytest.raises(ValidationError, match="Query too long"):
            validator.validate_query(long_query)
    
    def test_validate_query_sql_injection_detection(self):
        """Test SQL injection pattern detection."""
        validator = InputValidator()
        malicious_query = "SELECT * FROM users"
        
        with pytest.raises(ValidationError, match="potentially unsafe"):
            validator.validate_query(malicious_query)
    
    def test_validate_query_script_tag_detection(self):
        """Test XSS script tag detection."""
        validator = InputValidator()
        malicious_query = "<script>alert('xss')</script>"
        
        with pytest.raises(ValidationError, match="potentially unsafe"):
            validator.validate_query(malicious_query)
    
    def test_validate_file_success(self, tmp_path):
        """Test successful file validation."""
        validator = InputValidator()
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        result = validator.validate_file(test_file)
        
        assert result["is_valid"] is True
        assert result["file_path"] == test_file
        assert "file_size_mb" in result
    
    def test_validate_file_not_exists(self, tmp_path):
        """Test validation of non-existent file."""
        validator = InputValidator()
        non_existent = tmp_path / "nonexistent.txt"
        
        with pytest.raises(ValidationError, match="File not found"):
            validator.validate_file(non_existent)
    
    def test_validate_file_unsupported_type(self, tmp_path):
        """Test validation of unsupported file type."""
        validator = InputValidator()
        test_file = tmp_path / "test.doc"
        test_file.write_text("test")
        
        with pytest.raises(ValidationError, match="Unsupported file type"):
            validator.validate_file(test_file)
    
    def test_validate_file_too_large(self, tmp_path):
        """Test validation of file that's too large."""
        validator = InputValidator()
        test_file = tmp_path / "test.txt"
        # Create a file larger than MAX_FILE_SIZE_MB (100MB default)
        large_content = "x" * (101 * 1024 * 1024)
        test_file.write_text(large_content)
        
        with pytest.raises(ValidationError, match="File too large"):
            validator.validate_file(test_file)
    
    def test_sanitize_query(self):
        """Test query sanitization."""
        validator = InputValidator()
        query = "Test\x00query\nwith\ttabs"
        sanitized = validator.sanitize_query(query)
        
        assert "\x00" not in sanitized
        assert "\n" in sanitized  # Newlines should be preserved
        assert "\t" in sanitized  # Tabs should be preserved
    
    def test_sanitize_query_removes_control_chars(self):
        """Test that control characters are removed."""
        validator = InputValidator()
        query = "Test\x01\x02\x03query"
        sanitized = validator.sanitize_query(query)
        
        assert "\x01" not in sanitized
        assert "\x02" not in sanitized
        assert "\x03" not in sanitized

