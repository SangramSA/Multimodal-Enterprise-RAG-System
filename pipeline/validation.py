"""Input validation for queries and files."""

from pathlib import Path
from typing import Dict, Any, Optional
import re
from loguru import logger

from utils.config import MAX_FILE_SIZE_MB
from utils.errors import ValidationError


class InputValidator:
    """Validate user inputs."""
    
    MAX_QUERY_LENGTH = 1000
    MIN_QUERY_LENGTH = 1
    ALLOWED_FILE_TYPES = [".pdf", ".txt", ".jpg", ".jpeg", ".png", ".mp3", ".wav"]
    
    # Patterns to detect potentially malicious inputs
    SQL_INJECTION_PATTERNS = [
        r"(?i)(union|select|insert|update|delete|drop|create|alter|exec|execute)",
        r"(?i)(--|;|/\*|\*/)",
        r"(?i)(or|and)\s+\d+\s*=\s*\d+"
    ]
    
    SCRIPT_TAG_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*="
    ]
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate a query string.
        
        Returns:
            Validation result with is_valid flag and optional error message
        """
        if not query or not isinstance(query, str):
            raise ValidationError("Query must be a non-empty string")
        
        query = query.strip()
        
        # Check length
        if len(query) < self.MIN_QUERY_LENGTH:
            raise ValidationError(f"Query too short (minimum {self.MIN_QUERY_LENGTH} characters)")
        
        if len(query) > self.MAX_QUERY_LENGTH:
            raise ValidationError(f"Query too long (maximum {self.MAX_QUERY_LENGTH} characters)")
        
        # Check for SQL injection patterns
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, query):
                logger.warning(f"Potential SQL injection detected in query")
                raise ValidationError("Query contains potentially unsafe patterns")
        
        # Check for script tags
        for pattern in self.SCRIPT_TAG_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE | re.DOTALL):
                logger.warning(f"Potential XSS detected in query")
                raise ValidationError("Query contains potentially unsafe content")
        
        return {
            "is_valid": True,
            "sanitized_query": query
        }
    
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate a file before processing.
        
        Returns:
            Validation result with is_valid flag and file metadata
        """
        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")
        
        # Check file type
        if file_path.suffix.lower() not in self.ALLOWED_FILE_TYPES:
            raise ValidationError(
                f"Unsupported file type: {file_path.suffix}. "
                f"Allowed types: {', '.join(self.ALLOWED_FILE_TYPES)}"
            )
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise ValidationError(
                f"File too large: {file_size_mb:.2f}MB. "
                f"Maximum size: {MAX_FILE_SIZE_MB}MB"
            )
        
        # Check if file is readable
        try:
            with open(file_path, "rb") as f:
                f.read(1)  # Try to read first byte
        except Exception as e:
            raise ValidationError(f"File is not readable: {e}")
        
        return {
            "is_valid": True,
            "file_path": file_path,
            "file_size_mb": file_size_mb,
            "file_type": file_path.suffix.lower()
        }
    
    def sanitize_query(self, query: str) -> str:
        """Sanitize query by removing potentially dangerous characters."""
        # Remove null bytes
        query = query.replace("\x00", "")
        
        # Remove control characters except newlines and tabs
        query = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", query)
        
        return query.strip()

