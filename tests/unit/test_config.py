"""Unit tests for configuration."""

import pytest
import os
from unittest.mock import patch
from utils.config import (
    validate_config,
    OPENAI_MODEL,
    OPENAI_VISION_MODEL,
    OPENAI_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    NEO4J_URI,
    QDRANT_URL,
    DOMAIN_TAGS
)


class TestConfig:
    """Test suite for configuration."""
    
    @patch('utils.config.OPENAI_API_KEY', 'test-key')
    def test_validate_config_success(self):
        """Test successful config validation."""
        is_valid, error = validate_config()
        assert is_valid is True
        assert error is None
    
    @patch('utils.config.OPENAI_API_KEY', None)
    def test_validate_config_missing_api_key(self):
        """Test config validation with missing API key."""
        is_valid, error = validate_config()
        assert is_valid is False
        assert "OPENAI_API_KEY" in error
    
    def test_default_model_values(self):
        """Test that default model values are set."""
        # These should have defaults even if env vars aren't set
        assert OPENAI_MODEL is not None
        assert OPENAI_VISION_MODEL is not None
        assert OPENAI_EMBEDDING_MODEL is not None
    
    def test_default_embedding_dimension(self):
        """Test default embedding dimension."""
        assert isinstance(EMBEDDING_DIMENSION, int)
        assert EMBEDDING_DIMENSION > 0
    
    def test_default_neo4j_uri(self):
        """Test default Neo4j URI."""
        assert NEO4J_URI is not None
        assert "bolt://" in NEO4J_URI or "neo4j://" in NEO4J_URI
    
    def test_default_qdrant_url(self):
        """Test default Qdrant URL."""
        assert QDRANT_URL is not None
        assert "http://" in QDRANT_URL or "https://" in QDRANT_URL
    
    def test_domain_tags_defined(self):
        """Test that domain tags are defined."""
        assert len(DOMAIN_TAGS) > 0
        assert isinstance(DOMAIN_TAGS, list)
        assert all(isinstance(tag, str) for tag in DOMAIN_TAGS)

