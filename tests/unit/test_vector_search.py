"""Unit tests for VectorSearch."""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from search.vector_search import VectorSearch
from vector.vector_store import VectorStore


class TestVectorSearch:
    """Test suite for VectorSearch."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock VectorStore."""
        return Mock(spec=VectorStore)
    
    @pytest.fixture
    def vector_search(self, mock_vector_store):
        """Create VectorSearch instance."""
        return VectorSearch(mock_vector_store)
    
    def test_init(self, mock_vector_store):
        """Test VectorSearch initialization."""
        search = VectorSearch(mock_vector_store)
        assert search.vector_store == mock_vector_store
    
    def test_search_success(self, vector_search, mock_vector_store):
        """Test successful vector search."""
        mock_results = [
            {
                "chunk_id": "chunk1",
                "content": "test content",
                "score": 0.95,
                "metadata": {"file_name": "test.pdf"}
            },
            {
                "chunk_id": "chunk2",
                "content": "another content",
                "score": 0.85,
                "metadata": {"file_name": "test2.pdf"}
            }
        ]
        mock_vector_store.search.return_value = mock_results
        
        results = vector_search.search("test query", limit=10)
        
        assert len(results) == 2
        assert results[0]["chunk_id"] == "chunk1"
        assert results[0]["score"] == 0.95
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            limit=10,
            filters=None,
            score_threshold=0.0
        )
    
    def test_search_with_filters(self, vector_search, mock_vector_store):
        """Test vector search with filters."""
        mock_vector_store.search.return_value = []
        filters = {"modality": "text", "domain_tags": ["technical"]}
        
        vector_search.search("test query", limit=10, filters=filters)
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            limit=10,
            filters=filters,
            score_threshold=0.0
        )
    
    def test_search_with_score_threshold(self, vector_search, mock_vector_store):
        """Test vector search with score threshold."""
        mock_vector_store.search.return_value = []
        
        vector_search.search("test query", limit=10, score_threshold=0.7)
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            limit=10,
            filters=None,
            score_threshold=0.7
        )
    
    def test_search_empty_results(self, vector_search, mock_vector_store):
        """Test vector search with no results."""
        mock_vector_store.search.return_value = []
        
        results = vector_search.search("nonexistent query", limit=10)
        
        assert results == []
    
    def test_search_all_parameters(self, vector_search, mock_vector_store):
        """Test vector search with all parameters."""
        mock_vector_store.search.return_value = []
        filters = {"modality": "image"}
        
        vector_search.search(
            query="test query",
            limit=5,
            filters=filters,
            score_threshold=0.8
        )
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            limit=5,
            filters=filters,
            score_threshold=0.8
        )
    
    def test_search_preserves_scores(self, vector_search, mock_vector_store):
        """Test that search preserves similarity scores from vector store."""
        mock_results = [
            {"chunk_id": "chunk1", "content": "test", "score": 0.95},
            {"chunk_id": "chunk2", "content": "test2", "score": 0.85}
        ]
        mock_vector_store.search.return_value = mock_results
        
        results = vector_search.search("test query", limit=10)
        
        assert results[0]["score"] == 0.95
        assert results[1]["score"] == 0.85
    
    def test_search_error_handling(self, vector_search, mock_vector_store):
        """Test vector search error handling."""
        mock_vector_store.search.side_effect = Exception("Vector store error")
        
        with pytest.raises(Exception):
            vector_search.search("test query", limit=10)

