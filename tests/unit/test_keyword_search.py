"""Unit tests for keyword search."""

import pytest
from unittest.mock import Mock, patch
from search.keyword_search import KeywordSearch


class TestKeywordSearch:
    """Test suite for KeywordSearch."""
    
    def test_keyword_search_initialization(self):
        """Test keyword search initialization."""
        mock_vector_store = Mock()
        search = KeywordSearch(mock_vector_store)
        
        assert search.vector_store == mock_vector_store
        assert search.index is None
    
    def test_tokenize(self):
        """Test text tokenization."""
        mock_vector_store = Mock()
        search = KeywordSearch(mock_vector_store)
        
        tokens = search._tokenize("This is a test")
        assert tokens == ["this", "is", "a", "test"]
    
    def test_tokenize_lowercase(self):
        """Test that tokenization converts to lowercase."""
        mock_vector_store = Mock()
        search = KeywordSearch(mock_vector_store)
        
        tokens = search._tokenize("UPPERCASE TEXT")
        assert tokens == ["uppercase", "text"]
    
    def test_search_success(self):
        """Test successful keyword search."""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = [
            {"content": "This is about machine learning", "chunk_id": "chunk1"},
            {"content": "This is about deep learning", "chunk_id": "chunk2"},
            {"content": "This is about cooking", "chunk_id": "chunk3"}
        ]
        
        search = KeywordSearch(mock_vector_store)
        results = search.search("machine learning", limit=5)
        
        assert len(results) > 0
        assert all("keyword_score" in r for r in results)
        assert all("combined_score" in r for r in results)
        # Results should be sorted by score
        scores = [r["keyword_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_search_no_results(self):
        """Test search with no matching results."""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = []
        
        search = KeywordSearch(mock_vector_store)
        results = search.search("query", limit=10)
        
        assert len(results) == 0
    
    def test_search_filters_zero_scores(self):
        """Test that documents with zero scores are filtered."""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = [
            {"content": "Unrelated content", "chunk_id": "chunk1"},
            {"content": "Relevant content about query", "chunk_id": "chunk2"}
        ]
        
        search = KeywordSearch(mock_vector_store)
        results = search.search("query", limit=10)
        
        # Only documents with score > 0 should be included
        assert all(r["keyword_score"] > 0 for r in results)
    
    def test_search_respects_limit(self):
        """Test that search respects the limit parameter."""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = [
            {"content": f"Content {i} about query", "chunk_id": f"chunk{i}"}
            for i in range(20)
        ]
        
        search = KeywordSearch(mock_vector_store)
        results = search.search("query", limit=5)
        
        assert len(results) <= 5
    
    def test_filter_by_metadata_modality(self):
        """Test filtering by modality metadata."""
        mock_vector_store = Mock()
        search = KeywordSearch(mock_vector_store)
        
        results = [
            {"content": "Text", "metadata": {"modality": "text"}},
            {"content": "Image", "metadata": {"modality": "image"}},
            {"content": "Audio", "metadata": {"modality": "audio"}}
        ]
        
        filtered = search.filter_by_metadata(results, {"modality": "text"})
        
        assert len(filtered) == 1
        assert filtered[0]["metadata"]["modality"] == "text"
    
    def test_filter_by_metadata_domain_tags(self):
        """Test filtering by domain tags."""
        mock_vector_store = Mock()
        search = KeywordSearch(mock_vector_store)
        
        results = [
            {"content": "Tech", "metadata": {"domain_tags": ["technical"]}},
            {"content": "Finance", "metadata": {"domain_tags": ["finance"]}},
            {"content": "Legal", "metadata": {"domain_tags": ["legal"]}}
        ]
        
        filtered = search.filter_by_metadata(results, {"domain_tags": ["technical"]})
        
        assert len(filtered) == 1
        assert "technical" in filtered[0]["metadata"]["domain_tags"]
    
    def test_filter_by_metadata_multiple_tags(self):
        """Test filtering with multiple domain tags."""
        mock_vector_store = Mock()
        search = KeywordSearch(mock_vector_store)
        
        results = [
            {"content": "Tech", "metadata": {"domain_tags": ["technical"]}},
            {"content": "Finance", "metadata": {"domain_tags": ["finance", "technical"]}},
            {"content": "Legal", "metadata": {"domain_tags": ["legal"]}}
        ]
        
        filtered = search.filter_by_metadata(results, {"domain_tags": ["technical", "finance"]})
        
        assert len(filtered) == 2  # Both tech and finance+tech should match

