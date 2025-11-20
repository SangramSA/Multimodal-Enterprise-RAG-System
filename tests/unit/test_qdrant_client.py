"""Unit tests for QdrantClientWrapper."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from vector.qdrant_client import QdrantClientWrapper
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from utils.errors import DatabaseError


class TestQdrantClientWrapper:
    """Test suite for QdrantClientWrapper."""
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock QdrantClient."""
        return Mock()
    
    @pytest.fixture
    def qdrant_wrapper(self, mock_qdrant_client, mocker):
        """Create QdrantClientWrapper with mocked client."""
        with patch('vector.qdrant_client.QdrantClient', return_value=mock_qdrant_client):
            with patch('vector.qdrant_client.QDRANT_URL', 'http://localhost:6333'):
                with patch('vector.qdrant_client.QDRANT_API_KEY', None):
                    with patch('vector.qdrant_client.QDRANT_COLLECTION_NAME', 'test_collection'):
                        with patch('vector.qdrant_client.EMBEDDING_DIMENSION', 1536):
                            mock_qdrant_client.get_collections.return_value = Mock(collections=[])
                            wrapper = QdrantClientWrapper()
                            wrapper.client = mock_qdrant_client
                            return wrapper
    
    def test_init_with_api_key(self, mock_qdrant_client, mocker):
        """Test initialization with API key."""
        mock_qdrant_class = Mock(return_value=mock_qdrant_client)
        with patch('vector.qdrant_client.QdrantClient', mock_qdrant_class):
            with patch('vector.qdrant_client.QDRANT_URL', 'http://localhost:6333'):
                with patch('vector.qdrant_client.QDRANT_API_KEY', 'test-key'):
                    with patch('vector.qdrant_client.QDRANT_COLLECTION_NAME', 'test'):
                        with patch('vector.qdrant_client.EMBEDDING_DIMENSION', 1536):
                            mock_qdrant_client.get_collections.return_value = Mock(collections=[])
                            wrapper = QdrantClientWrapper()
                            # Verify QdrantClient was called with API key
                            mock_qdrant_class.assert_called_with(url='http://localhost:6333', api_key='test-key')
    
    def test_init_without_api_key(self, mock_qdrant_client, mocker):
        """Test initialization without API key."""
        mock_qdrant_class = Mock(return_value=mock_qdrant_client)
        with patch('vector.qdrant_client.QdrantClient', mock_qdrant_class):
            with patch('vector.qdrant_client.QDRANT_URL', 'http://localhost:6333'):
                with patch('vector.qdrant_client.QDRANT_API_KEY', None):
                    with patch('vector.qdrant_client.QDRANT_COLLECTION_NAME', 'test'):
                        with patch('vector.qdrant_client.EMBEDDING_DIMENSION', 1536):
                            mock_qdrant_client.get_collections.return_value = Mock(collections=[])
                            wrapper = QdrantClientWrapper()
                            # Verify QdrantClient was called without API key
                            mock_qdrant_class.assert_called_with(url='http://localhost:6333')
    
    def test_connect_error(self, mocker):
        """Test connection error handling."""
        with patch('vector.qdrant_client.QdrantClient', side_effect=Exception("Connection failed")):
            with patch('vector.qdrant_client.QDRANT_URL', 'http://localhost:6333'):
                with patch('vector.qdrant_client.QDRANT_API_KEY', None):
                    with patch('vector.qdrant_client.QDRANT_COLLECTION_NAME', 'test'):
                        with patch('vector.qdrant_client.EMBEDDING_DIMENSION', 1536):
                            with pytest.raises(DatabaseError, match="Qdrant connection failed"):
                                QdrantClientWrapper()
    
    def test_ensure_collection_exists(self, qdrant_wrapper, mock_qdrant_client):
        """Test ensure_collection when collection already exists."""
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        mock_qdrant_client.get_collections.return_value = Mock(collections=[mock_collection])
        
        qdrant_wrapper.ensure_collection()
        
        # Should not create collection
        mock_qdrant_client.create_collection.assert_not_called()
    
    def test_ensure_collection_create(self, qdrant_wrapper, mock_qdrant_client):
        """Test ensure_collection creates collection when it doesn't exist."""
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])
        
        qdrant_wrapper.ensure_collection()
        
        # Should create collection
        mock_qdrant_client.create_collection.assert_called_once()
    
    def test_ensure_collection_error(self, qdrant_wrapper, mock_qdrant_client):
        """Test ensure_collection error handling."""
        mock_qdrant_client.get_collections.side_effect = Exception("Database error")
        
        with pytest.raises(DatabaseError, match="Collection creation failed"):
            qdrant_wrapper.ensure_collection()
    
    def test_upsert_points_success(self, qdrant_wrapper, mock_qdrant_client):
        """Test successful point upsertion."""
        points = [
            PointStruct(id=1, vector=[0.1] * 1536, payload={"content": "test"})
        ]
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])
        mock_qdrant_client.upsert.return_value = None
        
        result = qdrant_wrapper.upsert_points(points)
        
        assert result is True
        mock_qdrant_client.upsert.assert_called_once()
    
    def test_upsert_points_error(self, qdrant_wrapper, mock_qdrant_client):
        """Test point upsertion error handling."""
        points = [PointStruct(id=1, vector=[0.1] * 1536, payload={})]
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])
        mock_qdrant_client.upsert.side_effect = Exception("Upsert failed")
        
        result = qdrant_wrapper.upsert_points(points)
        
        assert result is False
    
    def test_search_success(self, qdrant_wrapper, mock_qdrant_client):
        """Test successful vector search."""
        mock_result = Mock()
        mock_result.id = 1
        mock_result.score = 0.95
        mock_result.payload = {"content": "test"}
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])
        mock_qdrant_client.search.return_value = [mock_result]
        
        query_vector = [0.1] * 1536
        results = qdrant_wrapper.search(query_vector, limit=10)
        
        assert len(results) == 1
        assert results[0]["id"] == 1
        assert results[0]["score"] == 0.95
        assert results[0]["payload"] == {"content": "test"}
    
    def test_search_with_filter(self, qdrant_wrapper, mock_qdrant_client):
        """Test search with filter."""
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])
        mock_qdrant_client.search.return_value = []
        
        query_vector = [0.1] * 1536
        filter_condition = Filter(
            must=[FieldCondition(key="modality", match=MatchValue(value="text"))]
        )
        
        qdrant_wrapper.search(query_vector, limit=10, filter=filter_condition)
        
        # Verify filter was passed
        call_args = mock_qdrant_client.search.call_args
        assert call_args.kwargs.get("query_filter") == filter_condition
    
    def test_search_with_score_threshold(self, qdrant_wrapper, mock_qdrant_client):
        """Test search with score threshold."""
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])
        mock_qdrant_client.search.return_value = []
        
        query_vector = [0.1] * 1536
        qdrant_wrapper.search(query_vector, limit=10, score_threshold=0.7)
        
        call_args = mock_qdrant_client.search.call_args
        assert call_args.kwargs.get("score_threshold") == 0.7
    
    def test_search_error(self, qdrant_wrapper, mock_qdrant_client):
        """Test search error handling."""
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])
        mock_qdrant_client.search.side_effect = Exception("Search failed")
        
        query_vector = [0.1] * 1536
        results = qdrant_wrapper.search(query_vector, limit=10)
        
        assert results == []
    
    def test_delete_points_success(self, qdrant_wrapper, mock_qdrant_client):
        """Test successful point deletion."""
        mock_qdrant_client.delete.return_value = None
        
        result = qdrant_wrapper.delete_points([1, 2, 3])
        
        assert result is True
        mock_qdrant_client.delete.assert_called_once()
    
    def test_delete_points_error(self, qdrant_wrapper, mock_qdrant_client):
        """Test point deletion error handling."""
        mock_qdrant_client.delete.side_effect = Exception("Delete failed")
        
        result = qdrant_wrapper.delete_points([1, 2])
        
        assert result is False
    
    def test_get_point_success(self, qdrant_wrapper, mock_qdrant_client):
        """Test successful point retrieval."""
        mock_point = Mock()
        mock_point.id = 1
        mock_point.vector = [0.1] * 1536
        mock_point.payload = {"content": "test"}
        mock_qdrant_client.retrieve.return_value = [mock_point]
        
        result = qdrant_wrapper.get_point(1)
        
        assert result is not None
        assert result["id"] == 1
        assert result["vector"] == [0.1] * 1536
        assert result["payload"] == {"content": "test"}
    
    def test_get_point_not_found(self, qdrant_wrapper, mock_qdrant_client):
        """Test point retrieval when point doesn't exist."""
        mock_qdrant_client.retrieve.return_value = []
        
        result = qdrant_wrapper.get_point(999)
        
        assert result is None
    
    def test_get_point_error(self, qdrant_wrapper, mock_qdrant_client):
        """Test point retrieval error handling."""
        mock_qdrant_client.retrieve.side_effect = Exception("Retrieve failed")
        
        result = qdrant_wrapper.get_point(1)
        
        assert result is None
    
    def test_health_check_success(self, qdrant_wrapper, mock_qdrant_client):
        """Test successful health check."""
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])
        
        result = qdrant_wrapper.health_check()
        
        assert result is True
    
    def test_health_check_failure(self, qdrant_wrapper, mock_qdrant_client):
        """Test health check failure."""
        mock_qdrant_client.get_collections.side_effect = Exception("Health check failed")
        
        result = qdrant_wrapper.health_check()
        
        assert result is False

