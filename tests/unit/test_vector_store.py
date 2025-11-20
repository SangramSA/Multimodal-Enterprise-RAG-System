"""Unit tests for VectorStore."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import hashlib

from vector.vector_store import VectorStore
from vector.qdrant_client import QdrantClientWrapper
from vector.embedding_service import EmbeddingService


class TestVectorStore:
    """Test suite for VectorStore."""
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock QdrantClientWrapper."""
        return Mock(spec=QdrantClientWrapper)
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock EmbeddingService."""
        return Mock(spec=EmbeddingService)
    
    @pytest.fixture
    def vector_store(self, mock_qdrant_client, mock_embedding_service):
        """Create VectorStore instance."""
        return VectorStore(mock_qdrant_client, mock_embedding_service)
    
    def test_init_with_dependencies(self, mock_qdrant_client, mock_embedding_service):
        """Test initialization with provided dependencies."""
        store = VectorStore(mock_qdrant_client, mock_embedding_service)
        assert store.qdrant == mock_qdrant_client
        assert store.embeddings == mock_embedding_service
    
    def test_init_without_dependencies(self, mocker):
        """Test initialization without dependencies (creates new instances)."""
        with patch('vector.vector_store.QdrantClientWrapper') as mock_qdrant_class:
            with patch('vector.vector_store.EmbeddingService') as mock_embedding_class:
                mock_qdrant_class.return_value = Mock()
                mock_embedding_class.return_value = Mock()
                
                store = VectorStore()
                
                mock_qdrant_class.assert_called_once()
                mock_embedding_class.assert_called_once()
    
    def test_index_chunks_empty(self, vector_store):
        """Test indexing empty chunk list."""
        result = vector_store.index_chunks([])
        assert result == 0
    
    def test_index_chunks_success(self, vector_store, mock_embedding_service, mock_qdrant_client):
        """Test successful chunk indexing."""
        chunks = [
            {
                "chunk_id": "chunk1",
                "content": "test content 1",
                "metadata": {"modality": "text"}
            },
            {
                "chunk_id": "chunk2",
                "content": "test content 2",
                "metadata": {"modality": "text"}
            }
        ]
        mock_embedding_service.embed_batch.return_value = [
            [0.1] * 1536,
            [0.2] * 1536
        ]
        mock_qdrant_client.upsert_points.return_value = True
        
        result = vector_store.index_chunks(chunks)
        
        assert result == 2
        mock_embedding_service.embed_batch.assert_called_once()
        mock_qdrant_client.upsert_points.assert_called_once()
    
    def test_index_chunks_embedding_failure(self, vector_store, mock_embedding_service, mock_qdrant_client):
        """Test indexing when embedding fails for some chunks."""
        chunks = [
            {"chunk_id": "chunk1", "content": "test1"},
            {"chunk_id": "chunk2", "content": "test2"}
        ]
        # One embedding is None (failure)
        mock_embedding_service.embed_batch.return_value = [
            [0.1] * 1536,
            None  # Embedding failure
        ]
        mock_qdrant_client.upsert_points.return_value = True
        
        result = vector_store.index_chunks(chunks)
        
        # Should only index 1 chunk (the one with successful embedding)
        assert result == 1
    
    def test_index_chunks_point_id_conversion(self, vector_store, mock_embedding_service, mock_qdrant_client):
        """Test that chunk_id is converted to integer point_id."""
        chunks = [{"chunk_id": "test_chunk_1", "content": "test"}]
        mock_embedding_service.embed_batch.return_value = [[0.1] * 1536]
        mock_qdrant_client.upsert_points.return_value = True
        
        vector_store.index_chunks(chunks)
        
        # Verify point was created with integer ID
        call_args = mock_qdrant_client.upsert_points.call_args
        points = call_args[0][0]
        assert isinstance(points[0].id, int)
    
    def test_index_chunks_upsert_failure(self, vector_store, mock_embedding_service, mock_qdrant_client):
        """Test indexing when upsert fails."""
        chunks = [{"chunk_id": "chunk1", "content": "test"}]
        mock_embedding_service.embed_batch.return_value = [[0.1] * 1536]
        mock_qdrant_client.upsert_points.return_value = False
        
        result = vector_store.index_chunks(chunks)
        
        assert result == 0
    
    def test_search_success(self, vector_store, mock_embedding_service, mock_qdrant_client):
        """Test successful search."""
        mock_embedding_service.embed_text.return_value = [0.1] * 1536
        mock_qdrant_client.search.return_value = [
            {
                "id": 1,
                "score": 0.95,
                "payload": {
                    "chunk_id": "chunk1",
                    "content": "test content",
                    "modality": "text"
                }
            }
        ]
        
        results = vector_store.search("test query", limit=10)
        
        assert len(results) == 1
        assert results[0]["chunk_id"] == "chunk1"
        assert results[0]["score"] == 0.95
        assert results[0]["content"] == "test content"
    
    def test_search_with_filters(self, vector_store, mock_embedding_service, mock_qdrant_client):
        """Test search with metadata filters."""
        mock_embedding_service.embed_text.return_value = [0.1] * 1536
        mock_qdrant_client.search.return_value = [
            {
                "id": 1,
                "score": 0.95,
                "payload": {
                    "chunk_id": "chunk1",
                    "content": "test",
                    "modality": "text",
                    "domain_tags": ["technical"]
                }
            }
        ]
        
        filters = {"modality": "text"}
        results = vector_store.search("test query", limit=10, filters=filters)
        
        # Verify filter was passed to Qdrant
        call_args = mock_qdrant_client.search.call_args
        assert call_args is not None
    
    def test_search_with_domain_tag_filter(self, vector_store, mock_embedding_service, mock_qdrant_client):
        """Test search with domain tag filtering."""
        mock_embedding_service.embed_text.return_value = [0.1] * 1536
        mock_qdrant_client.search.return_value = [
            {
                "id": 1,
                "score": 0.95,
                "payload": {
                    "chunk_id": "chunk1",
                    "content": "test",
                    "domain_tags": ["technical", "finance"]
                }
            },
            {
                "id": 2,
                "score": 0.85,
                "payload": {
                    "chunk_id": "chunk2",
                    "content": "test2",
                    "domain_tags": ["legal"]
                }
            }
        ]
        
        filters = {"domain_tags": ["technical"]}
        results = vector_store.search("test query", limit=10, filters=filters)
        
        # Should filter to only chunks with "technical" tag
        assert len(results) == 1
        assert results[0]["chunk_id"] == "chunk1"
    
    def test_search_with_score_threshold(self, vector_store, mock_embedding_service, mock_qdrant_client):
        """Test search with score threshold."""
        mock_embedding_service.embed_text.return_value = [0.1] * 1536
        mock_qdrant_client.search.return_value = []
        
        vector_store.search("test query", limit=10, score_threshold=0.7)
        
        call_args = mock_qdrant_client.search.call_args
        assert call_args.kwargs.get("score_threshold") == 0.7
    
    def test_chunk_id_to_point_id(self, vector_store):
        """Test chunk_id to point_id conversion."""
        chunk_id = "test_chunk_123"
        point_id = vector_store._chunk_id_to_point_id(chunk_id)
        
        # Should be deterministic
        point_id2 = vector_store._chunk_id_to_point_id(chunk_id)
        assert point_id == point_id2
        
        # Should be integer
        assert isinstance(point_id, int)
        
        # Should match expected hash
        expected = int(hashlib.md5(chunk_id.encode('utf-8')).hexdigest(), 16) % (2**63)
        assert point_id == expected
    
    def test_delete_chunks_success(self, vector_store, mock_qdrant_client):
        """Test successful chunk deletion."""
        mock_qdrant_client.delete_points.return_value = True
        
        result = vector_store.delete_chunks(["chunk1", "chunk2"])
        
        assert result is True
        mock_qdrant_client.delete_points.assert_called_once()
        # Verify point IDs were converted
        call_args = mock_qdrant_client.delete_points.call_args
        point_ids = call_args[0][0]
        assert all(isinstance(pid, int) for pid in point_ids)
    
    def test_delete_chunks_failure(self, vector_store, mock_qdrant_client):
        """Test chunk deletion failure."""
        mock_qdrant_client.delete_points.return_value = False
        
        result = vector_store.delete_chunks(["chunk1"])
        
        assert result is False
    
    def test_get_chunk_success(self, vector_store, mock_qdrant_client):
        """Test successful chunk retrieval."""
        mock_qdrant_client.get_point.return_value = {
            "id": 12345,
            "vector": [0.1] * 1536,
            "payload": {
                "chunk_id": "chunk1",
                "content": "test content",
                "modality": "text"
            }
        }
        
        result = vector_store.get_chunk("chunk1")
        
        assert result is not None
        assert result["chunk_id"] == "chunk1"
        assert result["content"] == "test content"
        assert "metadata" in result
    
    def test_get_chunk_not_found(self, vector_store, mock_qdrant_client):
        """Test chunk retrieval when chunk doesn't exist."""
        mock_qdrant_client.get_point.return_value = None
        
        result = vector_store.get_chunk("nonexistent")
        
        assert result is None
    
    def test_get_chunks_by_file_id_success(self, vector_store, mock_qdrant_client, mock_embedding_service):
        """Test retrieving chunks by file_id."""
        mock_embedding_service.get_dimension.return_value = 1536
        mock_qdrant_client.search.return_value = [
            {
                "id": 1,
                "score": 1.0,
                "payload": {
                    "chunk_id": "file1_chunk_0",
                    "chunk_index": 0,
                    "content": "first chunk",
                    "file_id": "file1",
                    "modality": "text"
                }
            },
            {
                "id": 2,
                "score": 1.0,
                "payload": {
                    "chunk_id": "file1_chunk_1",
                    "chunk_index": 1,
                    "content": "second chunk",
                    "file_id": "file1",
                    "modality": "text"
                }
            }
        ]
        
        chunks = vector_store.get_chunks_by_file_id("file1")
        
        assert len(chunks) == 2
        assert chunks[0]["chunk_index"] == 0
        assert chunks[1]["chunk_index"] == 1
        # Should be sorted by chunk_index
        assert chunks[0]["chunk_index"] < chunks[1]["chunk_index"]
    
    def test_get_chunks_by_file_id_empty(self, vector_store, mock_qdrant_client, mock_embedding_service):
        """Test retrieving chunks when file_id has no chunks."""
        mock_embedding_service.get_dimension.return_value = 1536
        mock_qdrant_client.search.return_value = []
        
        chunks = vector_store.get_chunks_by_file_id("nonexistent")
        
        assert chunks == []
    
    def test_get_chunks_by_file_id_error(self, vector_store, mock_qdrant_client, mock_embedding_service):
        """Test error handling in get_chunks_by_file_id."""
        mock_embedding_service.get_dimension.return_value = 1536
        mock_qdrant_client.search.side_effect = Exception("Search error")
        
        chunks = vector_store.get_chunks_by_file_id("file1")
        
        assert chunks == []

