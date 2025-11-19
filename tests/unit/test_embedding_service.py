"""Unit tests for embedding service."""

import pytest
from unittest.mock import Mock, patch
from vector.embedding_service import EmbeddingService
from utils.errors import APIError
from tests.utils.mock_helpers import create_mock_openai_embedding_response


class TestEmbeddingService:
    """Test suite for EmbeddingService."""
    
    @patch('vector.embedding_service.openai.OpenAI')
    @patch('vector.embedding_service.OPENAI_API_KEY', 'test-key')
    def test_embedding_service_initialization(self, mock_openai):
        """Test embedding service initialization."""
        service = EmbeddingService()
        assert service.model == "text-embedding-3-small"
        assert service.dimension == 1536
        mock_openai.assert_called_once_with(api_key='test-key')
    
    @patch('vector.embedding_service.OPENAI_API_KEY', None)
    def test_embedding_service_missing_api_key(self):
        """Test that missing API key raises error."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            EmbeddingService()
    
    @patch('vector.embedding_service.openai.OpenAI')
    @patch('vector.embedding_service.OPENAI_API_KEY', 'test-key')
    @patch('vector.embedding_service.retry_with_backoff')
    def test_embed_text_success(self, mock_retry, mock_openai_class):
        """Test successful text embedding."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock embedding response
        mock_response = create_mock_openai_embedding_response(embedding_dim=1536)
        mock_client.embeddings.create.return_value = mock_response
        
        # Mock retry function to return the embedding directly
        mock_retry.side_effect = lambda func, **kwargs: func()
        
        service = EmbeddingService()
        embedding = service.embed_text("Test text")
        
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)
        mock_client.embeddings.create.assert_called_once()
    
    @patch('vector.embedding_service.openai.OpenAI')
    @patch('vector.embedding_service.OPENAI_API_KEY', 'test-key')
    @patch('vector.embedding_service.retry_with_backoff')
    def test_embed_text_api_error(self, mock_retry, mock_openai_class):
        """Test handling of API errors."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock retry to raise exception
        mock_retry.side_effect = Exception("API Error")
        
        service = EmbeddingService()
        
        with pytest.raises(APIError, match="Failed to generate embedding"):
            service.embed_text("Test text")
    
    @patch('vector.embedding_service.openai.OpenAI')
    @patch('vector.embedding_service.OPENAI_API_KEY', 'test-key')
    @patch('vector.embedding_service.retry_with_backoff')
    def test_embed_batch_success(self, mock_retry, mock_openai_class):
        """Test successful batch embedding."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock batch response
        mock_response = Mock()
        mock_data1 = Mock()
        mock_data1.embedding = [0.1] * 1536
        mock_data2 = Mock()
        mock_data2.embedding = [0.2] * 1536
        mock_response.data = [mock_data1, mock_data2]
        mock_client.embeddings.create.return_value = mock_response
        
        # Mock retry function
        mock_retry.side_effect = lambda func, **kwargs: func()
        
        service = EmbeddingService()
        texts = ["Text 1", "Text 2"]
        embeddings = service.embed_batch(texts, batch_size=100)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        assert len(embeddings[1]) == 1536
    
    @patch('vector.embedding_service.openai.OpenAI')
    @patch('vector.embedding_service.OPENAI_API_KEY', 'test-key')
    @patch('vector.embedding_service.retry_with_backoff')
    def test_embed_batch_large(self, mock_retry, mock_openai_class):
        """Test batch embedding with large number of texts."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock response for each batch
        def create_batch_response(count):
            mock_response = Mock()
            mock_response.data = [
                Mock(embedding=[0.1] * 1536) for _ in range(count)
            ]
            return mock_response
        
        mock_client.embeddings.create.side_effect = [
            create_batch_response(50),
            create_batch_response(30)
        ]
        
        # Mock retry function
        mock_retry.side_effect = lambda func, **kwargs: func()
        
        service = EmbeddingService()
        texts = ["Text"] * 80  # Larger than batch_size
        embeddings = service.embed_batch(texts, batch_size=50)
        
        assert len(embeddings) == 80
        assert mock_client.embeddings.create.call_count == 2
    
    @patch('vector.embedding_service.openai.OpenAI')
    @patch('vector.embedding_service.OPENAI_API_KEY', 'test-key')
    @patch('vector.embedding_service.retry_with_backoff')
    def test_embed_batch_error_handling(self, mock_retry, mock_openai_class):
        """Test batch embedding error handling."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # First batch succeeds, second fails
        mock_response1 = Mock()
        mock_response1.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.side_effect = [
            mock_response1,
            Exception("API Error")
        ]
        
        # Mock retry to raise on second call
        call_count = 0
        def mock_retry_side_effect(func, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return func()
            else:
                raise Exception("API Error")
        
        mock_retry.side_effect = mock_retry_side_effect
        
        service = EmbeddingService()
        texts = ["Text 1", "Text 2"]
        embeddings = service.embed_batch(texts, batch_size=1)
        
        # Should have one successful embedding and one empty list
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        assert len(embeddings[1]) == 0  # Failed batch
    
    def test_get_dimension(self):
        """Test getting embedding dimension."""
        with patch('vector.embedding_service.openai.OpenAI'), \
             patch('vector.embedding_service.OPENAI_API_KEY', 'test-key'):
            service = EmbeddingService()
            assert service.get_dimension() == 1536

