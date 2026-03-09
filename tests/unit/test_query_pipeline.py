"""Unit tests for QueryPipeline."""

import pytest
from unittest.mock import Mock, patch

from pipeline.query_pipeline import QueryPipeline
from agents.retrieval_agent import RetrievalAgent


class TestQueryPipeline:
    """Test suite for QueryPipeline."""
    
    @pytest.fixture
    def mock_retrieval_agent(self):
        """Create mock RetrievalAgent."""
        return Mock(spec=RetrievalAgent)
    
    @pytest.fixture
    def query_pipeline(self, mock_retrieval_agent, mocker):
        """Create QueryPipeline with mocked dependencies."""
        with patch('pipeline.query_pipeline.QueryRewriter') as mock_rewriter:
            with patch('pipeline.query_pipeline.InputValidator') as mock_validator:
                with patch('pipeline.query_pipeline.PipelineCache') as mock_cache_cls:
                    mock_rewriter.return_value = Mock()
                    mock_validator.return_value = Mock()
                    # Configure pipeline cache to behave like a simple miss-only cache.
                    mock_cache = mock_cache_cls.return_value
                    mock_cache.lookup.return_value = (False, None, None, None)
                    pipeline = QueryPipeline(mock_retrieval_agent)
                    pipeline.query_rewriter = mock_rewriter.return_value
                    pipeline.validator = mock_validator.return_value
                    pipeline.pipeline_cache = mock_cache
                    return pipeline
    
    def test_init(self, mock_retrieval_agent, mocker):
        """Test QueryPipeline initialization."""
        with patch('pipeline.query_pipeline.QueryRewriter'):
            pipeline = QueryPipeline(mock_retrieval_agent)
            assert pipeline.query_rewriter is not None
            assert pipeline.retrieval_agent == mock_retrieval_agent
    
    def test_process_query_success(self, query_pipeline, mock_retrieval_agent, mocker):
        """Test successful query processing."""
        query_pipeline.validator.validate_query.return_value = {
            "is_valid": True,
            "sanitized_query": "test query"
        }
        mock_rewritten = {"expanded_query": "rewritten query", "query_type": "factual"}
        query_pipeline.query_rewriter.rewrite_query.return_value = mock_rewritten
        
        mock_results = [{"chunk_id": "chunk1", "content": "result", "score": 0.9}]
        mock_retrieval_agent.retrieve.return_value = mock_results
        
        with patch.object(query_pipeline, 'client') as mock_client:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Answer text"
            mock_client.chat.completions.create.return_value = mock_response
            
            result = query_pipeline.process("test query")
            
            assert "answer" in result
            assert "sources" in result
            assert "confidence" in result
            metadata = result.get("metadata", {})
            # Stage-level timings and cache lookup timing should be present.
            assert "validation_time" in metadata
            assert "triage_time" in metadata
            assert "retrieval_time" in metadata
            assert "generation_time" in metadata
            assert "pipeline_cache_lookup_ms" in metadata
    
    def test_process_query_validation_error(self, query_pipeline, mocker):
        """Test query processing with validation error."""
        query_pipeline.validator.validate_query.side_effect = Exception("Validation failed")
        
        from utils.errors import ValidationError
        with pytest.raises(Exception):  # Should raise ValidationError or return error dict
            result = query_pipeline.process("")
            # If it returns a dict instead of raising, check for error
            if isinstance(result, dict):
                assert "error" in result or "answer" in result

