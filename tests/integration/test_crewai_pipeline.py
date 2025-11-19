"""Integration tests for CrewAI pipeline."""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from pipeline.agentic_query_pipeline import AgenticQueryPipeline
from agents.retrieval_orchestration_agent import RetrievalOrchestrationAgent
from search.hybrid_search import HybridSearch
from search.graph_search import GraphSearch
from search.keyword_search import KeywordSearch
from search.vector_search import VectorSearch


@pytest.fixture
def mock_retrieval_agent():
    """Create mock retrieval agent."""
    return Mock(spec=RetrievalOrchestrationAgent)


@pytest.fixture
def mock_search_components():
    """Create mock search components."""
    hybrid_search = Mock(spec=HybridSearch)
    graph_search = Mock(spec=GraphSearch)
    keyword_search = Mock(spec=KeywordSearch)
    vector_search = Mock(spec=VectorSearch)
    
    return {
        "hybrid_search": hybrid_search,
        "graph_search": graph_search,
        "keyword_search": keyword_search,
        "vector_search": vector_search
    }


@pytest.mark.skipif(
    not pytest.importorskip("crewai", reason="CrewAI not installed"),
    reason="CrewAI not available"
)
def test_crewai_pipeline_initialization(mock_retrieval_agent):
    """Test CrewAI pipeline initialization."""
    with patch('pipeline.agentic_query_pipeline.CREWAI_AVAILABLE', True):
        pipeline = AgenticQueryPipeline(mock_retrieval_agent, use_crewai=True)
        assert pipeline.use_crewai is True
        assert hasattr(pipeline, 'crewai_orchestrator')


def test_custom_pipeline_initialization(mock_retrieval_agent):
    """Test custom pipeline initialization (fallback)."""
    pipeline = AgenticQueryPipeline(mock_retrieval_agent, use_crewai=False)
    assert pipeline.use_crewai is False
    assert hasattr(pipeline, 'validation_agent')
    assert hasattr(pipeline, 'triage_agent')


@pytest.mark.skipif(
    not pytest.importorskip("crewai", reason="CrewAI not installed"),
    reason="CrewAI not available"
)
def test_crewai_pipeline_execution(mock_retrieval_agent, mock_search_components):
    """Test CrewAI pipeline execution."""
    # Mock retrieval result
    mock_retrieval_agent.retrieve.return_value = {
        "results": [
            {"content": "Test content", "chunk_id": "chunk1", "metadata": {"file_name": "test.pdf"}}
        ],
        "methods_used": ["hybrid"],
        "confidence": 0.8
    }
    
    with patch('pipeline.agentic_query_pipeline.CREWAI_AVAILABLE', True):
        pipeline = AgenticQueryPipeline(mock_retrieval_agent, use_crewai=True)
        
        # Mock orchestrator execution
        with patch.object(pipeline.crewai_orchestrator, 'execute_pipeline') as mock_execute:
            mock_execute.return_value = {
                "query": "test query",
                "answer": "Test answer",
                "sources": [],
                "confidence": 0.8,
                "metadata": {}
            }
            
            result = pipeline.process("test query")
            
            assert result["answer"] == "Test answer"
            assert result["confidence"] == 0.8
            mock_execute.assert_called_once()


def test_custom_pipeline_execution(mock_retrieval_agent):
    """Test custom pipeline execution."""
    pipeline = AgenticQueryPipeline(mock_retrieval_agent, use_crewai=False)
    
    # Mock agent responses
    with patch.object(pipeline.validation_agent, 'validate') as mock_validate, \
         patch.object(pipeline.triage_agent, 'triage') as mock_triage, \
         patch.object(pipeline.retrieval_agent, 'retrieve') as mock_retrieve, \
         patch.object(pipeline.answer_agent, 'generate') as mock_generate, \
         patch.object(pipeline.postprocess_agent, 'process') as mock_postprocess:
        
        mock_validate.return_value = {
            "is_valid": True,
            "sanitized_query": "test query",
            "complexity": "simple",
            "intent": "factual"
        }
        
        mock_triage.return_value = {
            "query_type": "factual_lookup",
            "expanded_query": "test query",
            "search_strategy": {"use_hybrid": True},
            "confidence": 0.8
        }
        
        mock_retrieve.return_value = {
            "results": [{"content": "test", "chunk_id": "chunk1"}],
            "methods_used": ["hybrid"],
            "confidence": 0.8
        }
        
        mock_generate.return_value = {
            "answer": "Test answer",
            "citations": [],
            "reasoning_steps": []
        }
        
        mock_postprocess.return_value = {
            "final_answer": "Test answer",
            "confidence": 0.8,
            "hallucination_score": 0.1
        }
        
        result = pipeline.process("test query")
        
        assert result["answer"] == "Test answer"
        assert result["confidence"] == 0.8


def test_pipeline_backward_compatibility(mock_retrieval_agent):
    """Test that pipeline maintains backward compatibility."""
    # Default initialization (no use_crewai parameter)
    pipeline = AgenticQueryPipeline(mock_retrieval_agent)
    assert pipeline.use_crewai is False  # Should default to False
    
    # Process method should work the same
    assert hasattr(pipeline, 'process')
    assert callable(pipeline.process)

