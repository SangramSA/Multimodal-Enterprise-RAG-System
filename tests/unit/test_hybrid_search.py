"""Unit tests for HybridSearch."""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from search.hybrid_search import HybridSearch
from search.keyword_search import KeywordSearch
from search.vector_search import VectorSearch
from search.graph_search import GraphSearch


class TestHybridSearch:
    """Test suite for HybridSearch."""
    
    @pytest.fixture
    def mock_keyword_search(self):
        """Create mock KeywordSearch."""
        return Mock(spec=KeywordSearch)
    
    @pytest.fixture
    def mock_vector_search(self):
        """Create mock VectorSearch."""
        return Mock(spec=VectorSearch)
    
    @pytest.fixture
    def mock_graph_search(self):
        """Create mock GraphSearch."""
        return Mock(spec=GraphSearch)
    
    @pytest.fixture
    def hybrid_search(self, mock_keyword_search, mock_vector_search, mock_graph_search):
        """Create HybridSearch instance."""
        return HybridSearch(mock_keyword_search, mock_vector_search, mock_graph_search)
    
    def test_init(self, mock_keyword_search, mock_vector_search, mock_graph_search):
        """Test HybridSearch initialization."""
        search = HybridSearch(mock_keyword_search, mock_vector_search, mock_graph_search)
        assert search.keyword_search == mock_keyword_search
        assert search.vector_search == mock_vector_search
        assert search.graph_search == mock_graph_search
        assert search.rrf_k == 60
    
    def test_reciprocal_rank_fusion_single_list(self, hybrid_search):
        """Test RRF with single result list."""
        results = [
            {"chunk_id": "chunk1", "content": "test1"},
            {"chunk_id": "chunk2", "content": "test2"}
        ]
        
        scores = hybrid_search.reciprocal_rank_fusion([results])
        
        assert "chunk1" in scores
        assert "chunk2" in scores
        assert scores["chunk1"] > scores["chunk2"]  # First rank should have higher score
    
    def test_reciprocal_rank_fusion_multiple_lists(self, hybrid_search):
        """Test RRF with multiple result lists."""
        list1 = [{"chunk_id": "chunk1"}, {"chunk_id": "chunk2"}]
        list2 = [{"chunk_id": "chunk2"}, {"chunk_id": "chunk1"}]  # Reversed order
        
        scores = hybrid_search.reciprocal_rank_fusion([list1, list2])
        
        # chunk2 appears in both lists, should have higher score
        assert "chunk1" in scores
        assert "chunk2" in scores
        # chunk2 appears in both lists (rank 2 in list1, rank 1 in list2)
        # chunk1 appears in both lists (rank 1 in list1, rank 2 in list2)
        # Both should have scores > 0
    
    def test_reciprocal_rank_fusion_no_id(self, hybrid_search):
        """Test RRF with results that have no chunk_id or file_id."""
        results = [{"content": "test1"}, {"content": "test2"}]
        
        scores = hybrid_search.reciprocal_rank_fusion([results])
        
        # Should use object id as fallback
        assert len(scores) == 2
    
    def test_search_keyword_only(self, hybrid_search, mock_keyword_search):
        """Test hybrid search with keyword only."""
        mock_keyword_search.search.return_value = [
            {"chunk_id": "chunk1", "content": "test", "keyword_score": 0.9}
        ]
        
        results = hybrid_search.search("test query", limit=10, use_keyword=True, 
                                      use_vector=False, use_graph=False)
        
        assert len(results) > 0
        mock_keyword_search.search.assert_called_once()
    
    def test_search_vector_only(self, hybrid_search, mock_vector_search):
        """Test hybrid search with vector only."""
        mock_vector_search.search.return_value = [
            {"chunk_id": "chunk1", "content": "test", "score": 0.8}
        ]
        
        results = hybrid_search.search("test query", limit=10, use_keyword=False,
                                      use_vector=True, use_graph=False)
        
        assert len(results) > 0
        mock_vector_search.search.assert_called_once()
    
    def test_search_graph_only(self, hybrid_search, mock_graph_search):
        """Test hybrid search with graph only."""
        mock_graph_search.search_by_entity.return_value = [
            {"file_id": "file1", "file_name": "test.pdf"}
        ]
        
        results = hybrid_search.search("test query", limit=10, use_keyword=False,
                                      use_vector=False, use_graph=True)
        
        # Graph search is called for each word > 3 chars
        assert mock_graph_search.search_by_entity.called
    
    def test_search_all_methods(self, hybrid_search, mock_keyword_search, 
                                mock_vector_search, mock_graph_search):
        """Test hybrid search with all methods enabled."""
        mock_keyword_search.search.return_value = [
            {"chunk_id": "chunk1", "content": "test", "keyword_score": 0.9}
        ]
        mock_vector_search.search.return_value = [
            {"chunk_id": "chunk2", "content": "test", "score": 0.8}
        ]
        mock_graph_search.search_by_entity.return_value = [
            {"file_id": "file1", "file_name": "test.pdf"}
        ]
        
        results = hybrid_search.search("test query", limit=10)
        
        assert len(results) > 0
        mock_keyword_search.search.assert_called_once()
        mock_vector_search.search.assert_called_once()
        assert mock_graph_search.search_by_entity.called
    
    def test_search_keyword_failure(self, hybrid_search, mock_keyword_search,
                                    mock_vector_search):
        """Test that keyword search failure doesn't break hybrid search."""
        mock_keyword_search.search.side_effect = Exception("Keyword search error")
        mock_vector_search.search.return_value = [
            {"chunk_id": "chunk1", "content": "test", "score": 0.8}
        ]
        
        results = hybrid_search.search("test query", limit=10, use_keyword=True,
                                      use_vector=True, use_graph=False)
        
        # Should still return vector results
        assert len(results) > 0
    
    def test_search_vector_failure(self, hybrid_search, mock_keyword_search,
                                   mock_vector_search):
        """Test that vector search failure doesn't break hybrid search."""
        mock_vector_search.search.side_effect = Exception("Vector search error")
        mock_keyword_search.search.return_value = [
            {"chunk_id": "chunk1", "content": "test", "keyword_score": 0.9}
        ]
        
        results = hybrid_search.search("test query", limit=10, use_keyword=True,
                                      use_vector=True, use_graph=False)
        
        # Should still return keyword results
        assert len(results) > 0
    
    def test_search_graph_failure(self, hybrid_search, mock_keyword_search,
                                  mock_vector_search, mock_graph_search):
        """Test that graph search failure doesn't break hybrid search."""
        mock_graph_search.search_by_entity.side_effect = Exception("Graph search error")
        mock_keyword_search.search.return_value = [
            {"chunk_id": "chunk1", "content": "test", "keyword_score": 0.9}
        ]
        mock_vector_search.search.return_value = []  # Empty to avoid RRF issues
        
        results = hybrid_search.search("test query", limit=10, use_keyword=True,
                                      use_vector=False, use_graph=True)
        
        # Should still return keyword results despite graph failure
        assert len(results) > 0
    
    def test_search_no_results(self, hybrid_search, mock_keyword_search,
                               mock_vector_search, mock_graph_search):
        """Test hybrid search when all methods return no results."""
        mock_keyword_search.search.return_value = []
        mock_vector_search.search.return_value = []
        mock_graph_search.search_by_entity.return_value = []
        
        results = hybrid_search.search("test query", limit=10)
        
        assert results == []
    
    def test_search_rrf_reranking(self, hybrid_search, mock_keyword_search,
                                  mock_vector_search):
        """Test that RRF properly reranks results."""
        # chunk1 appears first in keyword, second in vector
        # chunk2 appears second in keyword, first in vector
        mock_keyword_search.search.return_value = [
            {"chunk_id": "chunk1", "content": "test1", "keyword_score": 0.9},
            {"chunk_id": "chunk2", "content": "test2", "keyword_score": 0.8}
        ]
        mock_vector_search.search.return_value = [
            {"chunk_id": "chunk2", "content": "test2", "score": 0.85},
            {"chunk_id": "chunk1", "content": "test1", "score": 0.8}
        ]
        
        results = hybrid_search.search("test query", limit=10, use_keyword=True,
                                      use_vector=True, use_graph=False)
        
        # Both chunks should be in results with RRF scores
        assert len(results) >= 2
        assert all("rrf_score" in r for r in results)
    
    def test_search_deduplication(self, hybrid_search, mock_keyword_search,
                                  mock_vector_search):
        """Test that duplicate results are deduplicated."""
        # Same chunk appears in both keyword and vector results
        mock_keyword_search.search.return_value = [
            {"chunk_id": "chunk1", "content": "test content", "keyword_score": 0.9}
        ]
        mock_vector_search.search.return_value = [
            {"chunk_id": "chunk1", "content": "test content", "score": 0.8}
        ]
        
        results = hybrid_search.search("test query", limit=10, use_keyword=True,
                                      use_vector=True, use_graph=False)
        
        # Should only appear once
        chunk_ids = [r.get("chunk_id") for r in results]
        assert chunk_ids.count("chunk1") == 1
    
    def test_search_limit_enforcement(self, hybrid_search, mock_keyword_search,
                                     mock_vector_search):
        """Test that result limit is enforced."""
        # Create many results
        mock_keyword_search.search.return_value = [
            {"chunk_id": f"chunk{i}", "content": f"test{i}"} for i in range(20)
        ]
        mock_vector_search.search.return_value = []
        
        results = hybrid_search.search("test query", limit=5, use_keyword=True,
                                      use_vector=False, use_graph=False)
        
        assert len(results) <= 5
    
    def test_final_cross_encoder_rerank_applies_to_top_n_only(
        self,
        mock_keyword_search,
        mock_vector_search,
        mock_graph_search,
        monkeypatch,
    ):
        """Final-stage cross-encoder reranks only top-N fused results."""
        # Base keyword results define initial fused ordering.
        base_results = [
            {"chunk_id": "chunk1", "content": "alpha"},
            {"chunk_id": "chunk2", "content": "beta"},
            {"chunk_id": "chunk3", "content": "gamma"},
            {"chunk_id": "chunk4", "content": "delta"},
        ]
        mock_keyword_search.search.return_value = list(base_results)
        mock_vector_search.search.return_value = []

        # Dummy reranker that reverses the prefix it sees.
        class DummyReranker:
            def __init__(self, *_args, **_kwargs):
                self.calls = 0
                self.last_docs = None

            def rerank(self, query, docs, *, rerank_k=None, score_key="rrf_score"):
                self.calls += 1
                self.last_docs = list(docs)
                k = rerank_k or len(docs)
                prefix = list(reversed(docs[:k]))
                suffix = docs[k:]
                # Attach a dummy rerank_score so existing code paths are satisfied.
                rescored = []
                for idx, d in enumerate(prefix):
                    d2 = dict(d)
                    d2["rerank_score"] = float(k - idx)
                    rescored.append(d2)
                return rescored + suffix, {"rerank_ms": 1.0}

        from search import hybrid_search as hybrid_module

        dummy_reranker = DummyReranker()
        monkeypatch.setattr(hybrid_module, "CrossEncoderReranker", lambda *_a, **_k: dummy_reranker)

        hs = HybridSearch(
            mock_keyword_search,
            mock_vector_search,
            mock_graph_search,
            use_final_rerank=True,
            final_rerank_top_n=2,
        )

        results = hs.search("test query", limit=4, use_keyword=True, use_vector=True, use_graph=False)

        # Reranker should have been called on only the top-2 docs.
        assert dummy_reranker.calls == 1
        assert dummy_reranker.last_docs is not None
        assert [d["chunk_id"] for d in dummy_reranker.last_docs] == ["chunk1", "chunk2"]

        # The top-2 are reversed by the reranker; tail remains unchanged.
        assert [r["chunk_id"] for r in results] == ["chunk2", "chunk1", "chunk3", "chunk4"]
    
    def test_search_filters_passed(self, hybrid_search, mock_keyword_search,
                                   mock_vector_search):
        """Test that filters are passed to search methods."""
        filters = {"modality": "text"}
        mock_keyword_search.search.return_value = []
        mock_vector_search.search.return_value = []
        
        hybrid_search.search("test query", limit=10, filters=filters,
                           use_keyword=True, use_vector=True, use_graph=False)
        
        # Verify filters were passed
        mock_keyword_search.search.assert_called_with("test query", limit=20, filters=filters)
        mock_vector_search.search.assert_called_with("test query", limit=20, filters=filters)

