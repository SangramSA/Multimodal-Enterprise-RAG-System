"""Keyword search using BM25."""

from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from loguru import logger

from vector.vector_store import VectorStore


class KeywordSearch:
    """Keyword-based search using BM25."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.index: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []
        self._build_index()
    
    def _build_index(self):
        """Build BM25 index from all documents in vector store."""
        # Note: In a production system, you'd maintain this index separately
        # For now, we'll build it on-demand or maintain it in memory
        # This is a simplified implementation
        logger.info("Building BM25 index...")
        # Index will be built dynamically during search
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        return text.lower().split()
    
    def search(self, query: str, limit: int = 10, 
              filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search using keyword matching.
        
        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional metadata filters
        
        Returns:
            List of matching documents with scores
        """
        # Get candidate documents from vector store (using a broad search)
        # In practice, you'd maintain a separate keyword index
        # For now, we'll use vector store to get candidates and then rank by keywords
        
        # Get candidates with timing breakdown for observability, if available
        timings_ms = {"total_ms": 0.0}
        if isinstance(self.vector_store, VectorStore):
            candidates, timings_ms = self.vector_store.search_with_timings(
                query=query,
                limit=limit * 3,
                filters=filters,
                score_threshold=0.0,
            )
        else:
            # Test paths and alternative implementations fall back to basic search
            candidates = self.vector_store.search(query, limit=limit * 3, filters=filters)
        
        if not candidates:
            return []
        
        # Build BM25 index from candidates
        tokenized_docs = [self._tokenize(doc.get("content", "")) for doc in candidates]
        bm25 = BM25Okapi(tokenized_docs)
        
        # Score query
        query_tokens = self._tokenize(query)
        scores = bm25.get_scores(query_tokens)
        
        # Combine scores with candidates
        results = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            if score > 0:  # Only include documents with matches
                results.append({
                    **candidate,
                    "keyword_score": float(score),
                    "combined_score": float(score),  # For now, just keyword score
                    "keyword_latency_ms": timings_ms.get("total_ms", 0.0),
                })
        
        # Sort by score and limit
        results.sort(key=lambda x: x.get("keyword_score", 0), reverse=True)
        return results[:limit]
    
    def filter_by_metadata(self, results: List[Dict[str, Any]], 
                           filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter results by metadata."""
        filtered = []
        
        for result in results:
            metadata = result.get("metadata", {})
            match = True
            
            if "modality" in filters and metadata.get("modality") != filters["modality"]:
                match = False
            
            if "domain_tags" in filters:
                result_tags = metadata.get("domain_tags", [])
                filter_tags = filters["domain_tags"]
                if isinstance(filter_tags, str):
                    filter_tags = [filter_tags]
                if not any(tag in result_tags for tag in filter_tags):
                    match = False
            
            if match:
                filtered.append(result)
        
        return filtered

