"""Vector search using semantic similarity."""

from typing import List, Dict, Any, Optional
from loguru import logger

from vector.vector_store import VectorStore


class VectorSearch:
    """Semantic vector search."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def search(self, query: str, limit: int = 10,
              filters: Optional[Dict[str, Any]] = None,
              score_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Search using semantic similarity.
        
        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional metadata filters
            score_threshold: Minimum similarity score
        
        Returns:
            List of matching documents with similarity scores
        """
        results = self.vector_store.search(
            query=query,
            limit=limit,
            filters=filters,
            score_threshold=score_threshold
        )
        
        # Results already have scores from vector store
        return results

