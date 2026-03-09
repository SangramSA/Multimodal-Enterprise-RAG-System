"""Qdrant client for vector database operations."""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from loguru import logger

from utils.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, EMBEDDING_DIMENSION
from utils.errors import DatabaseError
from utils.errors import retry_with_backoff


class QdrantClientWrapper:
    """Wrapper for Qdrant client operations."""
    
    def __init__(self):
        self.url = QDRANT_URL
        self.api_key = QDRANT_API_KEY if QDRANT_API_KEY else None
        self.collection_name = QDRANT_COLLECTION_NAME
        self.dimension = EMBEDDING_DIMENSION
        self.client: Optional[QdrantClient] = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Qdrant."""
        try:
            if self.api_key:
                self.client = QdrantClient(url=self.url, api_key=self.api_key)
            else:
                self.client = QdrantClient(url=self.url)
            
            # Verify connection
            collections = self.client.get_collections()
            logger.success("Connected to Qdrant")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise DatabaseError(f"Qdrant connection failed: {e}")
    
    def ensure_collection(self):
        """Ensure collection exists, create if not."""
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.success(f"Collection '{self.collection_name}' created")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise DatabaseError(f"Collection creation failed: {e}")
    
    def upsert_points(self, points: List[PointStruct]) -> bool:
        """Upsert points into collection."""
        def _upsert():
            self.ensure_collection()
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return True
        
        try:
            return retry_with_backoff(_upsert, max_retries=3)
        except Exception as e:
            logger.error(f"Failed to upsert points: {e}")
            return False
    
    def search(self, query_vector: List[float], limit: int = 10, 
               filter: Optional[Filter] = None, score_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        def _search():
            self.ensure_collection()
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filter,
                score_threshold=score_threshold
            )
            return results
        
        try:
            results = retry_with_backoff(_search, max_retries=3)
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete_points(self, point_ids: List[int]) -> bool:
        """Delete points by IDs."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete points: {e}")
            return False
    
    def get_point(self, point_id: int) -> Optional[Dict[str, Any]]:
        """Get a point by ID."""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )
            if result:
                point = result[0]
                return {
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get point: {e}")
            return None
    
    def health_check(self) -> bool:
        """Check if database is accessible."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

