"""Vector store for indexing and retrieving chunks."""

from typing import List, Dict, Any, Optional, Tuple
import hashlib
import time
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from loguru import logger

from vector.qdrant_client import QdrantClientWrapper
from vector.embedding_service import EmbeddingService


class VectorStore:
    """Store and retrieve vector embeddings."""
    
    def __init__(self, qdrant_client: Optional[QdrantClientWrapper] = None,
                 embedding_service: Optional[EmbeddingService] = None):
        self.qdrant = qdrant_client or QdrantClientWrapper()
        self.embeddings = embedding_service or EmbeddingService()
    
    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Index chunks in vector database.
        
        Args:
            chunks: List of chunks with content and metadata
        
        Returns:
            Number of chunks indexed
        """
        if not chunks:
            return 0
        
        # Extract texts for embedding
        texts = [chunk.get("content", "") for chunk in chunks]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embeddings.embed_batch(texts)
        
        # Create points
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if not embedding:
                logger.warning(f"Skipping chunk {i} due to embedding failure")
                continue
            
            chunk_id = chunk.get("chunk_id") or f"chunk_{i}"
            
            # Convert chunk_id string to integer for Qdrant (deterministic hash)
            # Qdrant requires point IDs to be unsigned integers or UUIDs
            point_id = int(hashlib.md5(chunk_id.encode('utf-8')).hexdigest(), 16) % (2**63)
            # Use modulo to ensure it fits in signed 64-bit integer range that Qdrant accepts
            # Qdrant accepts signed 64-bit integers, but we'll use positive values
            
            # Prepare payload (metadata)
            # Store all metadata in payload for easy retrieval
            chunk_metadata = chunk.get("metadata", {})
            payload = {
                "content": chunk.get("content", ""),
                "chunk_id": chunk_id,  # Keep original chunk_id in payload for retrieval
                "file_id": chunk.get("file_id"),
                "chunk_index": chunk.get("chunk_index", i),
                "modality": chunk_metadata.get("modality"),
                "domain_tags": chunk_metadata.get("domain_tags", []),
                "timestamp": chunk.get("processing_timestamp"),
                "entity_ids": chunk_metadata.get("entity_ids", []),
                "related_file_ids": chunk_metadata.get("related_file_ids", []),
                "metadata": chunk_metadata  # Store full metadata for retrieval
            }
            
            # Add chunk-specific metadata at top level for filtering
            for key in ["page_number", "start_time", "end_time", "language", "confidence"]:
                if key in chunk_metadata:
                    payload[key] = chunk_metadata[key]
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            ))
        
        # Upsert points
        if points:
            success = self.qdrant.upsert_points(points)
            if success:
                logger.success(f"Indexed {len(points)} chunks")
                return len(points)
            else:
                logger.error("Failed to index chunks")
                return 0
        
        return 0
    
    def search(self, query: str, limit: int = 10, 
               filters: Optional[Dict[str, Any]] = None,
               score_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            filters: Optional metadata filters (e.g., {"modality": "text", "domain_tags": ["finance"]})
            score_threshold: Minimum similarity score
        
        Returns:
            List of matching chunks with scores
        """
        results, _timings = self.search_with_timings(
            query=query,
            limit=limit,
            filters=filters,
            score_threshold=score_threshold,
        )
        return results

    def search_with_timings(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.25,
        *,
        query_embedding: Optional[List[float]] = None,
        fetch_multiplier: int = 2,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Search for similar chunks and return per-stage timing breakdown.

        Timings are returned in milliseconds.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            filters: Optional metadata filters (e.g., {"modality": "text", "domain_tags": ["finance"]})
            score_threshold: Minimum similarity score
            query_embedding: Optional precomputed embedding to avoid re-embedding
            fetch_multiplier: Multiplier for initial Qdrant fetch to allow post-filtering

        Returns:
            (formatted_results, timings_ms)
        """
        timings_ms: Dict[str, float] = {
            "openai_embed_ms": 0.0,
            "qdrant_search_ms": 0.0,
            "post_filter_ms": 0.0,
            "format_results_ms": 0.0,
        }

        # 1) Embedding
        if query_embedding is None:
            t0 = time.perf_counter()
            query_embedding = self.embeddings.embed_text(query)
            timings_ms["openai_embed_ms"] = (time.perf_counter() - t0) * 1000.0

        # 2) Build Qdrant filter (server-side)
        qdrant_filter: Optional[Filter] = None
        if filters:
            conditions: List[FieldCondition] = []

            # Most common filters we support server-side
            if "modality" in filters and filters["modality"] is not None:
                conditions.append(
                    FieldCondition(key="modality", match=MatchValue(value=filters["modality"]))
                )
            if "file_id" in filters and filters["file_id"] is not None:
                conditions.append(
                    FieldCondition(key="file_id", match=MatchValue(value=filters["file_id"]))
                )

            # NOTE: domain_tags is stored as an array in payload; Qdrant filtering for arrays
            # depends on collection schema / match operators, so we apply it client-side below.

            if conditions:
                qdrant_filter = Filter(must=conditions)

        # 3) Qdrant search
        fetch_limit = max(limit * fetch_multiplier, limit)
        t1 = time.perf_counter()
        results = self.qdrant.search(
            query_vector=query_embedding,
            limit=fetch_limit,
            filter=qdrant_filter,
            score_threshold=score_threshold,
        )
        timings_ms["qdrant_search_ms"] = (time.perf_counter() - t1) * 1000.0

        # 4) Post-filtering (domain tags)
        t2 = time.perf_counter()
        if filters and "domain_tags" in filters and filters["domain_tags"] is not None:
            domain_tags = filters["domain_tags"]
            if isinstance(domain_tags, str):
                domain_tags = [domain_tags]

            filtered_results = []
            for result in results:
                result_tags = result.get("payload", {}).get("domain_tags", [])
                if any(tag in result_tags for tag in domain_tags):
                    filtered_results.append(result)
            results = filtered_results
        timings_ms["post_filter_ms"] = (time.perf_counter() - t2) * 1000.0

        # 5) Format + trim
        t3 = time.perf_counter()
        results = results[:limit]
        formatted_results: List[Dict[str, Any]] = []
        for result in results:
            payload = result.get("payload", {}) or {}
            formatted_results.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "content": payload.get("content"),
                    "score": result.get("score"),
                    "metadata": {k: v for k, v in payload.items() if k not in ["content", "chunk_id"]},
                }
            )
        timings_ms["format_results_ms"] = (time.perf_counter() - t3) * 1000.0

        return formatted_results, timings_ms
    
    def _chunk_id_to_point_id(self, chunk_id: str) -> int:
        """Convert chunk_id string to Qdrant point ID (integer)."""
        return int(hashlib.md5(chunk_id.encode('utf-8')).hexdigest(), 16) % (2**63)
    
    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """Delete chunks by IDs."""
        # Convert chunk_ids to point_ids
        point_ids = [self._chunk_id_to_point_id(chunk_id) for chunk_id in chunk_ids]
        return self.qdrant.delete_points(point_ids)
    
    def get_chunks_by_file_id(self, file_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a given file_id from Qdrant.
        
        Args:
            file_id: The file ID to retrieve chunks for
        
        Returns:
            List of chunks with their metadata, sorted by chunk_index
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Create filter for file_id
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key="file_id", match=MatchValue(value=file_id))
                ]
            )
            
            # Use a dummy vector for search (we're filtering by metadata anyway)
            # We need to use search because Qdrant doesn't have a direct "get all by filter" method
            dummy_vector = [0.0] * self.embeddings.get_dimension()
            results = self.qdrant.search(
                query_vector=dummy_vector,
                limit=1000,  # Get all chunks for this file
                filter=qdrant_filter,
                score_threshold=0.0
            )
            
            # Format results and sort by chunk_index
            chunks = []
            for result in results:
                payload = result.get("payload", {})
                # Get chunk_index from payload (stored at top level)
                chunk_index = payload.get("chunk_index", 0)
                
                # Get metadata - prefer nested metadata, fallback to reconstructing from payload
                metadata = payload.get("metadata", {})
                if not isinstance(metadata, dict):
                    # Reconstruct metadata from payload
                    metadata = {
                        k: v for k, v in payload.items()
                        if k not in ["content", "chunk_id", "chunk_index", "file_id", "metadata"]
                    }
                
                chunks.append({
                    "chunk_id": payload.get("chunk_id"),
                    "chunk_index": chunk_index,
                    "content": payload.get("content", ""),
                    "metadata": metadata
                })
            
            # Sort by chunk_index to maintain order
            chunks.sort(key=lambda x: x.get("chunk_index", 0))
            
            return chunks
        except Exception as e:
            logger.warning(f"Failed to retrieve chunks for file_id '{file_id}': {e}")
            return []
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a chunk by ID."""
        point_id = self._chunk_id_to_point_id(chunk_id)
        point = self.qdrant.get_point(point_id)
        if point:
            return {
                "chunk_id": point.get("payload", {}).get("chunk_id", chunk_id),  # Use original chunk_id from payload
                "content": point.get("payload", {}).get("content"),
                "metadata": {
                    k: v for k, v in point.get("payload", {}).items()
                    if k not in ["content", "chunk_id"]
                }
            }
        return None

