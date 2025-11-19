"""Embedding service using OpenAI."""

from typing import List, Dict, Any
import openai
from loguru import logger

from utils.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, EMBEDDING_DIMENSION
from utils.errors import APIError
from utils.errors import retry_with_backoff


class EmbeddingService:
    """Service for generating embeddings using OpenAI."""
    
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_EMBEDDING_MODEL
        self.dimension = EMBEDDING_DIMENSION
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        def _embed():
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        
        try:
            return retry_with_backoff(_embed, max_retries=3)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise APIError(f"Failed to generate embedding: {e}")
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            def _embed_batch():
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                return [item.embedding for item in response.data]
            
            try:
                batch_embeddings = retry_with_backoff(_embed_batch, max_retries=3)
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}")
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                # Add empty embeddings for failed batch
                all_embeddings.extend([[]] * len(batch))
        
        return all_embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension

