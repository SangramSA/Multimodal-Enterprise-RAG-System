"""Caching layer for DeepEval metric evaluations to avoid redundant API calls."""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from utils.config import LOGS_DIR


class DeepEvalCache:
    """Cache manager for DeepEval metric results."""
    
    def __init__(self, cache_file: Optional[Path] = None, enabled: bool = True):
        """
        Initialize the cache.
        
        Args:
            cache_file: Path to cache file (default: logs/deepeval_cache.json)
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        self.cache_file = cache_file or LOGS_DIR / "deepeval_cache.json"
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        if self.enabled:
            self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
                logger.debug(f"Loaded {len(self.cache)} cached DeepEval results from {self.cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, starting with empty cache")
                self.cache = {}
        else:
            self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        if not self.enabled:
            return
        
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f, indent=2)
            logger.debug(f"Saved {len(self.cache)} cached DeepEval results to {self.cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _generate_cache_key(
        self,
        input_text: str,
        actual_output: str,
        expected_output: Optional[str] = None,
        retrieval_context: Optional[list] = None,
        ground_truths: Optional[list] = None
    ) -> str:
        """
        Generate a cache key from the evaluation inputs.
        
        Args:
            input_text: The input query
            actual_output: The generated answer
            expected_output: Expected answer (optional)
            retrieval_context: List of retrieved context strings (optional)
            ground_truths: List of ground truth relevant documents (optional)
        
        Returns:
            MD5 hash of the inputs
        """
        # Normalize inputs for consistent hashing
        normalized = {
            "input": input_text.strip(),
            "actual": actual_output.strip(),
            "expected": expected_output.strip() if expected_output else "",
            "retrieval_context": json.dumps(sorted([str(c).strip() for c in (retrieval_context or [])]), sort_keys=True),
            "ground_truths": json.dumps(sorted([str(g).strip() for g in (ground_truths or [])]), sort_keys=True)
        }
        
        # Create a deterministic string representation
        cache_string = json.dumps(normalized, sort_keys=True)
        
        # Generate MD5 hash
        return hashlib.md5(cache_string.encode("utf-8")).hexdigest()
    
    def get(
        self,
        input_text: str,
        actual_output: str,
        expected_output: Optional[str] = None,
        retrieval_context: Optional[list] = None,
        ground_truths: Optional[list] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result if available.
        
        Args:
            input_text: The input query
            actual_output: The generated answer
            expected_output: Expected answer (optional)
            retrieval_context: List of retrieved context strings (optional)
            ground_truths: List of ground truth relevant documents (optional)
        
        Returns:
            Cached result dict or None if not found
        """
        if not self.enabled:
            return None
        
        cache_key = self._generate_cache_key(input_text, actual_output, expected_output, retrieval_context, ground_truths)
        result = self.cache.get(cache_key)
        
        if result:
            logger.debug(f"Cache hit for DeepEval evaluation (key: {cache_key[:8]}...)")
            return result
        
        logger.debug(f"Cache miss for DeepEval evaluation (key: {cache_key[:8]}...)")
        return None
    
    def set(
        self,
        input_text: str,
        actual_output: str,
        result: Dict[str, Any],
        expected_output: Optional[str] = None,
        retrieval_context: Optional[list] = None,
        ground_truths: Optional[list] = None
    ):
        """
        Store result in cache.
        
        Args:
            input_text: The input query
            actual_output: The generated answer
            result: The evaluation result to cache
            expected_output: Expected answer (optional)
            retrieval_context: List of retrieved context strings (optional)
            ground_truths: List of ground truth relevant documents (optional)
        """
        if not self.enabled:
            return
        
        cache_key = self._generate_cache_key(input_text, actual_output, expected_output, retrieval_context, ground_truths)
        self.cache[cache_key] = result
        self._save_cache()
        logger.debug(f"Cached DeepEval result (key: {cache_key[:8]}...)")
    
    def clear(self):
        """Clear the cache."""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("DeepEval cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "enabled": self.enabled,
            "cache_file": str(self.cache_file),
            "cached_entries": len(self.cache),
            "cache_size_mb": self.cache_file.stat().st_size / (1024 * 1024) if self.cache_file.exists() else 0
        }


# Global cache instance
_cache_instance: Optional[DeepEvalCache] = None


def get_cache(enabled: Optional[bool] = None) -> DeepEvalCache:
    """
    Get or create the global cache instance.
    
    Args:
        enabled: Override cache enabled state (uses env var if None)
    
    Returns:
        DeepEvalCache instance
    """
    global _cache_instance
    
    if _cache_instance is None or enabled is not None:
        import os
        cache_enabled = enabled if enabled is not None else os.getenv("DEEPEVAL_CACHE_ENABLED", "true").lower() in {"1", "true", "yes"}
        _cache_instance = DeepEvalCache(enabled=cache_enabled)
    
    return _cache_instance

