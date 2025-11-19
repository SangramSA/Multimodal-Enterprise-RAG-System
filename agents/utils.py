"""Utility functions for agent operations."""

from typing import List, Dict, Any, Optional
from loguru import logger

from extraction.entity_extractor import EntityExtractor


# Global entity extractor instance (lazy initialization)
_entity_extractor: Optional[EntityExtractor] = None


def get_entity_extractor() -> EntityExtractor:
    """Get or create entity extractor instance."""
    global _entity_extractor
    if _entity_extractor is None:
        _entity_extractor = EntityExtractor()
    return _entity_extractor


def extract_entities_from_query(query: str) -> List[Dict[str, Any]]:
    """
    Extract entities from a natural language query.
    
    Args:
        query: Natural language query string
    
    Returns:
        List of extracted entities with name and type
    """
    try:
        extractor = get_entity_extractor()
        result = extractor.extract(query)
        
        entities = []
        for entity in result.entities:
            entities.append({
                "name": entity.name,
                "type": entity.type,
                "description": entity.description,
                "confidence": entity.confidence
            })
        
        return entities
    except Exception as e:
        logger.warning(f"Entity extraction from query failed: {e}")
        # Fallback: simple keyword extraction
        return _simple_keyword_extraction(query)


def _simple_keyword_extraction(query: str) -> List[Dict[str, Any]]:
    """
    Simple fallback entity extraction using keywords.
    
    Args:
        query: Query string
    
    Returns:
        List of potential entities (keywords)
    """
    # Extract capitalized words and phrases as potential entities
    words = query.split()
    entities = []
    
    for i, word in enumerate(words):
        # Check if word is capitalized (potential entity)
        if word and word[0].isupper() and len(word) > 2:
            # Check if it's part of a multi-word entity
            entity_name = word
            if i + 1 < len(words) and words[i + 1][0].isupper():
                entity_name = f"{word} {words[i + 1]}"
            
            entities.append({
                "name": entity_name,
                "type": "Concept",  # Default type
                "description": None,
                "confidence": 0.5  # Lower confidence for fallback
            })
    
    return entities


def format_graph_results(results: List[Dict[str, Any]], search_type: str) -> str:
    """
    Format graph search results for agent consumption.
    
    Args:
        results: List of search results
        search_type: Type of search performed
    
    Returns:
        Formatted string representation
    """
    if not results:
        return f"No results found for {search_type} search."
    
    formatted_parts = [f"Found {len(results)} result(s) from {search_type} search:\n"]
    
    for i, result in enumerate(results[:10], 1):  # Limit to top 10
        if search_type == "entity" or search_type == "content":
            file_name = result.get("file_name", "Unknown")
            modality = result.get("modality", "unknown")
            score = result.get("score", 0.0)
            formatted_parts.append(f"{i}. {file_name} ({modality}) - Score: {score:.3f}")
        
        elif search_type == "relationship":
            entity_name = result.get("entity_name", "Unknown")
            entity_type = result.get("entity_type", "Unknown")
            rel_type = result.get("relationship_type", "Unknown")
            formatted_parts.append(f"{i}. {entity_name} ({entity_type}) - Relationship: {rel_type}")
        
        elif search_type == "path":
            path_length = result.get("path_length", 0)
            formatted_parts.append(f"{i}. Path found with length: {path_length}")
        
        else:
            # Generic formatting
            formatted_parts.append(f"{i}. {str(result)[:100]}")
    
    return "\n".join(formatted_parts)


def format_search_results(results: List[Dict[str, Any]], search_method: str) -> str:
    """
    Format search results for agent consumption.
    
    Args:
        results: List of search results
        search_method: Search method used (keyword, vector, hybrid)
    
    Returns:
        Formatted string representation
    """
    if not results:
        return f"No results found from {search_method} search."
    
    formatted_parts = [f"Found {len(results)} result(s) from {search_method} search:\n"]
    
    for i, result in enumerate(results[:10], 1):  # Limit to top 10
        content = result.get("content", "")[:200]  # Truncate content
        score = result.get("rrf_score", result.get("score", result.get("keyword_score", 0.0)))
        chunk_id = result.get("chunk_id", "unknown")
        
        formatted_parts.append(
            f"{i}. [Chunk: {chunk_id}] Score: {score:.3f}\n"
            f"   Content: {content}..."
        )
    
    return "\n".join(formatted_parts)


def normalize_entity_name(name: str) -> str:
    """
    Normalize entity name for matching.
    
    Args:
        name: Entity name to normalize
    
    Returns:
        Normalized name (lowercase, trimmed, single spaces)
    """
    return " ".join(name.lower().strip().split())


def calculate_confidence_from_scores(scores: List[float], method: str = "average") -> float:
    """
    Calculate overall confidence from multiple scores.
    
    Args:
        scores: List of confidence scores
        method: Calculation method ("average", "max", "min", "weighted")
    
    Returns:
        Overall confidence score (0-1)
    """
    if not scores:
        return 0.0
    
    if method == "average":
        return sum(scores) / len(scores)
    elif method == "max":
        return max(scores)
    elif method == "min":
        return min(scores)
    elif method == "weighted":
        # Weighted average (higher scores weighted more)
        weights = [s ** 2 for s in scores]  # Square to emphasize high scores
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    else:
        return sum(scores) / len(scores)  # Default to average

