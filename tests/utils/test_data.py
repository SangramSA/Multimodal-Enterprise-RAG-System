"""Test data generators."""

from typing import List, Dict, Any
from pathlib import Path


def generate_sample_text(length: int = 1000) -> str:
    """Generate sample text of specified length."""
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    text = " ".join(words * (length // len(" ".join(words))))
    return text[:length]


def generate_sample_chunks(count: int = 5, file_id: str = "test_file") -> List[Dict[str, Any]]:
    """Generate sample chunks for testing."""
    chunks = []
    for i in range(count):
        chunks.append({
            "content": f"This is chunk {i} with some content.",
            "chunk_id": f"{file_id}_chunk_{i}",
            "chunk_index": i,
            "metadata": {
                "file_id": file_id,
                "modality": "text",
                "word_count": 10,
                "character_count": 50
            }
        })
    return chunks


def generate_sample_entities(count: int = 3) -> List[Dict[str, Any]]:
    """Generate sample entities for testing."""
    entity_types = ["Person", "Organization", "Location", "Concept"]
    entities = []
    for i in range(count):
        entities.append({
            "name": f"Entity {i}",
            "type": entity_types[i % len(entity_types)],
            "description": f"Description for entity {i}",
            "confidence": 0.8 + (i * 0.05)
        })
    return entities


def generate_sample_relationships(count: int = 2) -> List[Dict[str, Any]]:
    """Generate sample relationships for testing."""
    rel_types = ["works_for", "located_in", "mentions", "related_to"]
    relationships = []
    for i in range(count):
        relationships.append({
            "source": f"Entity {i}",
            "target": f"Entity {i+1}",
            "relationship_type": rel_types[i % len(rel_types)],
            "description": f"Relationship {i}",
            "confidence": 0.85
        })
    return relationships

