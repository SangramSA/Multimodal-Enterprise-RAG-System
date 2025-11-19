"""Helper functions for creating mock responses."""

from typing import Dict, Any, List
from unittest.mock import Mock
import json


def create_mock_openai_embedding_response(embedding_dim: int = 1536) -> Mock:
    """Create a mock OpenAI embedding response."""
    mock_response = Mock()
    mock_data = Mock()
    mock_data.embedding = [0.1] * embedding_dim
    mock_response.data = [mock_data]
    return mock_response


def create_mock_openai_chat_response(content: Dict[str, Any]) -> Mock:
    """Create a mock OpenAI chat completion response."""
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = json.dumps(content) if isinstance(content, dict) else content
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    return mock_response


def create_mock_neo4j_result(records: List[Dict[str, Any]]) -> Mock:
    """Create a mock Neo4j query result."""
    mock_result = Mock()
    mock_records = []
    for record_data in records:
        mock_record = Mock()
        mock_record.data.return_value = record_data
        mock_records.append(mock_record)
    mock_result.__iter__ = Mock(return_value=iter(mock_records))
    return mock_result


def create_mock_qdrant_search_result(ids: List[int], scores: List[float], payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create a mock Qdrant search result."""
    return [
        {
            "id": id_val,
            "score": score,
            "payload": payload
        }
        for id_val, score, payload in zip(ids, scores, payloads)
    ]


def create_sample_entity_extraction_response() -> Dict[str, Any]:
    """Create a sample entity extraction response."""
    return {
        "entities": [
            {
                "name": "John Smith",
                "type": "Person",
                "description": "Software engineer",
                "confidence": 0.9
            },
            {
                "name": "OpenAI",
                "type": "Organization",
                "description": "AI research company",
                "confidence": 0.95
            }
        ],
        "relationships": [
            {
                "source": "John Smith",
                "target": "OpenAI",
                "relationship_type": "works_for",
                "description": "Employee",
                "confidence": 0.9
            }
        ]
    }

