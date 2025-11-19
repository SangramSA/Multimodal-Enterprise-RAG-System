"""Shared pytest fixtures for unit tests."""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List
import json

# Mock OpenAI client
@pytest.fixture
def mock_openai_client(mocker):
    """Mock OpenAI client for API calls."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1] * 1536)]
    mock_client.embeddings.create.return_value = mock_response
    
    # Mock chat completions
    mock_chat_response = Mock()
    mock_chat_response.choices = [Mock()]
    mock_chat_response.choices[0].message = Mock()
    mock_chat_response.choices[0].message.content = json.dumps({
        "entities": [],
        "relationships": []
    })
    mock_client.chat.completions.create.return_value = mock_chat_response
    
    return mock_client

# Mock Neo4j driver and session
@pytest.fixture
def mock_neo4j_driver(mocker):
    """Mock Neo4j driver."""
    mock_driver = Mock()
    mock_session = Mock()
    mock_result = Mock()
    mock_record = Mock()
    mock_record.data.return_value = {}
    mock_result.__iter__ = Mock(return_value=iter([mock_record]))
    mock_session.run.return_value = mock_result
    mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = Mock(return_value=None)
    return mock_driver, mock_session

# Mock Qdrant client
@pytest.fixture
def mock_qdrant_client(mocker):
    """Mock Qdrant client."""
    mock_client = Mock()
    mock_collection = Mock()
    mock_client.get_collection.return_value = mock_collection
    mock_client.create_collection.return_value = None
    return mock_client

# Sample test data
@pytest.fixture
def sample_text_content():
    """Sample text content for testing."""
    return """This is a test document about artificial intelligence.
    John Smith works at OpenAI, a company based in San Francisco.
    The company was founded in 2015 and focuses on AI research.
    Machine learning is a key technology in modern AI systems."""

@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            "content": "This is chunk 1",
            "chunk_id": "test_chunk_0",
            "chunk_index": 0,
            "metadata": {
                "file_id": "test_file_1",
                "modality": "text"
            }
        },
        {
            "content": "This is chunk 2",
            "chunk_id": "test_chunk_1",
            "chunk_index": 1,
            "metadata": {
                "file_id": "test_file_1",
                "modality": "text"
            }
        }
    ]

@pytest.fixture
def sample_extraction_result():
    """Sample entity extraction result."""
    return {
        "chunk_id": "test_chunk_0",
        "entities": [
            {
                "name": "John Smith",
                "type": "Person",
                "description": "Works at OpenAI",
                "confidence": 0.9
            },
            {
                "name": "OpenAI",
                "type": "Organization",
                "description": "AI company",
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

@pytest.fixture
def sample_file_metadata():
    """Sample file metadata."""
    return {
        "file_id": "test_file_1",
        "file_name": "test.txt",
        "file_path": "/tmp/test.txt",
        "file_size": 1024,
        "file_type": ".txt",
        "modality": "text",
        "upload_timestamp": "2024-01-01T00:00:00"
    }

@pytest.fixture
def tmp_text_file(tmp_path):
    """Create a temporary text file for testing."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test file with some content.")
    return test_file

@pytest.fixture
def tmp_pdf_file(tmp_path):
    """Create a temporary PDF file for testing (minimal PDF structure)."""
    # Note: This creates a minimal PDF structure
    # For real PDF testing, you'd need actual PDF bytes
    test_file = tmp_path / "test.pdf"
    # Write minimal PDF header (not a real PDF, but enough for some tests)
    test_file.write_bytes(b"%PDF-1.4\n%test\n")
    return test_file

@pytest.fixture
def tmp_image_file(tmp_path):
    """Create a temporary image file for testing."""
    from PIL import Image
    test_file = tmp_path / "test.jpg"
    # Create a simple 100x100 RGB image
    img = Image.new('RGB', (100, 100), color='red')
    img.save(test_file, 'JPEG')
    return test_file

@pytest.fixture
def tmp_audio_file(tmp_path):
    """Create a temporary audio file for testing."""
    test_file = tmp_path / "test.mp3"
    # Write minimal audio file header (not real audio, but enough for some tests)
    test_file.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00")
    return test_file

