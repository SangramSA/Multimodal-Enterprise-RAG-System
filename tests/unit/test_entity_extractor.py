"""Unit tests for entity extractor."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from extraction.entity_extractor import EntityExtractor, EntityExtractionResult
from utils.errors import APIError
from tests.utils.mock_helpers import create_mock_openai_chat_response


class TestEntityExtractor:
    """Test suite for EntityExtractor."""
    
    @patch('extraction.entity_extractor.openai.OpenAI')
    @patch('extraction.entity_extractor.OPENAI_API_KEY', 'test-key')
    def test_entity_extractor_initialization(self, mock_openai):
        """Test entity extractor initialization."""
        extractor = EntityExtractor()
        assert extractor.model == "gpt-4o"
        mock_openai.assert_called_once_with(api_key='test-key')
    
    @patch('extraction.entity_extractor.OPENAI_API_KEY', None)
    def test_entity_extractor_missing_api_key(self):
        """Test that missing API key raises error."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            EntityExtractor()
    
    @patch('extraction.entity_extractor.openai.OpenAI')
    @patch('extraction.entity_extractor.OPENAI_API_KEY', 'test-key')
    def test_extract_entities_success(self, mock_openai_class):
        """Test successful entity extraction."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock response
        mock_response = create_mock_openai_chat_response({
            "entities": [
                {
                    "name": "John Smith",
                    "type": "Person",
                    "description": "Software engineer",
                    "confidence": 0.9
                }
            ],
            "relationships": []
        })
        mock_client.chat.completions.create.return_value = mock_response
        
        extractor = EntityExtractor()
        result = extractor.extract("John Smith works at OpenAI.")
        
        assert isinstance(result, EntityExtractionResult)
        assert len(result.entities) == 1
        assert result.entities[0].name == "John Smith"
        assert result.entities[0].type == "Person"
        assert result.entities[0].confidence == 0.9
    
    @patch('extraction.entity_extractor.openai.OpenAI')
    @patch('extraction.entity_extractor.OPENAI_API_KEY', 'test-key')
    def test_extract_entities_with_relationships(self, mock_openai_class):
        """Test entity extraction with relationships."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = create_mock_openai_chat_response({
            "entities": [
                {"name": "John", "type": "Person", "confidence": 0.9},
                {"name": "OpenAI", "type": "Organization", "confidence": 0.95}
            ],
            "relationships": [
                {
                    "source": "John",
                    "target": "OpenAI",
                    "relationship_type": "works_for",
                    "confidence": 0.9
                }
            ]
        })
        mock_client.chat.completions.create.return_value = mock_response
        
        extractor = EntityExtractor()
        result = extractor.extract("John works at OpenAI.")
        
        assert len(result.entities) == 2
        assert len(result.relationships) == 1
        assert result.relationships[0].source == "John"
        assert result.relationships[0].target == "OpenAI"
    
    @patch('extraction.entity_extractor.openai.OpenAI')
    @patch('extraction.entity_extractor.OPENAI_API_KEY', 'test-key')
    def test_extract_filters_low_confidence(self, mock_openai_class):
        """Test that low confidence entities are filtered."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = create_mock_openai_chat_response({
            "entities": [
                {"name": "HighConf", "type": "Person", "confidence": 0.9},
                {"name": "LowConf", "type": "Person", "confidence": 0.5}  # Below threshold
            ],
            "relationships": []
        })
        mock_client.chat.completions.create.return_value = mock_response
        
        extractor = EntityExtractor()
        result = extractor.extract("Test text")
        
        assert len(result.entities) == 1
        assert result.entities[0].name == "HighConf"
    
    @patch('extraction.entity_extractor.openai.OpenAI')
    @patch('extraction.entity_extractor.OPENAI_API_KEY', 'test-key')
    def test_extract_with_context(self, mock_openai_class):
        """Test extraction with context metadata."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = create_mock_openai_chat_response({
            "entities": [],
            "relationships": []
        })
        mock_client.chat.completions.create.return_value = mock_response
        
        extractor = EntityExtractor()
        context = {"file_id": "file123", "modality": "text"}
        extractor.extract("Test", context)
        
        # Verify context was included in prompt
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_message = messages[1]["content"]
        assert "file123" in user_message
        assert "text" in user_message
    
    @patch('extraction.entity_extractor.openai.OpenAI')
    @patch('extraction.entity_extractor.OPENAI_API_KEY', 'test-key')
    def test_extract_json_decode_error(self, mock_openai_class):
        """Test handling of JSON decode errors."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock invalid JSON response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Invalid JSON {"
        mock_client.chat.completions.create.return_value = mock_response
        
        extractor = EntityExtractor()
        
        with pytest.raises(APIError, match="Invalid JSON"):
            extractor.extract("Test")
    
    @patch('extraction.entity_extractor.openai.OpenAI')
    @patch('extraction.entity_extractor.OPENAI_API_KEY', 'test-key')
    def test_extract_from_chunks(self, mock_openai_class):
        """Test extraction from multiple chunks."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = create_mock_openai_chat_response({
            "entities": [{"name": "Entity", "type": "Person", "confidence": 0.9}],
            "relationships": []
        })
        mock_client.chat.completions.create.return_value = mock_response
        
        extractor = EntityExtractor()
        chunks = [
            {"content": "Chunk 1", "chunk_id": "chunk1", "file_id": "file1", "metadata": {"modality": "text"}},
            {"content": "Chunk 2", "chunk_id": "chunk2", "file_id": "file1", "metadata": {"modality": "text"}}
        ]
        
        results = extractor.extract_from_chunks(chunks)
        
        assert len(results) == 2
        assert results[0]["chunk_id"] == "chunk1"
        assert results[1]["chunk_id"] == "chunk2"
        assert all("entities" in r for r in results)
    
    @patch('extraction.entity_extractor.openai.OpenAI')
    @patch('extraction.entity_extractor.OPENAI_API_KEY', 'test-key')
    def test_extract_from_chunks_error_handling(self, mock_openai_class):
        """Test error handling in extract_from_chunks."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # First call succeeds, second fails
        mock_response1 = create_mock_openai_chat_response({
            "entities": [],
            "relationships": []
        })
        mock_client.chat.completions.create.side_effect = [
            mock_response1,
            Exception("API Error")
        ]
        
        extractor = EntityExtractor()
        chunks = [
            {"content": "Chunk 1", "chunk_id": "chunk1", "file_id": "file1", "metadata": {"modality": "text"}},
            {"content": "Chunk 2", "chunk_id": "chunk2", "file_id": "file1", "metadata": {"modality": "text"}}
        ]
        
        results = extractor.extract_from_chunks(chunks)
        
        assert len(results) == 2
        assert "error" in results[1]
    
    def test_link_entities_across_modalities(self):
        """Test linking entities across modalities."""
        extractor = EntityExtractor()
        
        extraction_results = [
            {
                "chunk_id": "chunk1",
                "entities": [
                    {"name": "John Smith", "type": "Person"},
                    {"name": "OpenAI", "type": "Organization"}
                ]
            },
            {
                "chunk_id": "chunk2",
                "entities": [
                    {"name": "John Smith", "type": "Person"}  # Same entity
                ]
            }
        ]
        
        chunks = [
            {
                "chunk_id": "chunk1",
                "metadata": {"file_id": "file1", "modality": "text"}
            },
            {
                "chunk_id": "chunk2",
                "metadata": {"file_id": "file2", "modality": "image"}
            }
        ]
        
        entity_links = extractor.link_entities_across_modalities(extraction_results, chunks)
        
        assert "john smith" in entity_links
        assert len(entity_links["john smith"]) == 2
        assert entity_links["john smith"][0]["file_id"] == "file1"
        assert entity_links["john smith"][1]["file_id"] == "file2"
    
    def test_link_entities_normalizes_names(self):
        """Test that entity names are normalized (lowercase, stripped)."""
        extractor = EntityExtractor()
        
        extraction_results = [
            {
                "chunk_id": "chunk1",
                "entities": [
                    {"name": "  John Smith  ", "type": "Person"},
                    {"name": "JOHN SMITH", "type": "Person"}
                ]
            }
        ]
        
        chunks = [
            {"chunk_id": "chunk1", "metadata": {"file_id": "file1", "modality": "text"}}
        ]
        
        entity_links = extractor.link_entities_across_modalities(extraction_results, chunks)
        
        # Both should map to same normalized key
        assert "john smith" in entity_links
        assert len(entity_links["john smith"]) == 2

