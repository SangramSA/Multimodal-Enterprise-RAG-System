"""Unit tests for domain classifier."""

import pytest
import json
from unittest.mock import Mock, patch
from extraction.domain_classifier import DomainClassifier
from utils.errors import APIError
from tests.utils.mock_helpers import create_mock_openai_chat_response


class TestDomainClassifier:
    """Test suite for DomainClassifier."""
    
    @patch('extraction.domain_classifier.openai.OpenAI')
    @patch('extraction.domain_classifier.OPENAI_API_KEY', 'test-key')
    def test_domain_classifier_initialization(self, mock_openai):
        """Test domain classifier initialization."""
        classifier = DomainClassifier()
        assert classifier.model == "gpt-4o"
        assert len(classifier.domain_tags) > 0
        mock_openai.assert_called_once_with(api_key='test-key')
    
    @patch('extraction.domain_classifier.OPENAI_API_KEY', None)
    def test_domain_classifier_missing_api_key(self):
        """Test that missing API key raises error."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            DomainClassifier()
    
    @patch('extraction.domain_classifier.openai.OpenAI')
    @patch('extraction.domain_classifier.OPENAI_API_KEY', 'test-key')
    def test_classify_success(self, mock_openai_class):
        """Test successful domain classification."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = create_mock_openai_chat_response({
            "domains": [
                {"tag": "technical", "confidence": 0.9},
                {"tag": "finance", "confidence": 0.7}
            ],
            "reasoning": "Technical content with financial aspects"
        })
        mock_client.chat.completions.create.return_value = mock_response
        
        classifier = DomainClassifier()
        result = classifier.classify("This is about machine learning and financial markets.")
        
        assert len(result) == 2
        assert result[0]["tag"] == "technical"
        assert result[0]["confidence"] == 0.9
        assert result[1]["tag"] == "finance"
    
    @patch('extraction.domain_classifier.openai.OpenAI')
    @patch('extraction.domain_classifier.OPENAI_API_KEY', 'test-key')
    def test_classify_filters_low_confidence(self, mock_openai_class):
        """Test that low confidence domains are filtered."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = create_mock_openai_chat_response({
            "domains": [
                {"tag": "technical", "confidence": 0.9},
                {"tag": "finance", "confidence": 0.5}  # Below threshold
            ],
            "reasoning": "Test"
        })
        mock_client.chat.completions.create.return_value = mock_response
        
        classifier = DomainClassifier()
        result = classifier.classify("Test text")
        
        assert len(result) == 1
        assert result[0]["tag"] == "technical"
    
    @patch('extraction.domain_classifier.openai.OpenAI')
    @patch('extraction.domain_classifier.OPENAI_API_KEY', 'test-key')
    def test_classify_with_entities(self, mock_openai_class):
        """Test classification with entity context."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = create_mock_openai_chat_response({
            "domains": [{"tag": "medical", "confidence": 0.9}],
            "reasoning": "Medical entities found"
        })
        mock_client.chat.completions.create.return_value = mock_response
        
        classifier = DomainClassifier()
        entities = [
            {"type": "Person", "name": "Dr. Smith"},
            {"type": "Concept", "name": "Diagnosis"}
        ]
        result = classifier.classify("Medical text", entities=entities)
        
        # Verify entities were included in prompt
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_message = messages[1]["content"]
        assert "Entity types found" in user_message
    
    @patch('extraction.domain_classifier.openai.OpenAI')
    @patch('extraction.domain_classifier.OPENAI_API_KEY', 'test-key')
    def test_classify_with_document_structure(self, mock_openai_class):
        """Test classification with document structure."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = create_mock_openai_chat_response({
            "domains": [{"tag": "legal", "confidence": 0.9}],
            "reasoning": "Legal document structure"
        })
        mock_client.chat.completions.create.return_value = mock_response
        
        classifier = DomainClassifier()
        structure = {
            "title": "Legal Agreement",
            "headers": ["Section 1", "Section 2"]
        }
        result = classifier.classify("Legal text", document_structure=structure)
        
        # Verify structure was included in prompt
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_message = messages[1]["content"]
        assert "Legal Agreement" in user_message
    
    @patch('extraction.domain_classifier.openai.OpenAI')
    @patch('extraction.domain_classifier.OPENAI_API_KEY', 'test-key')
    def test_classify_chunk(self, mock_openai_class):
        """Test classify_chunk method."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = create_mock_openai_chat_response({
            "domains": [{"tag": "technical", "confidence": 0.9}],
            "reasoning": "Technical content"
        })
        mock_client.chat.completions.create.return_value = mock_response
        
        classifier = DomainClassifier()
        chunk = {
            "content": "Technical content",
            "chunk_id": "chunk1",
            "metadata": {}
        }
        entities = [{"type": "Concept", "name": "AI"}]
        
        result = classifier.classify_chunk(chunk, entities)
        
        assert isinstance(result, list)
        assert len(result) > 0

