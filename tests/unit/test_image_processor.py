"""Unit tests for ImageProcessor."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import base64

from ingestion.image_processor import ImageProcessor
from utils.errors import APIError, ProcessingError


class TestImageProcessor:
    """Test suite for ImageProcessor."""
    
    @pytest.fixture
    def image_processor(self, mocker):
        """Create ImageProcessor instance with mocked OpenAI client."""
        with patch('ingestion.image_processor.OPENAI_API_KEY', 'test-key'):
            processor = ImageProcessor()
            processor.client = Mock()
            return processor
    
    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create a sample image file."""
        img = Image.new('RGB', (100, 100), color='red')
        img_path = tmp_path / "test.jpg"
        img.save(img_path, 'JPEG')
        return img_path
    
    @pytest.fixture
    def mock_vision_response(self):
        """Mock GPT-4 Vision API response."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        # content should be a string, not a list
        mock_message.content = "CAPTION: This is a test image showing a red square.\nTEXT: No text found"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 100
        return mock_response
    
    def test_init_without_api_key(self, mocker):
        """Test initialization fails without API key."""
        with patch('ingestion.image_processor.OPENAI_API_KEY', None):
            with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
                ImageProcessor()
    
    def test_supported_extensions(self, image_processor):
        """Test supported file extensions."""
        assert ".jpg" in image_processor.SUPPORTED_EXTENSIONS
        assert ".jpeg" in image_processor.SUPPORTED_EXTENSIONS
        assert ".png" in image_processor.SUPPORTED_EXTENSIONS
    
    def test_validate_image_success(self, image_processor, sample_image):
        """Test successful image validation."""
        result = image_processor._validate_image(sample_image)
        
        assert "width" in result
        assert "height" in result
        assert "format" in result
        assert "image" in result
        assert result["width"] == 100
        assert result["height"] == 100
    
    def test_validate_image_invalid_file(self, image_processor, tmp_path):
        """Test validation with invalid image file."""
        invalid_file = tmp_path / "invalid.jpg"
        invalid_file.write_bytes(b"not an image")
        
        with pytest.raises(ProcessingError, match="Invalid or corrupted image"):
            image_processor._validate_image(invalid_file)
    
    def test_resize_if_needed_small_image(self, image_processor):
        """Test resize when image is already small enough."""
        img = Image.new('RGB', (100, 100), color='blue')
        resized = image_processor._resize_if_needed(img)
        
        assert resized.size == (100, 100)
        assert resized is img  # Should return same image
    
    def test_resize_if_needed_large_image(self, image_processor):
        """Test resize when image is too large."""
        img = Image.new('RGB', (5000, 5000), color='blue')
        resized = image_processor._resize_if_needed(img)
        
        assert resized.size[0] <= image_processor.MAX_IMAGE_DIMENSION
        assert resized.size[1] <= image_processor.MAX_IMAGE_DIMENSION
        # Aspect ratio should be maintained
        assert abs(resized.size[0] / resized.size[1] - 1.0) < 0.01
    
    def test_encode_image(self, image_processor):
        """Test image encoding to base64."""
        img = Image.new('RGB', (100, 100), color='green')
        encoded = image_processor._encode_image(img, "JPEG")
        
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        # Should be valid base64
        decoded = base64.b64decode(encoded)
        assert len(decoded) > 0
    
    @patch('ingestion.image_processor.TESSERACT_AVAILABLE', True)
    @patch('ingestion.image_processor.pytesseract')
    def test_ocr_with_tesseract_success(self, mock_pytesseract, image_processor, sample_image):
        """Test successful Tesseract OCR."""
        mock_pytesseract.image_to_string.return_value = "Extracted text from image"
        mock_pytesseract.image_to_data.return_value = "data with confidence"
        
        result = image_processor._ocr_with_tesseract(sample_image)
        
        assert result is not None
        assert "text" in result
        assert result["text"] == "Extracted text from image"
    
    @patch('ingestion.image_processor.TESSERACT_AVAILABLE', False)
    def test_ocr_with_tesseract_not_available(self, image_processor, sample_image):
        """Test OCR when Tesseract is not available."""
        result = image_processor._ocr_with_tesseract(sample_image)
        assert result is None
    
    @patch('ingestion.image_processor.TESSERACT_AVAILABLE', True)
    @patch('ingestion.image_processor.pytesseract')
    def test_ocr_with_tesseract_empty_text(self, mock_pytesseract, image_processor, sample_image):
        """Test OCR when Tesseract returns empty text."""
        mock_pytesseract.image_to_string.return_value = ""
        
        result = image_processor._ocr_with_tesseract(sample_image)
        assert result is None
    
    def test_caption_with_gpt4v_success(self, image_processor, sample_image):
        """Test successful GPT-4 Vision captioning."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        # content should be a string directly
        mock_message.content = "CAPTION: This is a test image showing a red square.\nTEXT: No text found"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        image_processor.client.chat.completions.create.return_value = mock_response
        
        img = Image.new('RGB', (100, 100))
        result = image_processor._caption_with_gpt4v(img, "JPEG")
        
        assert result is not None
        assert "caption" in result
        assert "extracted_text" in result
        assert "confidence" in result
    
    def test_caption_with_gpt4v_error(self, image_processor, sample_image):
        """Test GPT-4 Vision captioning error handling."""
        image_processor.client.chat.completions.create.side_effect = Exception("API error")
        
        img = Image.new('RGB', (100, 100))
        with pytest.raises(APIError):
            image_processor._caption_with_gpt4v(img, "JPEG")
    
    def test_process_success(self, image_processor, sample_image):
        """Test successful image processing."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "CAPTION: This is a test image.\nTEXT: Some extracted text"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        image_processor.client.chat.completions.create.return_value = mock_response
        
        chunks = image_processor.process(sample_image)
        
        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
        assert all("chunk_id" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
    
    def test_process_empty_extraction(self, image_processor, sample_image):
        """Test processing when extraction returns empty."""
        # Mock to return empty caption and text
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = "CAPTION: \nTEXT: No text found"
        mock_message.content = [mock_content]
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        image_processor.client.chat.completions.create.return_value = mock_response
        
        chunks = image_processor.process(sample_image)
        
        # Should still create chunks (at least with basic metadata)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
    
    def test_generate_file_id(self, image_processor, sample_image):
        """Test file ID generation."""
        file_id = image_processor.generate_file_id(sample_image)
        
        assert file_id.startswith("image_")
        assert len(file_id) > 6
    
    def test_chunk_metadata_structure(self, image_processor, sample_image):
        """Test that chunk metadata has correct structure."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "CAPTION: Test caption.\nTEXT: Test text"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        image_processor.client.chat.completions.create.return_value = mock_response
        
        chunks = image_processor.process(sample_image)
        
        for chunk in chunks:
            metadata = chunk["metadata"]
            assert "width" in metadata
            assert "height" in metadata
            assert "format" in metadata
            assert "caption" in metadata
            assert "extracted_text" in metadata
            assert "ocr_method" in metadata

