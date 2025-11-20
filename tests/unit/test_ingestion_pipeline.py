"""Unit tests for IngestionPipeline."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from pipeline.ingestion_pipeline import IngestionPipeline


class TestIngestionPipeline:
    """Test suite for IngestionPipeline."""
    
    @pytest.fixture
    def ingestion_pipeline(self, mocker):
        """Create IngestionPipeline with mocked dependencies."""
        with patch('pipeline.ingestion_pipeline.TextProcessor') as mock_text:
            with patch('pipeline.ingestion_pipeline.ImageProcessor') as mock_image:
                with patch('pipeline.ingestion_pipeline.AudioProcessor') as mock_audio:
                    mock_text.return_value = Mock()
                    mock_image.return_value = Mock()
                    mock_audio.return_value = Mock()
                    pipeline = IngestionPipeline()
                    pipeline.text_processor = mock_text.return_value
                    pipeline.image_processor = mock_image.return_value
                    pipeline.audio_processor = mock_audio.return_value
                    return pipeline
    
    def test_init(self, mocker):
        """Test IngestionPipeline initialization."""
        with patch('pipeline.ingestion_pipeline.TextProcessor'):
            with patch('pipeline.ingestion_pipeline.ImageProcessor'):
                with patch('pipeline.ingestion_pipeline.AudioProcessor'):
                    pipeline = IngestionPipeline()
                    assert pipeline.text_processor is not None
                    assert pipeline.image_processor is not None
                    assert pipeline.audio_processor is not None
    
    def test_process_txt_file(self, ingestion_pipeline, tmp_path):
        """Test processing TXT file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        mock_result = {
            "processing_status": "success",
            "file_id": "text_test",
            "chunks": [{"chunk_id": "chunk1", "content": "Test content"}]
        }
        ingestion_pipeline.text_processor.process_file.return_value = mock_result
        
        result = ingestion_pipeline.process_file(test_file)
        
        assert result["processing_status"] == "success"
        ingestion_pipeline.text_processor.process_file.assert_called_once()
    
    def test_process_pdf_file(self, ingestion_pipeline, tmp_path):
        """Test processing PDF file."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%test\n")
        
        mock_result = {
            "processing_status": "success",
            "file_id": "text_test",
            "chunks": [{"chunk_id": "chunk1", "content": "PDF content"}]
        }
        ingestion_pipeline.text_processor.process_file.return_value = mock_result
        
        result = ingestion_pipeline.process_file(test_file)
        
        assert result["processing_status"] == "success"
        ingestion_pipeline.text_processor.process_file.assert_called_once()
    
    def test_process_image_file(self, ingestion_pipeline, tmp_path):
        """Test processing image file."""
        from PIL import Image
        test_file = tmp_path / "test.jpg"
        img = Image.new('RGB', (100, 100), color='red')
        img.save(test_file, 'JPEG')
        
        mock_result = {
            "processing_status": "success",
            "file_id": "image_test",
            "chunks": [{"chunk_id": "chunk1", "content": "Image content"}]
        }
        ingestion_pipeline.image_processor.process_file.return_value = mock_result
        
        result = ingestion_pipeline.process_file(test_file)
        
        assert result["processing_status"] == "success"
        ingestion_pipeline.image_processor.process_file.assert_called_once()
    
    def test_process_audio_file(self, ingestion_pipeline, tmp_path):
        """Test processing audio file."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"ID3\x03\x00")
        
        mock_result = {
            "processing_status": "success",
            "file_id": "audio_test",
            "chunks": [{"chunk_id": "chunk1", "content": "Audio transcription"}]
        }
        ingestion_pipeline.audio_processor.process_file.return_value = mock_result
        
        result = ingestion_pipeline.process_file(test_file)
        
        assert result["processing_status"] == "success"
        ingestion_pipeline.audio_processor.process_file.assert_called_once()
    
    def test_process_unsupported_file(self, ingestion_pipeline, tmp_path):
        """Test processing unsupported file type."""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("test")
        
        with pytest.raises(Exception):  # Should raise ProcessingError
            ingestion_pipeline.process_file(test_file)

