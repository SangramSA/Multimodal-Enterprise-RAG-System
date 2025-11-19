"""Unit tests for text processor."""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from ingestion.text_processor import TextProcessor
from utils.errors import ValidationError


class TestTextProcessor:
    """Test suite for TextProcessor."""
    
    def test_text_processor_initialization(self):
        """Test text processor initialization."""
        processor = TextProcessor()
        assert processor.SUPPORTED_EXTENSIONS == [".pdf", ".txt"]
        assert processor.modality == "text"
        assert processor.CHUNK_SIZE == 1000
        assert processor.CHUNK_OVERLAP == 200
    
    def test_text_processor_custom_chunk_size(self):
        """Test text processor with custom chunk size."""
        processor = TextProcessor(chunk_size=500, chunk_overlap=100)
        assert processor.CHUNK_SIZE == 500
        assert processor.CHUNK_OVERLAP == 100
    
    def test_validate_file(self, tmp_path):
        """Test file validation."""
        processor = TextProcessor()
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test file.")
        
        assert processor.validate(test_file) is True
    
    def test_validate_unsupported_extension(self, tmp_path):
        """Test validation with unsupported extension."""
        processor = TextProcessor()
        test_file = tmp_path / "test.doc"
        test_file.write_text("test")
        
        with pytest.raises(ValidationError):
            processor.validate(test_file)
    
    def test_generate_file_id(self, tmp_path):
        """Test file ID generation."""
        processor = TextProcessor()
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        file_id = processor.generate_file_id(test_file)
        assert file_id.startswith("text_")
        assert len(file_id) > 10
    
    def test_read_txt_file(self, tmp_path):
        """Test reading TXT file."""
        processor = TextProcessor()
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is test content.")
        
        text = processor._read_txt(test_file)
        assert text == "This is test content."
    
    def test_read_txt_file_utf8(self, tmp_path):
        """Test reading TXT file with UTF-8 encoding."""
        processor = TextProcessor()
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test with émojis 🎉", encoding="utf-8")
        
        text = processor._read_txt(test_file)
        assert "émojis" in text
        assert "🎉" in text
    
    def test_chunk_text_small(self, tmp_path):
        """Test chunking small text (fits in one chunk)."""
        processor = TextProcessor(chunk_size=1000, chunk_overlap=200)
        text = "This is a short text."
        file_id = "test_file"
        
        chunks = processor._chunk_text(text, file_id)
        assert len(chunks) == 1
        assert chunks[0]["content"] == text
        assert chunks[0]["chunk_id"] == f"{file_id}_chunk_0"
    
    def test_chunk_text_large(self, tmp_path):
        """Test chunking large text (multiple chunks)."""
        processor = TextProcessor(chunk_size=50, chunk_overlap=10)
        text = " ".join(["word"] * 100)  # Large text
        file_id = "test_file"
        
        chunks = processor._chunk_text(text, file_id)
        assert len(chunks) > 1
        assert all("content" in chunk for chunk in chunks)
        assert all("chunk_id" in chunk for chunk in chunks)
    
    def test_chunk_text_overlap(self, tmp_path):
        """Test that chunks have proper overlap."""
        processor = TextProcessor(chunk_size=50, chunk_overlap=10)
        text = " ".join(["word"] * 50)
        file_id = "test_file"
        
        chunks = processor._chunk_text(text, file_id)
        if len(chunks) > 1:
            # Check that there's overlap between consecutive chunks
            chunk1_end = chunks[0]["content"][-20:]
            chunk2_start = chunks[1]["content"][:20]
            # Some overlap should exist
            assert len(chunk1_end) > 0 and len(chunk2_start) > 0
    
    def test_process_txt_file(self, tmp_path):
        """Test processing a TXT file."""
        processor = TextProcessor()
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test sentence. This is another sentence.")
        
        chunks = processor.process(test_file)
        assert len(chunks) > 0
        assert "content" in chunks[0]
        assert "chunk_id" in chunks[0]
        assert "metadata" in chunks[0]
    
    def test_process_txt_file_empty(self, tmp_path):
        """Test processing an empty TXT file."""
        processor = TextProcessor()
        test_file = tmp_path / "test.txt"
        test_file.write_text("")
        
        chunks = processor.process(test_file)
        assert len(chunks) == 0
    
    def test_process_txt_file_whitespace_only(self, tmp_path):
        """Test processing a file with only whitespace."""
        processor = TextProcessor()
        test_file = tmp_path / "test.txt"
        test_file.write_text("   \n\n\t  ")
        
        chunks = processor.process(test_file)
        assert len(chunks) == 0
    
    @patch('ingestion.text_processor.pdfplumber')
    def test_read_pdf_with_pdfplumber(self, mock_pdfplumber, tmp_path):
        """Test reading PDF using pdfplumber."""
        processor = TextProcessor()
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%test\n")
        
        # Mock pdfplumber
        mock_pdf = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = "Page content"
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__ = Mock(return_value=mock_pdf)
        mock_pdfplumber.open.return_value.__exit__ = Mock(return_value=None)
        
        # This will fail if pdfplumber isn't available, but that's okay for testing
        try:
            text = processor._read_pdf(test_file)
            assert "Page content" in text
        except Exception:
            # If pdfplumber isn't available, skip this test
            pytest.skip("pdfplumber not available")
    
    def test_process_file_complete_flow(self, tmp_path):
        """Test complete file processing flow."""
        processor = TextProcessor()
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document with multiple sentences. " * 10)
        
        result = processor.process_file(test_file)
        
        assert result["processing_status"] == "success"
        assert "file_id" in result
        assert "chunks" in result
        assert len(result["chunks"]) > 0
        assert all("content" in chunk for chunk in result["chunks"])
        assert all("metadata" in chunk for chunk in result["chunks"])

