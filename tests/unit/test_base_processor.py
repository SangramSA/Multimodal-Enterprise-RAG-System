"""Unit tests for base processor."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from ingestion.base import BaseProcessor
from utils.errors import FileError, ValidationError


class ConcreteProcessor(BaseProcessor):
    """Concrete implementation for testing."""
    SUPPORTED_EXTENSIONS = [".txt", ".pdf"]
    MAX_FILE_SIZE_MB = 10
    
    def process(self, file_path: Path):
        """Dummy process implementation."""
        return [{"content": "test", "chunk_id": "test_0", "chunk_index": 0}]


class TestBaseProcessor:
    """Test suite for BaseProcessor."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = ConcreteProcessor()
        assert processor.modality == "concrete"
        assert processor.SUPPORTED_EXTENSIONS == [".txt", ".pdf"]
        assert processor.MAX_FILE_SIZE_MB == 10
    
    def test_validate_file_exists(self, tmp_path):
        """Test file validation when file exists."""
        processor = ConcreteProcessor()
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        result = processor.validate(test_file)
        assert result is True
    
    def test_validate_file_not_found(self, tmp_path):
        """Test file validation when file doesn't exist."""
        processor = ConcreteProcessor()
        non_existent = tmp_path / "nonexistent.txt"
        
        with pytest.raises(FileError):
            processor.validate(non_existent)
    
    def test_validate_unsupported_extension(self, tmp_path):
        """Test file validation with unsupported extension."""
        processor = ConcreteProcessor()
        test_file = tmp_path / "test.doc"
        test_file.write_text("test")
        
        with pytest.raises(ValidationError) as exc_info:
            processor.validate(test_file)
        assert "Unsupported file type" in str(exc_info.value)
    
    def test_validate_file_too_large(self, tmp_path):
        """Test file validation when file exceeds size limit."""
        processor = ConcreteProcessor()
        test_file = tmp_path / "test.txt"
        # Create a file larger than 10MB
        large_content = "x" * (11 * 1024 * 1024)  # 11MB
        test_file.write_text(large_content)
        
        with pytest.raises(ValidationError) as exc_info:
            processor.validate(test_file)
        assert "File too large" in str(exc_info.value)
    
    def test_generate_file_id(self, tmp_path):
        """Test file ID generation."""
        processor = ConcreteProcessor()
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        file_id = processor.generate_file_id(test_file)
        assert file_id.startswith("concrete_")
        assert len(file_id) > 10
    
    def test_generate_file_id_uniqueness(self, tmp_path):
        """Test that file IDs are unique for different files."""
        processor = ConcreteProcessor()
        file1 = tmp_path / "test1.txt"
        file2 = tmp_path / "test2.txt"
        file1.write_text("content 1")
        file2.write_text("content 2")
        
        id1 = processor.generate_file_id(file1)
        id2 = processor.generate_file_id(file2)
        
        assert id1 != id2
    
    def test_extract_metadata(self, tmp_path):
        """Test metadata extraction."""
        processor = ConcreteProcessor()
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        file_id = "test_id_123"
        
        metadata = processor.extract_metadata(test_file, file_id)
        
        assert metadata["file_id"] == file_id
        assert metadata["file_name"] == "test.txt"
        assert metadata["file_type"] == ".txt"
        assert metadata["modality"] == "concrete"
        assert "upload_timestamp" in metadata
        assert "file_size" in metadata
    
    def test_process_file_success(self, tmp_path):
        """Test successful file processing."""
        processor = ConcreteProcessor()
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        result = processor.process_file(test_file)
        
        assert result["processing_status"] == "success"
        assert "file_id" in result
        assert "chunks" in result
        assert len(result["chunks"]) > 0
        assert result["chunks"][0]["chunk_id"] == "test_0"
        assert result["chunks"][0]["file_id"] == result["file_id"]
    
    def test_process_file_enriches_chunks(self, tmp_path):
        """Test that chunks are enriched with metadata."""
        processor = ConcreteProcessor()
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        result = processor.process_file(test_file)
        chunk = result["chunks"][0]
        
        assert "metadata" in chunk
        assert chunk["metadata"]["file_id"] == result["file_id"]
        assert chunk["metadata"]["modality"] == "concrete"
        assert "processing_timestamp" in chunk
        assert "total_chunks" in chunk
    
    def test_process_file_validation_error(self, tmp_path):
        """Test that validation errors are raised during processing."""
        processor = ConcreteProcessor()
        test_file = tmp_path / "test.doc"  # Unsupported extension
        test_file.write_text("test")  # Create file first
        
        with pytest.raises(ValidationError):
            processor.process_file(test_file)

