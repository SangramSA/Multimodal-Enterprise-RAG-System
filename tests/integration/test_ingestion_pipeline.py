"""Integration tests for ingestion pipeline."""

import pytest
from pathlib import Path
from pipeline.ingestion_pipeline import IngestionPipeline


def test_ingestion_pipeline_initialization():
    """Test ingestion pipeline initialization."""
    pipeline = IngestionPipeline()
    assert pipeline.text_processor is not None
    assert pipeline.image_processor is not None
    assert pipeline.audio_processor is not None


def test_process_txt_file(tmp_path):
    """Test processing a text file through pipeline."""
    pipeline = IngestionPipeline()
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document with some content.")
    
    result = pipeline.process_file(test_file)
    assert result["processing_status"] == "success"
    assert "file_id" in result
    assert "chunks" in result
    assert len(result["chunks"]) > 0

