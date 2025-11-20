"""Unit tests for AudioProcessor."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from ingestion.audio_processor import AudioProcessor
from utils.errors import APIError, ProcessingError


class TestAudioProcessor:
    """Test suite for AudioProcessor."""
    
    @pytest.fixture
    def audio_processor(self, mocker):
        """Create AudioProcessor instance with mocked OpenAI client."""
        with patch('ingestion.audio_processor.OPENAI_API_KEY', 'test-key'):
            processor = AudioProcessor()
            processor.client = Mock()
            return processor
    
    @pytest.fixture
    def mock_whisper_response(self):
        """Mock Whisper API response."""
        mock_response = Mock()
        mock_response.text = "This is a test transcription."
        mock_response.language = "en"
        mock_response.duration = 10.5
        mock_response.segments = []
        return mock_response
    
    def test_init_without_api_key(self, mocker):
        """Test initialization fails without API key."""
        with patch('ingestion.audio_processor.OPENAI_API_KEY', None):
            with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
                AudioProcessor()
    
    def test_supported_extensions(self, audio_processor):
        """Test supported file extensions."""
        assert ".mp3" in audio_processor.SUPPORTED_EXTENSIONS
        assert ".wav" in audio_processor.SUPPORTED_EXTENSIONS
    
    def test_get_audio_info_with_pydub(self, audio_processor, tmp_audio_file):
        """Test getting audio info when pydub works."""
        with patch('ingestion.audio_processor.AudioSegment') as mock_audio:
            mock_segment = Mock()
            mock_segment.__len__ = Mock(return_value=10000)  # 10 seconds
            mock_segment.frame_rate = 44100
            mock_segment.channels = 2
            mock_segment.sample_width = 2
            mock_audio.from_file.return_value = mock_segment
            
            info = audio_processor._get_audio_info(tmp_audio_file)
            
            assert info["duration_seconds"] == 10.0
            assert info["duration_ms"] == 10000
            assert info["frame_rate"] == 44100
            assert info["channels"] == 2
            assert info["sample_width"] == 2
    
    def test_get_audio_info_without_ffmpeg(self, audio_processor, tmp_audio_file):
        """Test getting audio info when ffmpeg is not installed."""
        with patch('ingestion.audio_processor.AudioSegment') as mock_audio:
            mock_audio.from_file.side_effect = FileNotFoundError("ffmpeg not found")
            
            info = audio_processor._get_audio_info(tmp_audio_file)
            
            assert info["duration_seconds"] == 0
            assert info["duration_ms"] == 0
            assert info["frame_rate"] == 0
    
    def test_get_audio_info_error_handling(self, audio_processor, tmp_audio_file):
        """Test error handling in get_audio_info."""
        with patch('ingestion.audio_processor.AudioSegment') as mock_audio:
            mock_audio.from_file.side_effect = Exception("Unknown error")
            
            info = audio_processor._get_audio_info(tmp_audio_file)
            
            # Should return default values on error
            assert info["duration_seconds"] == 0
            assert info["duration_ms"] == 0
    
    def test_transcribe_with_whisper_success(self, audio_processor, tmp_audio_file, mock_whisper_response):
        """Test successful Whisper transcription."""
        audio_processor.client.audio.transcriptions.create.return_value = mock_whisper_response
        
        result = audio_processor._transcribe_with_whisper(tmp_audio_file)
        
        assert result["text"] == "This is a test transcription."
        assert result["language"] == "en"
        assert result["duration"] == 10.5
        audio_processor.client.audio.transcriptions.create.assert_called_once()
    
    def test_transcribe_with_whisper_error(self, audio_processor, tmp_audio_file):
        """Test Whisper transcription error handling."""
        audio_processor.client.audio.transcriptions.create.side_effect = Exception("API error")
        
        with pytest.raises(APIError, match="Failed to transcribe audio"):
            audio_processor._transcribe_with_whisper(tmp_audio_file)
    
    def test_chunk_transcription_short_audio(self, audio_processor):
        """Test chunking for short audio files."""
        transcription = "This is a short transcription."
        audio_info = {"duration_seconds": 5.0}
        file_id = "test_audio"
        
        chunks = audio_processor._chunk_transcription(transcription, audio_info, file_id)
        
        assert len(chunks) == 1
        assert chunks[0]["content"] == transcription
        assert chunks[0]["chunk_id"] == "test_audio_chunk_0"
        assert chunks[0]["chunk_index"] == 0
    
    def test_chunk_transcription_long_audio(self, audio_processor):
        """Test chunking for long audio files."""
        # Create a long transcription (must be > 2000 chars to trigger chunking)
        # Each sentence is ~20 chars, so we need > 100 sentences
        sentences = [f"This is a longer sentence number {i} with more content." for i in range(150)]
        transcription = ". ".join(sentences)
        audio_info = {"duration_seconds": 600.0}  # 10 minutes
        file_id = "test_audio"
        
        chunks = audio_processor._chunk_transcription(transcription, audio_info, file_id)
        
        # Should create multiple chunks for long audio (if transcription > 2000 chars)
        # Note: The logic checks both duration AND transcription length
        # If transcription < 2000 chars, it returns single chunk even for long duration
        if len(transcription) >= 2000:
            assert len(chunks) > 1
        assert all("chunk_id" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
    
    def test_process_success(self, audio_processor, tmp_audio_file, mock_whisper_response):
        """Test successful audio processing."""
        audio_processor.client.audio.transcriptions.create.return_value = mock_whisper_response
        
        with patch.object(audio_processor, '_get_audio_info', return_value={
            "duration_seconds": 10.5,
            "duration_ms": 10500,
            "frame_rate": 44100,
            "channels": 2,
            "sample_width": 2
        }):
            chunks = audio_processor.process(tmp_audio_file)
        
        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
        assert all("chunk_id" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
        assert all(chunk["metadata"].get("language") == "en" for chunk in chunks)
    
    def test_process_empty_transcription(self, audio_processor, tmp_audio_file):
        """Test processing when transcription is empty."""
        mock_response = Mock()
        mock_response.text = ""
        mock_response.language = "en"
        mock_response.duration = 0
        mock_response.segments = []
        audio_processor.client.audio.transcriptions.create.return_value = mock_response
        
        with patch.object(audio_processor, '_get_audio_info', return_value={"duration_seconds": 0}):
            chunks = audio_processor.process(tmp_audio_file)
        
        assert chunks == []
    
    def test_process_uses_whisper_duration_fallback(self, audio_processor, tmp_audio_file, mock_whisper_response):
        """Test that Whisper duration is used as fallback when pydub fails."""
        audio_processor.client.audio.transcriptions.create.return_value = mock_whisper_response
        
        with patch.object(audio_processor, '_get_audio_info', return_value={
            "duration_seconds": 0,  # pydub failed
            "duration_ms": 0,
            "frame_rate": 0,
            "channels": 0,
            "sample_width": 0
        }):
            chunks = audio_processor.process(tmp_audio_file)
        
        # Should still process successfully using Whisper duration
        assert len(chunks) > 0
    
    def test_generate_file_id(self, audio_processor, tmp_audio_file):
        """Test file ID generation."""
        file_id = audio_processor.generate_file_id(tmp_audio_file)
        
        assert file_id.startswith("audio_")
        assert len(file_id) > 6
    
    def test_chunk_metadata_structure(self, audio_processor):
        """Test that chunk metadata has correct structure."""
        transcription = "Test transcription."
        audio_info = {"duration_seconds": 5.0}
        file_id = "test_audio"
        
        chunks = audio_processor._chunk_transcription(transcription, audio_info, file_id)
        
        for chunk in chunks:
            metadata = chunk["metadata"]
            assert "start_time" in metadata
            assert "end_time" in metadata
            assert "word_count" in metadata
            assert "character_count" in metadata

