"""Audio processor with Whisper transcription."""

from pathlib import Path
from typing import List, Dict, Any
import openai
from pydub import AudioSegment

from utils.logging import logger
from utils.config import OPENAI_API_KEY
from utils.errors import APIError, ProcessingError
from ingestion.base import BaseProcessor


class AudioProcessor(BaseProcessor):
    """Processor for audio files (MP3, WAV)."""
    
    SUPPORTED_EXTENSIONS = [".mp3", ".wav"]
    CHUNK_DURATION_SECONDS = 300  # 5 minutes per chunk
    
    def __init__(self):
        super().__init__()
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for audio processing")
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    def _get_audio_info(self, file_path: Path) -> Dict[str, Any]:
        """Get audio file metadata."""
        try:
            audio = AudioSegment.from_file(str(file_path))
            return {
                "duration_seconds": len(audio) / 1000.0,
                "duration_ms": len(audio),
                "frame_rate": audio.frame_rate,
                "channels": audio.channels,
                "sample_width": audio.sample_width
            }
        except Exception as e:
            logger.warning(f"Could not read audio metadata: {e}")
            return {
                "duration_seconds": 0,
                "duration_ms": 0,
                "frame_rate": 0,
                "channels": 0,
                "sample_width": 0
            }
    
    def _transcribe_with_whisper(self, file_path: Path, language: str = None) -> Dict[str, Any]:
        """Transcribe audio using OpenAI Whisper API."""
        try:
            with open(file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    response_format="verbose_json"
                )
            
            return {
                "text": transcript.text,
                "language": transcript.language,
                "duration": transcript.duration,
                "segments": getattr(transcript, "segments", [])
            }
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise APIError(f"Failed to transcribe audio: {e}")
    
    def _chunk_transcription(self, transcription: str, audio_info: Dict[str, Any], file_id: str) -> List[Dict[str, Any]]:
        """Split transcription into chunks based on duration or content."""
        duration = audio_info.get("duration_seconds", 0)
        
        # If audio is short, return single chunk
        if duration <= self.CHUNK_DURATION_SECONDS or len(transcription) < 2000:
            return [{
                "content": transcription,
                "chunk_id": f"{file_id}_chunk_0",
                "chunk_index": 0,
                "metadata": {
                    "start_time": 0,
                    "end_time": duration,
                    "word_count": len(transcription.split()),
                    "character_count": len(transcription)
                }
            }]
        
        # Split by sentences or fixed size
        sentences = transcription.split(". ")
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        estimated_chars_per_second = len(transcription) / duration if duration > 0 else 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_length + sentence_length > estimated_chars_per_second * self.CHUNK_DURATION_SECONDS and current_chunk:
                chunk_text = ". ".join(current_chunk) + "."
                start_time = (chunk_index * self.CHUNK_DURATION_SECONDS)
                end_time = min((chunk_index + 1) * self.CHUNK_DURATION_SECONDS, duration)
                
                chunks.append({
                    "content": chunk_text,
                    "chunk_id": f"{file_id}_chunk_{chunk_index}",
                    "chunk_index": chunk_index,
                    "metadata": {
                        "start_time": start_time,
                        "end_time": end_time,
                        "word_count": len(chunk_text.split()),
                        "character_count": len(chunk_text)
                    }
                })
                
                current_chunk = [sentence]
                current_length = sentence_length
                chunk_index += 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 2  # +2 for ". "
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ". ".join(current_chunk)
            start_time = (chunk_index * self.CHUNK_DURATION_SECONDS)
            end_time = duration
            
            chunks.append({
                "content": chunk_text,
                "chunk_id": f"{file_id}_chunk_{chunk_index}",
                "chunk_index": chunk_index,
                "metadata": {
                    "start_time": start_time,
                    "end_time": end_time,
                    "word_count": len(chunk_text.split()),
                    "character_count": len(chunk_text)
                }
            })
        
        return chunks
    
    def process(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process audio file and return chunks."""
        file_id = self.generate_file_id(file_path)
        
        # Get audio metadata
        audio_info = self._get_audio_info(file_path)
        
        # Transcribe audio
        transcription_result = self._transcribe_with_whisper(file_path)
        transcription_text = transcription_result["text"]
        language = transcription_result.get("language", "unknown")
        
        if not transcription_text or not transcription_text.strip():
            logger.warning(f"No transcription extracted from {file_path.name}")
            return []
        
        # Chunk the transcription
        chunks = self._chunk_transcription(transcription_text, audio_info, file_id)
        
        # Add audio-specific metadata to each chunk
        for chunk in chunks:
            chunk["metadata"].update({
                "language": language,
                "duration_seconds": audio_info.get("duration_seconds"),
                "frame_rate": audio_info.get("frame_rate"),
                "channels": audio_info.get("channels")
            })
        
        return chunks

