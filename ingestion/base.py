"""Base processor interface for all ingestion modules."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import hashlib
import uuid

from utils.logging import logger
from utils.errors import FileError, ValidationError


class BaseProcessor(ABC):
    """Abstract base class for all file processors."""
    
    SUPPORTED_EXTENSIONS: List[str] = []
    MAX_FILE_SIZE_MB: int = 100
    
    def __init__(self):
        self.modality = self.__class__.__name__.replace("Processor", "").lower()
    
    def validate(self, file_path: Path) -> bool:
        """Validate file before processing."""
        if not file_path.exists():
            raise FileError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValidationError(
                f"Unsupported file type: {file_path.suffix}. "
                f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            raise ValidationError(
                f"File too large: {file_size_mb:.2f}MB. "
                f"Maximum size: {self.MAX_FILE_SIZE_MB}MB"
            )
        
        return True
    
    def generate_file_id(self, file_path: Path) -> str:
        """Generate unique file ID based on file path and content."""
        file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
        return f"{self.modality}_{file_hash[:16]}"
    
    def extract_metadata(self, file_path: Path, file_id: str) -> Dict[str, Any]:
        """Extract basic metadata from file."""
        stat = file_path.stat()
        return {
            "file_id": file_id,
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_size": stat.st_size,
            "file_type": file_path.suffix.lower(),
            "modality": self.modality,
            "upload_timestamp": datetime.utcnow().isoformat(),
        }
    
    @abstractmethod
    def process(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Process file and return list of chunks with metadata.
        
        Returns:
            List of dictionaries, each containing:
            - content: str - The processed content
            - chunk_id: str - Unique chunk identifier
            - chunk_index: int - Index of chunk in file
            - metadata: Dict - Additional metadata
        """
        pass
    
    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Main entry point for processing a file."""
        try:
            self.validate(file_path)
            file_id = self.generate_file_id(file_path)
            metadata = self.extract_metadata(file_path, file_id)
            
            logger.info(f"Processing {self.modality} file: {file_path.name}")
            chunks = self.process(file_path)
            
            # Enrich chunks with metadata
            for i, chunk in enumerate(chunks):
                chunk["chunk_id"] = chunk.get("chunk_id", f"{file_id}_chunk_{i}")
                chunk["chunk_index"] = chunk.get("chunk_index", i)
                chunk["total_chunks"] = len(chunks)
                chunk["file_id"] = file_id
                chunk["metadata"] = {**metadata, **chunk.get("metadata", {})}
                chunk["processing_timestamp"] = datetime.utcnow().isoformat()
            
            logger.success(f"Processed {len(chunks)} chunks from {file_path.name}")
            return {
                "file_id": file_id,
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "file_type": file_path.suffix.lower(),
                "upload_timestamp": datetime.utcnow().isoformat(),
                "modality": self.modality,
                "chunks": chunks,
                "metadata": metadata,
                "processing_status": "success"
            }
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise

