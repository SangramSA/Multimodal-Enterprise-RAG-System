"""Text processor for PDF and TXT files."""

from pathlib import Path
from typing import List, Dict, Any
import re

from utils.logging import logger
from ingestion.base import BaseProcessor

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("langchain-text-splitters not available, falling back to basic chunking")

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("pdfplumber not available, PDF processing will be limited")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False


class TextProcessor(BaseProcessor):
    """Processor for text files (PDF, TXT)."""
    
    SUPPORTED_EXTENSIONS = [".pdf", ".txt"]
    CHUNK_SIZE = 1000  # characters
    CHUNK_OVERLAP = 200  # characters
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        super().__init__()
        if chunk_size:
            self.CHUNK_SIZE = chunk_size
        if chunk_overlap:
            self.CHUNK_OVERLAP = chunk_overlap
    
    def _read_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        text = ""
        
        # Try pdfplumber first (better for complex layouts)
        if PDF_AVAILABLE:
            try:
                with pdfplumber.open(file_path) as pdf:
                    pages_text = []
                    for page_num, page in enumerate(pdf.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            pages_text.append(f"[Page {page_num}]\n{page_text}")
                    text = "\n\n".join(pages_text)
                    logger.info(f"Extracted text from {len(pdf.pages)} pages using pdfplumber")
                    return text
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}, trying PyPDF2")
        
        # Fallback to PyPDF2
        if PYPDF2_AVAILABLE:
            try:
                with open(file_path, "rb") as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    pages_text = []
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            pages_text.append(f"[Page {page_num}]\n{page_text}")
                    text = "\n\n".join(pages_text)
                    logger.info(f"Extracted text from {len(pdf_reader.pages)} pages using PyPDF2")
                    return text
            except Exception as e:
                logger.error(f"PyPDF2 failed: {e}")
                raise
        
        raise Exception("No PDF library available")
    
    def _read_txt(self, file_path: Path) -> str:
        """Read text from TXT file."""
        try:
            # Try UTF-8 first
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()
    
    def _chunk_text(self, text: str, file_id: str) -> List[Dict[str, Any]]:
        """Split text into chunks with overlap using LangChain's RecursiveCharacterTextSplitter."""
        # Use LangChain's splitter if available, otherwise fall back to basic chunking
        if LANGCHAIN_AVAILABLE:
            try:
                # Initialize the splitter with configured chunk size and overlap
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.CHUNK_SIZE,
                    chunk_overlap=self.CHUNK_OVERLAP,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # Tries these in order
                )
                
                # Create documents from text
                documents = splitter.create_documents([text])
                
                # Convert to our format
                chunks = []
                for i, doc in enumerate(documents):
                    chunks.append({
                        "content": doc.page_content,
                        "chunk_id": f"{file_id}_chunk_{i}",
                        "chunk_index": i,
                        "metadata": {
                            **doc.metadata,  # Preserves any metadata from LangChain
                            "word_count": len(doc.page_content.split()),
                            "character_count": len(doc.page_content)
                        }
                    })
                
                return chunks
            except Exception as e:
                logger.warning(f"LangChain chunking failed: {e}, falling back to basic chunking")
        
        # Fallback to basic chunking if LangChain is not available or fails
        if len(text) <= self.CHUNK_SIZE:
            return [{
                "content": text,
                "chunk_id": f"{file_id}_chunk_0",
                "chunk_index": 0,
                "metadata": {
                    "word_count": len(text.split()),
                    "character_count": len(text)
                }
            }]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Find end position
            end = start + self.CHUNK_SIZE
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                sentence_end = max(
                    text.rfind(". ", start, end),
                    text.rfind(".\n", start, end),
                    text.rfind("! ", start, end),
                    text.rfind("? ", start, end),
                    text.rfind("\n\n", start, end)
                )
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "chunk_id": f"{file_id}_chunk_{chunk_index}",
                    "chunk_index": chunk_index,
                    "metadata": {
                        "word_count": len(chunk_text.split()),
                        "character_count": len(chunk_text),
                        "start_char": start,
                        "end_char": end
                    }
                })
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.CHUNK_OVERLAP if end < len(text) else end
        
        return chunks
    
    def process(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process text file and return chunks."""
        file_id = self.generate_file_id(file_path)
        
        # Read file based on extension
        if file_path.suffix.lower() == ".pdf":
            text = self._read_pdf(file_path)
        elif file_path.suffix.lower() == ".txt":
            text = self._read_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        if not text or not text.strip():
            logger.warning(f"No text extracted from {file_path.name}")
            return []
        
        # Chunk the text
        chunks = self._chunk_text(text, file_id)
        
        # Add page numbers for PDFs
        if file_path.suffix.lower() == ".pdf":
            for chunk in chunks:
                # Extract page number from content if present
                page_match = re.search(r'\[Page (\d+)\]', chunk["content"])
                if page_match:
                    chunk["metadata"]["page_number"] = int(page_match.group(1))
        
        return chunks

