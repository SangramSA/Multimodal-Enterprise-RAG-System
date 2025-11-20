"""Multi-modal file processors for text, image, and audio.

This module contains processors for different file types:
- BaseProcessor: Base class for all processors
- TextProcessor: Handles PDF and TXT files
- ImageProcessor: Handles JPG, PNG files with OCR and captioning
- AudioProcessor: Handles MP3, WAV files with transcription
"""

__all__ = [
    "BaseProcessor",
    "TextProcessor",
    "ImageProcessor",
    "AudioProcessor",
]
