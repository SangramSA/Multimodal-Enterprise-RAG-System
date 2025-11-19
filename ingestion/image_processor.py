"""Image processor with OCR and GPT-4 Vision captioning."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import base64
from io import BytesIO

from PIL import Image
import openai
from utils.logging import logger
from utils.config import OPENAI_API_KEY, OPENAI_VISION_MODEL
from utils.errors import APIError, ProcessingError, retry_with_backoff
from ingestion.base import BaseProcessor

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.debug("pytesseract not available, OCR will use GPT-4 Vision only")


class ImageProcessor(BaseProcessor):
    """Processor for image files (JPG, PNG)."""
    
    SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png"]
    MAX_IMAGE_DIMENSION = 4096  # Max width or height for API
    MAX_IMAGE_SIZE_MB = 20  # Max file size for processing
    
    def __init__(self):
        super().__init__()
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for image processing")
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    def _validate_image(self, image_path: Path) -> Dict[str, Any]:
        """Validate and get image metadata in one pass."""
        try:
            with Image.open(image_path) as img:
                # Verify it's a valid image
                img.verify()
            
            # Reopen for actual processing (verify() closes the image)
            with Image.open(image_path) as img:
                width, height = img.size
                format_name = img.format or "unknown"
                
                # Check dimensions
                if width > self.MAX_IMAGE_DIMENSION or height > self.MAX_IMAGE_DIMENSION:
                    logger.warning(
                        f"Image dimensions ({width}x{height}) exceed max ({self.MAX_IMAGE_DIMENSION}). "
                        f"Will resize if needed for API."
                    )
                
                return {
                    "width": width,
                    "height": height,
                    "format": format_name,
                    "image": img.copy()  # Copy to avoid closing issues
                }
        except Exception as e:
            raise ProcessingError(f"Invalid or corrupted image: {e}")
    
    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        """Resize image if it's too large for API."""
        width, height = image.size
        if width <= self.MAX_IMAGE_DIMENSION and height <= self.MAX_IMAGE_DIMENSION:
            return image
        
        # Calculate new dimensions maintaining aspect ratio
        ratio = min(self.MAX_IMAGE_DIMENSION / width, self.MAX_IMAGE_DIMENSION / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _encode_image(self, image: Image.Image, format_name: str) -> str:
        """Encode image to base64."""
        buffer = BytesIO()
        # Use format_name or default to PNG
        save_format = format_name.upper() if format_name != "unknown" else "PNG"
        image.save(buffer, format=save_format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _ocr_with_tesseract(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """Extract text using Tesseract OCR with confidence scores."""
        if not TESSERACT_AVAILABLE:
            logger.debug("Tesseract not available, skipping OCR")
            return None
        
        try:
            logger.debug(f"Attempting Tesseract OCR for {image_path.name}")
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            
            if not text or not text.strip():
                logger.debug("Tesseract OCR returned no text")
                return None
            
            # Get confidence scores from Tesseract
            try:
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                # Filter out empty confidences and calculate average
                confidences = [conf for conf in data.get('conf', []) if conf != -1]
                avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.5
            except Exception:
                # Fallback if confidence extraction fails
                avg_confidence = 0.5
            
            logger.debug(f"Tesseract OCR extracted {len(text)} characters with confidence {avg_confidence:.2f}")
            return {
                "text": text.strip(),
                "confidence": avg_confidence
            }
        except Exception as e:
            logger.debug(f"Tesseract OCR failed (this is optional): {e}")
            return None
    
    def _caption_with_gpt4v(self, image: Image.Image, format_name: str) -> Dict[str, Any]:
        """Generate caption and extract text using GPT-4 Vision."""
        import time
        
        def _call_api():
            # Resize if needed
            processed_image = self._resize_if_needed(image)
            original_size = image.size
            processed_size = processed_image.size
            if original_size != processed_size:
                logger.info(f"Resized image from {original_size} to {processed_size} for GPT-4 Vision")
            
            # Encode image
            base64_image = self._encode_image(processed_image, format_name)
            image_size_kb = len(base64_image) * 3 / 4 / 1024  # Approximate size
            logger.debug(f"Encoded image size: {image_size_kb:.1f} KB")
            
            # Ensure format_name is safe for data URL
            safe_format = format_name.lower() if format_name != "unknown" else "png"
            
            logger.info(f"Calling GPT-4 Vision API (model: {OPENAI_VISION_MODEL}) for image analysis...")
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=OPENAI_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this image and provide:
1) A detailed caption describing the image content, context, and key elements
2) Any text visible in the image (OCR)

Format your response EXACTLY as:
CAPTION: [your detailed caption here]
TEXT: [any extracted text here, or "No text found" if none]"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{safe_format};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000  # Increased for detailed captions
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"GPT-4 Vision API call completed in {elapsed_time:.2f}s")
            
            # Log token usage if available
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                logger.debug(
                    f"Token usage - Prompt: {usage.prompt_tokens}, "
                    f"Completion: {usage.completion_tokens}, "
                    f"Total: {usage.total_tokens}"
                )
            
            return response
        
        try:
            # Use retry logic
            logger.debug("Starting GPT-4 Vision API call with retry logic")
            response = retry_with_backoff(_call_api, max_retries=3)
            result_text = response.choices[0].message.content
            logger.debug(f"GPT-4 Vision response length: {len(result_text)} characters")
            
            # More robust parsing
            caption = ""
            extracted_text = ""
            
            if "CAPTION:" in result_text:
                logger.debug("Successfully parsed GPT-4 Vision response with CAPTION: prefix")
                parts = result_text.split("TEXT:", 1)  # Split only once
                caption = parts[0].replace("CAPTION:", "").strip()
                if len(parts) > 1:
                    extracted_text = parts[1].strip()
                    if extracted_text.lower() in ["no text found", "none", ""]:
                        extracted_text = ""
                        logger.debug("GPT-4 Vision indicated no text found in image")
            else:
                # Fallback: try to extract caption from response
                logger.warning("GPT-4 Vision response didn't match expected format, using fallback parsing")
                caption = result_text.strip()
                extracted_text = ""
            
            # Calculate confidence based on response quality
            # GPT-4 Vision doesn't provide confidence, so we estimate based on:
            # - Successful parsing (1.0 if parsed correctly, 0.7 if fallback)
            # - Response completeness (length of caption and text)
            parsing_confidence = 1.0 if "CAPTION:" in result_text else 0.7
            content_quality = min(
                (len(caption) / 100.0) * 0.3 + (len(extracted_text) / 50.0) * 0.2,
                0.5
            )
            estimated_confidence = min(parsing_confidence + content_quality, 1.0)
            
            logger.success(
                f"GPT-4 Vision analysis complete - Caption: {len(caption)} chars, "
                f"Text: {len(extracted_text)} chars, Confidence: {estimated_confidence:.2f}"
            )
            
            return {
                "caption": caption,
                "extracted_text": extracted_text,
                "confidence": estimated_confidence
            }
        except Exception as e:
            logger.error(f"GPT-4 Vision API call failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            if hasattr(e, 'status_code'):
                logger.error(f"HTTP status code: {e.status_code}")
            raise APIError(f"Failed to process image with GPT-4 Vision: {e}")
    
    def process(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process image file and return chunks."""
        file_id = self.generate_file_id(file_path)
        
        # Validate and get image metadata in one pass
        image_info = self._validate_image(file_path)
        width = image_info["width"]
        height = image_info["height"]
        format_name = image_info["format"]
        image = image_info["image"]
        
        # Try Tesseract OCR first (faster, free) - optional fallback
        logger.debug("Attempting optional Tesseract OCR...")
        ocr_result = self._ocr_with_tesseract(file_path)
        ocr_text = ocr_result["text"] if ocr_result else None
        ocr_confidence = ocr_result["confidence"] if ocr_result else None
        if ocr_text:
            logger.debug(f"Tesseract OCR extracted {len(ocr_text)} characters")
        else:
            logger.debug("Tesseract OCR not available or returned no text (this is expected)")
        
        # Use GPT-4 Vision for captioning and enhanced OCR (primary method)
        logger.info(f"Processing image with GPT-4 Vision: {file_path.name} ({width}x{height}, {format_name})")
        try:
            gpt4v_result = self._caption_with_gpt4v(image, format_name)
            caption = gpt4v_result["caption"]
            gpt4v_text = gpt4v_result["extracted_text"]
            gpt4v_confidence = gpt4v_result["confidence"]
            logger.info(f"GPT-4 Vision processing successful for {file_path.name}")
        except Exception as e:
            logger.warning(f"GPT-4 Vision failed for {file_path.name}: {e}")
            logger.warning(f"Falling back to Tesseract OCR (if available) or basic metadata")
            caption = f"Image: {file_path.name}"
            gpt4v_text = ""
            gpt4v_confidence = None
        
        # Combine OCR results (prefer GPT-4 Vision text if available)
        final_text = gpt4v_text if gpt4v_text else (ocr_text or "")
        
        # Calculate overall confidence:
        # - If GPT-4V succeeded: use its confidence (it's more reliable)
        # - If only Tesseract: use Tesseract confidence
        # - If both failed: None (no confidence available)
        if gpt4v_confidence is not None:
            confidence = gpt4v_confidence
        elif ocr_confidence is not None:
            confidence = ocr_confidence
        else:
            confidence = None  # No reliable confidence available
        
        # Create content - keep structure but make it searchable
        content_parts = []
        if caption:
            content_parts.append(f"Image Description: {caption}")
        if final_text:
            content_parts.append(f"Extracted Text: {final_text}")
        
        content = "\n\n".join(content_parts) if content_parts else caption or "No content extracted"
        
        return [{
            "content": content,
            "chunk_id": f"{file_id}_chunk_0",
            "chunk_index": 0,
            "metadata": {
                "width": width,
                "height": height,
                "format": format_name,
                "caption": caption,  # Keep separate for better searchability
                "extracted_text": final_text,  # Keep separate
                "ocr_method": "gpt4v" if gpt4v_text else ("tesseract" if ocr_text else "none"),
                "confidence": confidence,  # None if no reliable confidence available
                "word_count": len(content.split()),
                "character_count": len(content)
            }
        }]

