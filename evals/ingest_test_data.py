"""Script to ingest test data from SQuAD v2, DocVQA, and FLEURS into the system."""

import sys
from pathlib import Path
import tempfile
import json
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from datasets import load_dataset
from pipeline.ingestion_pipeline import IngestionPipeline
from extraction.entity_extractor import EntityExtractor
from extraction.domain_classifier import DomainClassifier
from graph.graph_builder import GraphBuilder
from vector.vector_store import VectorStore
from utils.config import validate_config


class TestDataIngester:
    """Ingest test data from evaluation datasets into the system."""
    
    def __init__(self, ingestion_pipeline: IngestionPipeline,
                 entity_extractor: EntityExtractor,
                 domain_classifier: DomainClassifier,
                 graph_builder: GraphBuilder,
                 vector_store: VectorStore):
        self.ingestion_pipeline = ingestion_pipeline
        self.entity_extractor = entity_extractor
        self.domain_classifier = domain_classifier
        self.graph_builder = graph_builder
        self.vector_store = vector_store
        self.ingested_files: List[Dict[str, Any]] = []
    
    def ingest_squad_v2(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Ingest SQuAD v2 contexts as text documents."""
        logger.info(f"Ingesting {num_samples} SQuAD v2 contexts...")
        
        try:
            dataset = load_dataset("rajpurkar/squad_v2", split="validation")
            ingested = []
            
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                
                # Skip unanswerable questions
                if not example["answers"]["text"]:
                    continue
                
                context = example["context"]
                question = example["question"]
                answer = example["answers"]["text"][0]
                
                # Create temporary text file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                    f.write(context)
                    temp_path = Path(f.name)
                
                try:
                    # Process file
                    result = self.ingestion_pipeline.process_file(temp_path)
                    
                    # Extract entities from chunks
                    extraction_results = self.entity_extractor.extract_from_chunks(result["chunks"])
                    
                    # Classify domain for each chunk
                    for j, chunk in enumerate(result["chunks"]):
                        entities = extraction_results[j].get("entities", []) if j < len(extraction_results) else []
                        domain_tags = self.domain_classifier.classify_chunk(chunk, entities)
                        chunk["metadata"]["domain_tags"] = domain_tags
                    
                    # Build knowledge graph
                    file_metadata = {
                        **result["metadata"],
                        "domain_tags": [tag for chunk in result["chunks"] 
                                       for tag in chunk.get("metadata", {}).get("domain_tags", [])]
                    }
                    self.graph_builder.build_from_extraction(extraction_results, file_metadata)
                    
                    # Index in vector store
                    self.vector_store.index_chunks(result["chunks"])
                    
                    ingested.append({
                        "file_id": result["file_id"],
                        "question": question,
                        "expected_answer": answer,
                        "context": context,
                        "dataset": "squad_v2"
                    })
                    
                    logger.info(f"Ingested SQuAD sample {i+1}/{num_samples}")
                
                except Exception as e:
                    logger.error(f"Failed to ingest SQuAD sample {i+1}: {e}")
                
                finally:
                    # Clean up temp file
                    if temp_path.exists():
                        temp_path.unlink()
            
            logger.success(f"Ingested {len(ingested)} SQuAD v2 contexts")
            return ingested
            
        except Exception as e:
            logger.error(f"Failed to ingest SQuAD v2: {e}")
            return []
    
    def ingest_docvqa(self, num_samples: int = 50) -> List[Dict[str, Any]]:
        """
        Ingest DocVQA images by extracting text via OCR and processing through pipeline.
        
        Extracts text from PIL Image objects in the dataset using GPT-4 Vision OCR,
        creates chunks, and processes them through entity extraction, domain classification,
        graph building, and vector indexing.
        """
        logger.info(f"Ingesting {num_samples} DocVQA images (extracting text via OCR)...")
        
        try:
            from ingestion.image_processor import ImageProcessor
            from ingestion.text_processor import TextProcessor
            import hashlib
            from datetime import datetime, timezone
            from io import BytesIO
            import tempfile
            from PIL import Image
            
            dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
            ingested = []
            
            # Initialize processors
            image_processor = ImageProcessor()
            text_processor = TextProcessor()
            
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                
                try:
                    question = example.get("question", "")
                    answers = example.get("answers", [])
                    if not answers:
                        continue
                    
                    # Get PIL Image from dataset
                    image = example.get("image")
                    if not image or not isinstance(image, Image.Image):
                        logger.warning(f"Skipping DocVQA sample {i+1}: No valid image found")
                        continue
                    
                    # Extract image reference
                    doc_id = example.get("docId", "")
                    question_id = example.get("questionId", "")
                    if doc_id and question_id:
                        image_ref = f"docvqa_{doc_id}_{question_id}"
                    elif doc_id:
                        image_ref = f"docvqa_{doc_id}"
                    else:
                        image_ref = f"docvqa_sample_{i}"
                    
                    # Get expected answer
                    expected_answer = answers[0] if isinstance(answers, list) and answers else str(answers) if answers else ""
                    
                    # Extract text from image using GPT-4 Vision OCR
                    logger.info(f"Extracting text from DocVQA image {i+1}/{num_samples} (docId: {doc_id})...")
                    
                    # Save PIL image to temporary file for processing
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        image.save(tmp_file.name, format='PNG')
                        temp_path = Path(tmp_file.name)
                    
                    try:
                        # Use image processor to extract text via OCR
                        # We'll use the GPT-4 Vision method directly
                        format_name = image.format or "PNG"
                        gpt4v_result = image_processor._caption_with_gpt4v(image, format_name)
                        extracted_text = gpt4v_result.get("extracted_text", "")
                        caption = gpt4v_result.get("caption", "")
                        
                        # Combine caption and extracted text for processing
                        combined_text = f"{caption}\n\n{extracted_text}".strip()
                        
                        if not combined_text:
                            logger.warning(f"No text extracted from DocVQA image {i+1}, skipping")
                            continue
                        
                        # Generate file_id
                        file_id = f"image_{hashlib.md5(image_ref.encode()).hexdigest()[:16]}"
                        
                        # Create chunks from extracted text
                        chunks = text_processor._chunk_text(combined_text, file_id)
                        
                        # Add image-specific metadata to chunks
                        now = datetime.now(timezone.utc)
                        for chunk in chunks:
                            chunk["metadata"].update({
                                "modality": "image",
                                "source_file": image_ref,
                                "doc_id": doc_id,
                                "question_id": question_id,
                                "extraction_method": "gpt4v_ocr",
                                "upload_timestamp": now.isoformat()
                            })
                        
                        # Create file metadata
                        file_metadata = {
                            "file_id": file_id,
                            "file_name": image_ref,
                            "modality": "image",
                            "upload_timestamp": now.isoformat(),
                            "created_at": now.isoformat(),
                            "source": "docvqa",
                            "doc_id": doc_id,
                            "question_id": question_id,
                            "extracted_text": extracted_text,
                            "caption": caption
                        }
                        
                        # Extract entities from chunks
                        extraction_results = self.entity_extractor.extract_from_chunks(chunks)
                        
                        # Classify domain for each chunk
                        for j, chunk in enumerate(chunks):
                            entities = extraction_results[j].get("entities", []) if j < len(extraction_results) else []
                            domain_tags = self.domain_classifier.classify_chunk(chunk, entities)
                            chunk["metadata"]["domain_tags"] = domain_tags
                        
                        # Add domain tags to file metadata
                        file_metadata["domain_tags"] = [
                            tag for chunk in chunks 
                            for tag in chunk.get("metadata", {}).get("domain_tags", [])
                        ]
                        
                        # Build knowledge graph
                        self.graph_builder.build_from_extraction(extraction_results, file_metadata)
                        
                        # Index in vector store
                        self.vector_store.index_chunks(chunks)
                        
                        ingested.append({
                            "file_id": file_id,
                            "question": question,
                            "expected_answer": expected_answer,
                            "answers": answers if isinstance(answers, list) else [str(answers)],
                            "image_reference": image_ref,
                            "doc_id": doc_id,
                            "question_id": question_id,
                            "dataset": "docvqa"
                        })
                        
                        logger.info(f"Ingested DocVQA sample {i+1}/{num_samples}: {image_ref}")
                    
                    finally:
                        # Clean up temp file
                        if temp_path.exists():
                            temp_path.unlink()
                
                except Exception as e:
                    logger.error(f"Failed to ingest DocVQA sample {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            logger.success(f"Ingested {len(ingested)} DocVQA samples")
            if len(ingested) < num_samples:
                logger.warning(f"Only ingested {len(ingested)} samples (requested {num_samples}) - some may have been skipped")
            
            return ingested
            
        except Exception as e:
            logger.error(f"Failed to ingest DocVQA: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def ingest_fleurs(self, num_samples: int = 50) -> List[Dict[str, Any]]:
        """
        Ingest FLEURS transcriptions from TSV file (no audio processing).
        
        Reads transcriptions from TSV file, creates chunks, and processes them
        through entity extraction, domain classification, graph building, and
        vector indexing - all without processing audio files.
        """
        logger.info(f"Ingesting {num_samples} FLEURS transcriptions from TSV file (no audio processing)...")
        
        tsv_file = Path(__file__).parent.parent / "fleurs-en_us-dataset.tsv"
        ingested = []
        
        if not tsv_file.exists():
            logger.warning(f"FLEURS TSV file not found: {tsv_file}")
            return []
        
        # Read TSV file and process transcriptions
        import csv
        import hashlib
        from datetime import datetime, timezone
        from ingestion.text_processor import TextProcessor
        
        # Initialize text processor for chunking
        text_processor = TextProcessor()
        
        with open(tsv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            
            for i, row in enumerate(reader):
                if i >= num_samples:
                    break
                
                try:
                    filename = row.get("filename", "")
                    raw_transcription = row.get("raw_transcription", "")
                    
                    if not filename or not raw_transcription:
                        continue
                    
                    # Generate file_id based on filename (similar to audio processor)
                    file_id = f"audio_{hashlib.md5(filename.encode()).hexdigest()[:16]}"
                    
                    # Create chunks from transcription using text processor's chunking method
                    chunks = text_processor._chunk_text(raw_transcription, file_id)
                    
                    # Add audio-specific metadata to chunks
                    now = datetime.now(timezone.utc)
                    for chunk in chunks:
                        chunk["metadata"].update({
                            "modality": "audio",
                            "source_file": filename,
                            "transcription_source": "tsv",
                            "upload_timestamp": now.isoformat()
                        })
                    
                    # Create file metadata
                    file_metadata = {
                        "file_id": file_id,
                        "file_name": filename,
                        "modality": "audio",
                        "upload_timestamp": now.isoformat(),
                        "created_at": now.isoformat(),
                        "source": "fleurs_tsv",
                        "transcription": raw_transcription
                    }
                    
                    # Extract entities from chunks
                    extraction_results = self.entity_extractor.extract_from_chunks(chunks)
                    
                    # Classify domain for each chunk
                    for j, chunk in enumerate(chunks):
                        entities = extraction_results[j].get("entities", []) if j < len(extraction_results) else []
                        domain_tags = self.domain_classifier.classify_chunk(chunk, entities)
                        chunk["metadata"]["domain_tags"] = domain_tags
                    
                    # Add domain tags to file metadata
                    file_metadata["domain_tags"] = [
                        tag for chunk in chunks 
                        for tag in chunk.get("metadata", {}).get("domain_tags", [])
                    ]
                    
                    # Build knowledge graph
                    self.graph_builder.build_from_extraction(extraction_results, file_metadata)
                    
                    # Index in vector store
                    self.vector_store.index_chunks(chunks)
                    
                    ingested.append({
                        "file_id": file_id,
                        "filename": filename,
                        "expected_transcription": raw_transcription,
                        "dataset": "fleurs"
                    })
                    
                    logger.info(f"Ingested FLEURS transcription {i+1}/{num_samples}: {filename}")
                
                except Exception as e:
                    logger.error(f"Failed to ingest FLEURS transcription {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
        
        logger.success(f"Ingested {len(ingested)} FLEURS transcriptions from TSV file")
        return ingested
    
    def ingest_all(self, squad_samples: int = 100, docvqa_samples: int = 0, 
                   fleurs_samples: int = 0) -> Dict[str, Any]:
        """
        Ingest test datasets.
        
        By default, only ingests SQuAD v2 data. Set docvqa_samples and fleurs_samples > 0
        to include those datasets.
        """
        logger.info("Starting test data ingestion...")
        
        results = {
            "squad_v2": self.ingest_squad_v2(squad_samples),
            "docvqa": self.ingest_docvqa(docvqa_samples) if docvqa_samples > 0 else [],
            "fleurs": self.ingest_fleurs(fleurs_samples) if fleurs_samples > 0 else []
        }
        
        total_ingested = sum(len(v) for v in results.values())
        logger.success(f"Total ingested: {total_ingested} files")
        
        # Save ingestion metadata
        metadata_file = Path(__file__).parent / "test_data" / "ingestion_metadata.json"
        metadata_file.parent.mkdir(exist_ok=True)
        
        with open(metadata_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Ingestion metadata saved to {metadata_file}")
        
        return results


def main():
    """Main function to run test data ingestion."""
    # Validate config
    is_valid, error = validate_config()
    if not is_valid:
        logger.error(f"Configuration error: {error}")
        return 1
    
    try:
        # Initialize components
        logger.info("Initializing system components...")
        
        from graph.neo4j_client import Neo4jClient
        from vector.qdrant_client import QdrantClientWrapper
        from vector.embedding_service import EmbeddingService
        
        # Database clients
        neo4j_client = Neo4jClient()
        qdrant_client = QdrantClientWrapper()
        
        # Services
        embedding_service = EmbeddingService()
        vector_store = VectorStore(qdrant_client, embedding_service)
        
        # Pipelines and extractors
        ingestion_pipeline = IngestionPipeline()
        entity_extractor = EntityExtractor()
        domain_classifier = DomainClassifier()
        graph_builder = GraphBuilder(neo4j_client)
        
        # Create ingester
        ingester = TestDataIngester(
            ingestion_pipeline=ingestion_pipeline,
            entity_extractor=entity_extractor,
            domain_classifier=domain_classifier,
            graph_builder=graph_builder,
            vector_store=vector_store
        )
        
        # Ingest all test data
        results = ingester.ingest_all(
            squad_samples=100,
            docvqa_samples=50,
            fleurs_samples=50
        )
        
        logger.success("Test data ingestion completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

