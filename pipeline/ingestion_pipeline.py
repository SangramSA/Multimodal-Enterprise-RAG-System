"""End-to-end ingestion pipeline."""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from utils.logging import logger
from utils.errors import FileError, ProcessingError
from ingestion.text_processor import TextProcessor
from ingestion.image_processor import ImageProcessor
from ingestion.audio_processor import AudioProcessor


class IngestionPipeline:
    """Orchestrates the ingestion of multimodal files."""
    
    def __init__(self, 
                 entity_extractor=None,
                 domain_classifier=None,
                 graph_builder=None,
                 vector_store=None):
        """
        Initialize ingestion pipeline.
        
        Args:
            entity_extractor: Optional EntityExtractor instance
            domain_classifier: Optional DomainClassifier instance
            graph_builder: Optional GraphBuilder instance
            vector_store: Optional VectorStore instance
        """
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        self.processors = {
            ".pdf": self.text_processor,
            ".txt": self.text_processor,
            ".jpg": self.image_processor,
            ".jpeg": self.image_processor,
            ".png": self.image_processor,
            ".mp3": self.audio_processor,
            ".wav": self.audio_processor
        }
        
        # Optional components for complete ingestion
        self.entity_extractor = entity_extractor
        self.domain_classifier = domain_classifier
        self.graph_builder = graph_builder
        self.vector_store = vector_store
    
    def get_processor(self, file_path: Path):
        """Get appropriate processor for file type."""
        suffix = file_path.suffix.lower()
        processor = self.processors.get(suffix)
        if not processor:
            raise FileError(f"No processor available for file type: {suffix}")
        return processor
    
    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file through the ingestion pipeline."""
        try:
            processor = self.get_processor(file_path)
            result = processor.process_file(file_path)
            
            # Add pipeline-level metadata
            result["pipeline_timestamp"] = datetime.utcnow().isoformat()
            result["pipeline_version"] = "1.0"
            logger.info(f"Processed file: {file_path}")
            return result
        except Exception as e:
            logger.error(f"Pipeline error processing {file_path}: {e}")
            raise ProcessingError(f"Failed to process file: {e}")
    
    def process_files(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """Process multiple files."""
        results = []
        for file_path in file_paths:
            try:
                result = self.process_file(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append({
                    "file_path": str(file_path),
                    "processing_status": "failed",
                    "error": str(e)
                })
        return results
    
    def process_and_index_file(self, file_path: Path, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Complete ingestion pipeline: process file, extract entities, classify domains,
        build knowledge graph, and index in vector store.
        
        This is the full end-to-end ingestion that stores data in Neo4j and Qdrant.
        
        Args:
            file_path: Path to file to process
            force_reprocess: If True, reprocess even if file already exists (default: False)
            
        Returns:
            Dictionary with processing results including:
            - file_id, modality, chunks (from process_file)
            - extraction_results (entities and relationships)
            - graph_summary (nodes and relationships created)
            - vector_indexed (number of chunks indexed)
            - processing_status
            - cached: True if results were retrieved from cache, False if newly processed
        """
        if not all([self.entity_extractor, self.domain_classifier, 
                   self.graph_builder, self.vector_store]):
            raise ProcessingError(
                "Complete ingestion requires entity_extractor, domain_classifier, "
                "graph_builder, and vector_store. Use process_file() for chunking only, "
                "or initialize IngestionPipeline with all components."
            )
        
        try:
            # Check if file already exists (based on content hash)
            # Generate file_id first to check for duplicates
            file_id = None
            for processor in [self.text_processor, self.image_processor, self.audio_processor]:
                try:
                    if processor.validate(file_path):
                        file_id = processor.generate_file_id(file_path)
                        break
                except Exception:
                    continue  # Try next processor
            
            if not file_id:
                raise ProcessingError(f"Could not determine file type for {file_path}")
            
            # Check for existing file in Neo4j
            if not force_reprocess:
                existing_node = self.graph_builder.client.find_content_node_by_file_id(file_id)
                if existing_node:
                    logger.info(f"File {file_path.name} (ID: {file_id}) already exists in knowledge graph")
                    logger.info("Retrieving cached chunks from vector store...")
                    
                    # Retrieve existing chunks from Qdrant
                    cached_chunks = self.vector_store.get_chunks_by_file_id(file_id)
                    
                    if cached_chunks:
                        logger.success(f"Retrieved {len(cached_chunks)} cached chunks for {file_path.name}")
                        logger.info("Skipping OpenAI API calls (GPT-4 Vision/Whisper) - using cached data")
                        return {
                            "file_id": file_id,
                            "file_name": file_path.name,
                            "file_path": str(file_path),
                            "modality": existing_node.get("modality", "document"),
                            "chunks": cached_chunks,
                            "processing_status": "cached",
                            "cached": True,
                            "extraction_results": [],  # Not retrieved from cache
                            "graph_summary": {
                                "nodes_created": 0,
                                "relationships_created": 0,
                                "content_node_id": file_id
                            },
                            "vector_indexed": len(cached_chunks),
                            "cross_modal_links": {
                                "same_session_links": 0,
                                "cross_session_links": 0,
                                "total_links": 0
                            }
                        }
                    else:
                        logger.warning(f"File exists in graph but chunks not found in vector store, reprocessing...")
            
            # Step 1: Process file into chunks
            logger.info(f"Step 1/5: Processing file {file_path.name}")
            result = self.process_file(file_path)
            chunks = result.get("chunks", [])
            
            if not chunks:
                logger.warning(f"No chunks created from {file_path.name}")
                return {
                    **result,
                    "processing_status": "partial",
                    "extraction_results": [],
                    "graph_summary": {"nodes_created": 0, "relationships_created": 0},
                    "vector_indexed": 0
                }
            
            # Step 2: Extract entities and relationships
            logger.info(f"Step 2/5: Extracting entities from {len(chunks)} chunks")
            extraction_results = self.entity_extractor.extract_from_chunks(chunks)
            
            # Step 3: Classify domain for each chunk
            logger.info(f"Step 3/5: Classifying domains for {len(chunks)} chunks")
            for j, chunk in enumerate(chunks):
                entities = extraction_results[j].get("entities", []) if j < len(extraction_results) else []
                try:
                    domain_tags = set(self.domain_classifier.classify_chunk(chunk, entities) or [])
                    # Store as sorted list for serialization while ensuring uniqueness
                    chunk["metadata"]["domain_tags"] = sorted(domain_tags)
                except Exception as e:
                    logger.warning(f"Domain classification failed for chunk {j}: {e}")
                    chunk["metadata"]["domain_tags"] = []
            
            # Step 4: Build knowledge graph
            logger.info(f"Step 4a/5: Building knowledge graph")
            aggregated_domain_tags = set()
            for chunk in chunks:
                aggregated_domain_tags.update(chunk.get("metadata", {}).get("domain_tags", []))
            
            file_metadata = {
                **result["metadata"],
                "domain_tags": sorted(aggregated_domain_tags)
            }
            graph_summary = self.graph_builder.build_from_extraction(extraction_results, file_metadata)
            
            # Step 4b: Link cross-modal entities
            logger.info("Step 4b/5: Linking cross-modal entities")
            entity_links = self.entity_extractor.link_entities_across_modalities(extraction_results, chunks)
            cross_modal_result = self.graph_builder.link_cross_modal_entities(entity_links)
            cross_modal_links = cross_modal_result.get("total_links", 0)
            cross_session_links = cross_modal_result.get("cross_session_links", 0)
            
            # Step 5: Index in vector store
            logger.info(f"Step 5/5: Indexing {len(chunks)} chunks in vector store")
            vector_indexed = self.vector_store.index_chunks(chunks)
            
            logger.success(
                f"Complete ingestion finished: {vector_indexed} chunks indexed, "
                f"{graph_summary.get('nodes_created', 0)} nodes, "
                f"{graph_summary.get('relationships_created', 0)} relationships, "
                f"{cross_modal_links} cross-modal links ({cross_session_links} cross-session)"
            )
            
            return {
                **result,
                "processing_status": "complete",
                "extraction_results": extraction_results,
                "graph_summary": graph_summary,
                "vector_indexed": vector_indexed,
                "cross_modal_links": cross_modal_result,
                "cached": False
            }
            
        except Exception as e:
            logger.error(f"Complete ingestion failed for {file_path}: {e}")
            raise ProcessingError(f"Failed to complete ingestion: {e}")
    
    def process_and_index_files(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """Process and index multiple files through complete pipeline."""
        results = []
        for file_path in file_paths:
            try:
                result = self.process_and_index_file(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process and index {file_path}: {e}")
                results.append({
                    "file_path": str(file_path),
                    "processing_status": "failed",
                    "error": str(e)
                })
        return results

