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
        """Ingest DocVQA images."""
        logger.info(f"Ingesting {num_samples} DocVQA images...")
        
        try:
            dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
            ingested = []
            
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                
                question = example.get("question", "")
                answers = example.get("answers", [])
                if not answers:
                    continue
                
                # Get image - this is more complex as we need to download it
                # For now, we'll skip actual image ingestion and just log
                # In production, you'd download and process the images
                logger.warning(f"DocVQA image ingestion not fully implemented - skipping image {i+1}")
                # TODO: Implement image download and processing
                
            logger.success(f"Processed {len(ingested)} DocVQA images")
            return ingested
            
        except Exception as e:
            logger.error(f"Failed to ingest DocVQA: {e}")
            return []
    
    def ingest_fleurs(self, num_samples: int = 50) -> List[Dict[str, Any]]:
        """Ingest FLEURS audio files using TSV metadata."""
        logger.info(f"Ingesting {num_samples} FLEURS audio files...")
        
        fleurs_dir = Path(__file__).parent.parent / "google-fleurs-audio-files"
        tsv_file = Path(__file__).parent.parent / "fleurs-en_us-dataset.tsv"
        ingested = []
        
        if not fleurs_dir.exists():
            logger.warning(f"FLEURS directory not found: {fleurs_dir}")
            return []
        
        if not tsv_file.exists():
            logger.warning(f"FLEURS TSV file not found: {tsv_file}")
            return []
        
        # Read TSV file to get transcriptions
        import csv
        audio_files_map = {}
        
        with open(tsv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                filename = row.get("filename", "")
                raw_transcription = row.get("raw_transcription", "")
                
                if filename and raw_transcription:
                    audio_files_map[filename] = raw_transcription
        
        # Get audio files and match with transcriptions
        audio_files = list(fleurs_dir.glob("*.wav"))[:num_samples]
        
        for i, audio_file in enumerate(audio_files):
            try:
                filename = audio_file.name
                expected_transcription = audio_files_map.get(filename, "")
                
                # Process audio file
                result = self.ingestion_pipeline.process_file(audio_file)
                
                # Extract entities from transcription chunks
                extraction_results = self.entity_extractor.extract_from_chunks(result["chunks"])
                
                # Classify domain
                for j, chunk in enumerate(result["chunks"]):
                    entities = extraction_results[j].get("entities", []) if j < len(extraction_results) else []
                    domain_tags = self.domain_classifier.classify_chunk(chunk, entities)
                    chunk["metadata"]["domain_tags"] = domain_tags
                
                # Build graph
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
                    "audio_path": str(audio_file),
                    "expected_transcription": expected_transcription,
                    "dataset": "fleurs"
                })
                
                logger.info(f"Ingested FLEURS audio {i+1}/{len(audio_files)}: {filename}")
            
            except Exception as e:
                logger.error(f"Failed to ingest FLEURS audio {i+1}: {e}")
        
        logger.success(f"Ingested {len(ingested)} FLEURS audio files")
        return ingested
    
    def ingest_all(self, squad_samples: int = 100, docvqa_samples: int = 50, 
                   fleurs_samples: int = 50) -> Dict[str, Any]:
        """Ingest all test datasets."""
        logger.info("Starting test data ingestion...")
        
        results = {
            "squad_v2": self.ingest_squad_v2(squad_samples),
            "docvqa": self.ingest_docvqa(docvqa_samples),
            "fleurs": self.ingest_fleurs(fleurs_samples)
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

