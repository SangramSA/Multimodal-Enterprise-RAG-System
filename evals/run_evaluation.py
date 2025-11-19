"""Main script to run the complete evaluation pipeline."""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from evals.test_suite import TestSuite
from evals.ingest_test_data import TestDataIngester
from utils.config import validate_config, EVAL_LOG_PATH
from utils.logging import logger as app_logger


def run_evaluation(ingest_data: bool = True, squad_samples: int = 100,
                   docvqa_samples: int = 50, fleurs_samples: int = 50):
    """
    Run the complete evaluation pipeline.
    
    Args:
        ingest_data: Whether to ingest test data first (if False, assumes data is already ingested)
        squad_samples: Number of SQuAD v2 samples to use
        docvqa_samples: Number of DocVQA samples to use
        fleurs_samples: Number of FLEURS samples to use
    """
    # Validate config
    is_valid, error = validate_config()
    if not is_valid:
        logger.error(f"Configuration error: {error}")
        return None
    
    try:
        # Initialize all components
        logger.info("Initializing system components...")
        
        from graph.neo4j_client import Neo4jClient
        from vector.qdrant_client import QdrantClientWrapper
        from vector.embedding_service import EmbeddingService
        from pipeline.ingestion_pipeline import IngestionPipeline
        from pipeline.query_pipeline import QueryPipeline
        from agents.retrieval_agent import RetrievalAgent
        from search.hybrid_search import HybridSearch
        from search.keyword_search import KeywordSearch
        from search.vector_search import VectorSearch
        from search.graph_search import GraphSearch
        from extraction.entity_extractor import EntityExtractor
        from extraction.domain_classifier import DomainClassifier
        from graph.graph_builder import GraphBuilder
        from vector.vector_store import VectorStore
        
        # Database clients
        neo4j_client = Neo4jClient()
        qdrant_client = QdrantClientWrapper()
        
        # Services
        embedding_service = EmbeddingService()
        vector_store = VectorStore(qdrant_client, embedding_service)
        
        # Search components
        keyword_search = KeywordSearch(vector_store)
        vector_search = VectorSearch(vector_store)
        graph_search = GraphSearch(neo4j_client)
        hybrid_search = HybridSearch(keyword_search, vector_search, graph_search)
        
        # Agent
        retrieval_agent = RetrievalAgent(hybrid_search)
        
        # Pipelines
        ingestion_pipeline = IngestionPipeline()
        query_pipeline = QueryPipeline(retrieval_agent)
        
        # Extractors
        entity_extractor = EntityExtractor()
        domain_classifier = DomainClassifier()
        graph_builder = GraphBuilder(neo4j_client)
        
        # Step 1: Ingest test data (if requested)
        if ingest_data:
            logger.info("=" * 60)
            logger.info("STEP 1: Ingesting test data into the system")
            logger.info("=" * 60)
            
            ingester = TestDataIngester(
                ingestion_pipeline=ingestion_pipeline,
                entity_extractor=entity_extractor,
                domain_classifier=domain_classifier,
                graph_builder=graph_builder,
                vector_store=vector_store
            )
            
            ingestion_results = ingester.ingest_all(
                squad_samples=squad_samples,
                docvqa_samples=docvqa_samples,
                fleurs_samples=fleurs_samples
            )
            
            logger.success("Test data ingestion completed")
        else:
            logger.info("Skipping data ingestion (assuming data is already ingested)")
            ingestion_results = None
        
        # Step 2: Build test suite
        logger.info("=" * 60)
        logger.info("STEP 2: Building test suite")
        logger.info("=" * 60)
        
        test_suite = TestSuite()
        test_cases = test_suite.build_test_suite()
        
        logger.success(f"Test suite built with {len(test_cases)} test cases")
        
        # Step 3: Run evaluation
        logger.info("=" * 60)
        logger.info("STEP 3: Running evaluation")
        logger.info("=" * 60)
        
        evaluation_results = test_suite.evaluate(query_pipeline, test_cases=test_cases)
        
        # Step 4: Save results
        logger.info("=" * 60)
        logger.info("STEP 4: Saving evaluation results")
        logger.info("=" * 60)
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "ingestion_results": ingestion_results,
            "evaluation_results": evaluation_results,
            "test_cases_count": len(test_cases)
        }
        
        # Save to file
        log_path = Path(EVAL_LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.success(f"Evaluation results saved to {log_path}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {evaluation_results.get('total_tests', 0)}")
        logger.info("")
        logger.info("Custom Metrics:")
        logger.info(f"  Precision@5: {evaluation_results.get('avg_precision_at_5', 0):.3f}")
        logger.info(f"  Recall@5: {evaluation_results.get('avg_recall_at_5', 0):.3f}")
        logger.info(f"  F1 Score: {evaluation_results.get('avg_f1', 0):.3f}")
        logger.info(f"  Exact Match Rate: {evaluation_results.get('exact_match_rate', 0):.3f}")
        logger.info(f"  Avg Semantic Similarity: {evaluation_results.get('avg_semantic_similarity', 0):.3f}")
        logger.info("")
        logger.info("DeepEval Metrics:")
        hallucination_score = evaluation_results.get('avg_hallucination_score')
        if hallucination_score is not None:
            logger.info(f"  Hallucination Score: {hallucination_score:.3f} (lower is better)")
            logger.info(f"  Hallucination Rate: {evaluation_results.get('hallucination_rate', 0):.3f}")
        else:
            logger.info("  Hallucination Score: N/A (no context provided)")
        
        relevancy = evaluation_results.get('avg_answer_relevancy')
        if relevancy is not None:
            logger.info(f"  Answer Relevancy: {relevancy:.3f}")
        else:
            logger.info("  Answer Relevancy: N/A (no expected output)")
        
        faithfulness = evaluation_results.get('avg_faithfulness')
        if faithfulness is not None:
            logger.info(f"  Faithfulness: {faithfulness:.3f}")
        else:
            logger.info("  Faithfulness: N/A (no context provided)")
        logger.info("")
        logger.info("Performance:")
        latency = evaluation_results.get('latency', {})
        logger.info(f"  Mean Latency: {latency.get('mean', 0):.3f}s")
        logger.info(f"  P95 Latency: {latency.get('p95', 0):.3f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument("--skip-ingestion", action="store_true",
                        help="Skip data ingestion (assumes data is already ingested)")
    parser.add_argument("--squad-samples", type=int, default=100,
                        help="Number of SQuAD v2 samples (default: 100)")
    parser.add_argument("--docvqa-samples", type=int, default=50,
                        help="Number of DocVQA samples (default: 50)")
    parser.add_argument("--fleurs-samples", type=int, default=50,
                        help="Number of FLEURS samples (default: 50)")
    
    args = parser.parse_args()
    
    results = run_evaluation(
        ingest_data=not args.skip_ingestion,
        squad_samples=args.squad_samples,
        docvqa_samples=args.docvqa_samples,
        fleurs_samples=args.fleurs_samples
    )
    
    if results:
        logger.success("Evaluation completed successfully!")
        return 0
    else:
        logger.error("Evaluation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

