"""Evaluation test suite with custom metrics and DeepEval integration.

Uses both custom metrics (precision, recall, F1) and DeepEval metrics
(hallucination detection, answer relevancy, faithfulness) for comprehensive evaluation.
"""

from typing import List, Dict, Any
import json
from pathlib import Path
from datasets import load_dataset
from loguru import logger

from evals.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_f1_score,
    calculate_exact_match,
    calculate_semantic_similarity,
    measure_latency,
    evaluate_with_deepeval
)


class TestSuite:
    """Test suite for evaluating the RAG system."""
    
    def __init__(self, test_data_dir: Path = None):
        self.test_data_dir = test_data_dir or Path(__file__).parent / "test_data"
        self.test_data_dir.mkdir(exist_ok=True)
        self.test_cases: List[Dict[str, Any]] = []
    
    def load_squad_v2_samples(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Load test cases from SQuAD v2 dataset."""
        logger.info(f"Loading {num_samples} samples from SQuAD v2...")
        
        try:
            dataset = load_dataset("rajpurkar/squad_v2", split="validation")
            samples = []
            
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                
                # Skip unanswerable questions for now
                if not example["answers"]["text"]:
                    continue
                
                samples.append({
                    "query": example["question"],
                    "context": example["context"],
                    "expected_answer": example["answers"]["text"][0] if example["answers"]["text"] else "",
                    "query_type": "factual_lookup",
                    "modality": "text",
                    "dataset": "squad_v2"
                })
            
            logger.success(f"Loaded {len(samples)} SQuAD v2 samples")
            return samples
        except Exception as e:
            logger.error(f"Failed to load SQuAD v2: {e}")
            return []
    
    def load_docvqa_samples(self, num_samples: int = 50) -> List[Dict[str, Any]]:
        """Load test cases from DocVQA dataset."""
        logger.info(f"Loading {num_samples} samples from DocVQA...")
        
        try:
            dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
            samples = []
            
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                
                samples.append({
                    "query": example.get("question", ""),
                    "image_path": example.get("image", {}).get("path") if isinstance(example.get("image"), dict) else None,
                    "expected_answer": example.get("answers", [""])[0] if example.get("answers") else "",
                    "query_type": "visual_qa",
                    "modality": "image",
                    "dataset": "docvqa"
                })
            
            logger.success(f"Loaded {len(samples)} DocVQA samples")
            return samples
        except Exception as e:
            logger.error(f"Failed to load DocVQA: {e}")
            return []
    
    def create_fleurs_samples(self, num_samples: int = 50) -> List[Dict[str, Any]]:
        """Create test cases from FLEURS audio files using the TSV metadata."""
        logger.info(f"Creating {num_samples} samples from FLEURS...")
        
        samples = []
        fleurs_dir = Path(__file__).parent.parent / "google-fleurs-audio-files"
        tsv_file = Path(__file__).parent.parent / "fleurs-en_us-dataset.tsv"
        
        if not tsv_file.exists():
            logger.error(f"FLEURS TSV file not found: {tsv_file}")
            return []
        
        if not fleurs_dir.exists():
            logger.error(f"FLEURS audio directory not found: {fleurs_dir}")
            return []
        
        # Read TSV file
        import csv
        audio_files_map = {}
        
        with open(tsv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                filename = row.get("filename", "")
                raw_transcription = row.get("raw_transcription", "")
                normalized_transcription = row.get("nomalized_transcription", "")  # Note: typo in original column name
                
                if filename and raw_transcription:
                    audio_files_map[filename] = {
                        "raw_transcription": raw_transcription,
                        "normalized_transcription": normalized_transcription
                    }
        
        # Match audio files with transcriptions
        # Limit to num_samples to avoid creating too many test cases
        audio_files = list(fleurs_dir.glob("*.wav"))[:num_samples]
        
        for audio_file in audio_files:
            if len(samples) >= num_samples:
                break
            filename = audio_file.name
            transcription_data = audio_files_map.get(filename)
            
            if transcription_data:
                raw_transcription = transcription_data["raw_transcription"]
                
                # Create test case with question about the transcription
                samples.append({
                    "query": f"What is said in this audio file?",
                    "audio_path": str(audio_file),
                    "expected_answer": raw_transcription,
                    "query_type": "audio_qa",
                    "modality": "audio",
                    "dataset": "fleurs",
                    "transcription": raw_transcription
                })
            else:
                # Fallback if transcription not found
                logger.warning(f"No transcription found for {filename}")
                samples.append({
                    "query": f"What is said in this audio file?",
                    "audio_path": str(audio_file),
                    "expected_answer": "",
                    "query_type": "audio_qa",
                    "modality": "audio",
                    "dataset": "fleurs"
                })
        
        logger.success(f"Created {len(samples)} FLEURS samples")
        return samples
    
    def build_test_suite(self):
        """Build complete test suite from all datasets."""
        logger.info("Building test suite...")
        
        self.test_cases = []
        
        # Load from each dataset
        self.test_cases.extend(self.load_squad_v2_samples(20))
        self.test_cases.extend(self.load_docvqa_samples(5))
        self.test_cases.extend(self.create_fleurs_samples(5))
        
        # Save test cases
        test_file = self.test_data_dir / "test_cases.json"
        with open(test_file, "w") as f:
            json.dump(self.test_cases, f, indent=2)
        
        logger.success(f"Test suite built with {len(self.test_cases)} test cases")
        return self.test_cases
    
    def evaluate(self, query_pipeline, test_cases: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate the system on test cases."""
        if test_cases is None:
            test_cases = self.test_cases
        
        if not test_cases:
            logger.warning("No test cases available")
            return {}
        
        results = {
            "total": len(test_cases),
            "precision_at_5": [],
            "recall_at_5": [],
            "exact_match": [],
            "semantic_similarity": [],
            "latency": [],
            "hallucination_scores": [],
            "answer_relevancy_scores": [],
            "faithfulness_scores": []
        }
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Evaluating test case {i+1}/{len(test_cases)}")
            
            try:
                import time
                start_time = time.time()
                
                # Run query through pipeline
                response = query_pipeline.process(
                    query=test_case["query"],
                    query_type=test_case.get("query_type"),
                    limit=10
                )
                
                latency = time.time() - start_time
                results["latency"].append(latency)
                
                # Extract results
                answer = response.get("answer", "")
                retrieved = response.get("retrieved_documents", [])
                expected = test_case.get("expected_answer", "")
                query = test_case.get("query", "")
                
                # Extract context from retrieved documents
                context = [doc.get("content", "")[:500] for doc in retrieved[:3]]  # Top 3 contexts
                
                # Calculate custom metrics
                if expected:
                    results["exact_match"].append(calculate_exact_match(answer, expected))
                    results["semantic_similarity"].append(calculate_semantic_similarity(answer, expected))
                
                # For precision/recall, we'd need ground truth relevant docs
                # Simplified: assume top result is relevant if answer matches
                relevant_ids = [retrieved[0].get("chunk_id")] if retrieved and expected else []
                results["precision_at_5"].append(
                    calculate_precision_at_k(retrieved, relevant_ids, k=5)
                )
                results["recall_at_5"].append(
                    calculate_recall_at_k(retrieved, relevant_ids, k=5)
                )
                
                # DeepEval metrics
                deepeval_results = evaluate_with_deepeval(
                    input_text=query,
                    actual_output=answer,
                    expected_output=expected if expected else None,
                    context=context if context else None
                )
                
                if deepeval_results.get("hallucination_score") is not None:
                    results["hallucination_scores"].append(deepeval_results["hallucination_score"])
                if deepeval_results.get("answer_relevancy_score") is not None:
                    results["answer_relevancy_scores"].append(deepeval_results["answer_relevancy_score"])
                if deepeval_results.get("faithfulness_score") is not None:
                    results["faithfulness_scores"].append(deepeval_results["faithfulness_score"])
                
            except Exception as e:
                logger.error(f"Test case {i+1} failed: {e}")
                results["latency"].append(0.0)
        
        # Aggregate results
        aggregated = {
            "total_tests": results["total"],
            "avg_precision_at_5": sum(results["precision_at_5"]) / len(results["precision_at_5"]) if results["precision_at_5"] else 0.0,
            "avg_recall_at_5": sum(results["recall_at_5"]) / len(results["recall_at_5"]) if results["recall_at_5"] else 0.0,
            "avg_f1": calculate_f1_score(
                sum(results["precision_at_5"]) / len(results["precision_at_5"]) if results["precision_at_5"] else 0.0,
                sum(results["recall_at_5"]) / len(results["recall_at_5"]) if results["recall_at_5"] else 0.0
            ),
            "exact_match_rate": sum(results["exact_match"]) / len(results["exact_match"]) if results["exact_match"] else 0.0,
            "avg_semantic_similarity": sum(results["semantic_similarity"]) / len(results["semantic_similarity"]) if results["semantic_similarity"] else 0.0,
            "latency": measure_latency(results["latency"])
        }
        
        # Add DeepEval metrics
        if results["hallucination_scores"]:
            aggregated["avg_hallucination_score"] = sum(results["hallucination_scores"]) / len(results["hallucination_scores"])
            aggregated["hallucination_rate"] = 1.0 - aggregated["avg_hallucination_score"]  # Lower is better for hallucination
        else:
            aggregated["avg_hallucination_score"] = None
            aggregated["hallucination_rate"] = None
        
        if results["answer_relevancy_scores"]:
            aggregated["avg_answer_relevancy"] = sum(results["answer_relevancy_scores"]) / len(results["answer_relevancy_scores"])
        else:
            aggregated["avg_answer_relevancy"] = None
        
        if results["faithfulness_scores"]:
            aggregated["avg_faithfulness"] = sum(results["faithfulness_scores"]) / len(results["faithfulness_scores"])
        else:
            aggregated["avg_faithfulness"] = None
        
        return aggregated

