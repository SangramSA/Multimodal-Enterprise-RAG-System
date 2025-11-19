"""Evaluation test suite with DeepEval integration.

Uses DeepEval metrics (hallucination detection, answer relevancy, faithfulness)
for comprehensive evaluation. Results are automatically uploaded to Confident AI
for hosted reporting and tracking.
"""

from typing import List, Dict, Any
import json
from pathlib import Path
from datasets import load_dataset
from loguru import logger

from evals.metrics import measure_latency, evaluate_with_deepeval


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
                    "query": f"What is said in this audio file: {filename}?",
                    "filename": str(filename),
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
                    "query": f"What is said in this audio file: {filename}?",
                    "filename": str(filename),
                    "expected_answer": "",
                    "query_type": "audio_qa",
                    "modality": "audio",
                    "dataset": "fleurs",
                    "transcription": ""
                })
        
        logger.success(f"Created {len(samples)} FLEURS samples")
        return samples
    
    def build_test_suite(self, squad_samples: int = 20, docvqa_samples: int = 0, fleurs_samples: int = 0):
        """
        Build complete test suite from datasets.
        
        By default, only loads SQuAD v2 samples. Set docvqa_samples and fleurs_samples > 0
        to include those datasets.
        
        Args:
            squad_samples: Number of SQuAD v2 samples to load
            docvqa_samples: Number of DocVQA samples to load (0 to skip)
            fleurs_samples: Number of FLEURS samples to create (0 to skip)
        """
        logger.info("Building test suite...")
        
        self.test_cases = []
        
        # Load from SQuAD v2 (always included)
        self.test_cases.extend(self.load_squad_v2_samples(squad_samples))
        
        # Load from DocVQA (only if samples > 0)
        if docvqa_samples > 0:
            self.test_cases.extend(self.load_docvqa_samples(docvqa_samples))
        else:
            logger.info("Skipping DocVQA samples (docvqa_samples=0)")
        
        # Load from FLEURS (only if samples > 0)
        if fleurs_samples > 0:
            self.test_cases.extend(self.create_fleurs_samples(fleurs_samples))
        else:
            logger.info("Skipping FLEURS samples (fleurs_samples=0)")
        
        # Save test cases
        test_file = self.test_data_dir / "test_cases.json"
        with open(test_file, "w") as f:
            json.dump(self.test_cases, f, indent=2)
        
        logger.success(f"Test suite built with {len(self.test_cases)} test cases")
        return self.test_cases
    
    def evaluate(self, query_pipeline, test_cases: List[Dict[str, Any]] = None, 
                 max_workers: int = 1, use_automatic_upload: bool = False) -> Dict[str, Any]:
        """
        Evaluate the system on test cases.
        
        Args:
            query_pipeline: The query pipeline to use for evaluation
            test_cases: List of test cases to evaluate (uses self.test_cases if None)
            max_workers: Number of parallel workers (1 = sequential, >1 = parallel)
            use_automatic_upload: If True, use DeepEval's evaluate() for automatic Confident AI uploads
        
        Returns:
            Dictionary with evaluation results
        """
        if test_cases is None:
            test_cases = self.test_cases
        
        if not test_cases:
            logger.warning("No test cases available")
            return {}
        
        # Use automatic upload if requested
        if use_automatic_upload:
            return self._evaluate_with_automatic_upload(query_pipeline, test_cases)
        
        results = {
            "total": len(test_cases),
            "latency": [],
            # Generator metrics
            "hallucination_scores": [],
            "answer_relevancy_scores": [],
            "faithfulness_scores": [],
            "per_test_results": []
        }
        
        # Use parallel execution if max_workers > 1
        if max_workers > 1:
            logger.info(f"Running evaluation with {max_workers} parallel workers")
            return self._evaluate_parallel(query_pipeline, test_cases, max_workers, results)
        else:
            # Sequential execution (original implementation)
            return self._evaluate_sequential(query_pipeline, test_cases, results)
    
    def _evaluate_sequential(self, query_pipeline, test_cases: List[Dict[str, Any]], 
                            results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate test cases sequentially."""
        for i, test_case in enumerate(test_cases):
            logger.info(f"Evaluating test case {i+1}/{len(test_cases)}")
            
            try:
                import time
                from pathlib import Path
                start_time = time.time()
                
                # Enhance query for audio files to help with retrieval
                query = test_case["query"]
                if test_case.get("modality") == "audio" and test_case.get("audio_path"):
                    # Extract audio filename and include in query for better retrieval
                    audio_filename = Path(test_case["audio_path"]).name
                    query = f"{query} [Audio file: {audio_filename}]"
                
                # Run query through pipeline
                response = query_pipeline.process(
                    query=query,
                    query_type=test_case.get("query_type"),
                    limit=10
                )
                
                latency = time.time() - start_time
                results["latency"].append(latency)
                
                # Extract results
                answer = response.get("answer", "")
                retrieved = response.get("retrieved_documents", [])
                expected = test_case.get("expected_answer", "")
                original_query = test_case.get("query", "")
                dataset = test_case.get("dataset")
                modality = test_case.get("modality")
                
                # Truncate long answers to prevent timeout (max 500 chars)
                max_answer_length = 500
                if len(answer) > max_answer_length:
                    answer = answer[:max_answer_length] + "... [truncated]"
                
                # Extract retrieval context from retrieved documents (reduced from 1000 to 500 chars)
                # This is the actual retrieved document chunks - required for RAG evaluation
                retrieval_context = [doc.get("content", "")[:500] for doc in retrieved[:3]]
                
                # DeepEval RAG metrics (following DeepEval best practices)
                # Uses retrieval_context (not context) as per DeepEval RAG evaluation guidelines
                deepeval_results = evaluate_with_deepeval(
                    input_text=original_query,
                    actual_output=answer,
                    expected_output=expected if expected else None,
                    retrieval_context=retrieval_context if retrieval_context else None,
                    ground_truths=None  # TODO: Add ground truth relevant documents if available
                )
                
                # Collect generator metrics
                if deepeval_results.get("hallucination_score") is not None:
                    results["hallucination_scores"].append(deepeval_results["hallucination_score"])
                if deepeval_results.get("answer_relevancy_score") is not None:
                    results["answer_relevancy_scores"].append(deepeval_results["answer_relevancy_score"])
                if deepeval_results.get("faithfulness_score") is not None:
                    results["faithfulness_scores"].append(deepeval_results["faithfulness_score"])
                
                results["per_test_results"].append({
                    "query": original_query,
                    "expected_answer": expected,
                    "answer": answer,
                    "dataset": dataset,
                    "modality": modality,
                    "deepeval": deepeval_results,
                    "latency": latency,
                    "retrieved_documents": retrieved
                })
                
            except Exception as e:
                logger.error(f"Test case {i+1} failed: {e}")
                results["latency"].append(0.0)
        
        return self._aggregate_results(results)
    
    def _evaluate_parallel(self, query_pipeline, test_cases: List[Dict[str, Any]], 
                          max_workers: int, results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate test cases in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        # Thread-safe counter for progress tracking
        completed_count = threading.Lock()
        completed = [0]
        
        def evaluate_single_case(test_case_data):
            """Evaluate a single test case."""
            i, test_case = test_case_data
            
            try:
                import time
                from pathlib import Path
                start_time = time.time()
                
                # Enhance query for audio files to help with retrieval
                query = test_case["query"]
                if test_case.get("modality") == "audio" and test_case.get("audio_path"):
                    # Extract audio filename and include in query for better retrieval
                    audio_filename = Path(test_case["audio_path"]).name
                    query = f"{query} [Audio file: {audio_filename}]"
                
                # Run query through pipeline
                response = query_pipeline.process(
                    query=query,
                    query_type=test_case.get("query_type"),
                    limit=10
                )
                
                latency = time.time() - start_time
                
                # Extract results
                answer = response.get("answer", "")
                retrieved = response.get("retrieved_documents", [])
                expected = test_case.get("expected_answer", "")
                original_query = test_case.get("query", "")
                dataset = test_case.get("dataset")
                modality = test_case.get("modality")
                
                # Truncate long answers to prevent timeout (max 500 chars)
                max_answer_length = 500
                if len(answer) > max_answer_length:
                    answer = answer[:max_answer_length] + "... [truncated]"
                
                # Extract retrieval context from retrieved documents (reduced from 1000 to 500 chars)
                # This is the actual retrieved document chunks - required for RAG evaluation
                retrieval_context = [doc.get("content", "")[:500] for doc in retrieved[:3]]
                
                # DeepEval RAG metrics (following DeepEval best practices)
                from evals.metrics import evaluate_with_deepeval
                deepeval_results = evaluate_with_deepeval(
                    input_text=original_query,
                    actual_output=answer,
                    expected_output=expected if expected else None,
                    retrieval_context=retrieval_context if retrieval_context else None,
                    ground_truths=None  # TODO: Add ground truth relevant documents if available
                )
                
                # Update progress
                with completed_count:
                    completed[0] += 1
                    logger.info(f"Completed test case {completed[0]}/{len(test_cases)}")
                
                return {
                    "success": True,
                    "latency": latency,
                    # Generator metrics
                    "hallucination_score": deepeval_results.get("hallucination_score"),
                    "answer_relevancy_score": deepeval_results.get("answer_relevancy_score"),
                    "faithfulness_score": deepeval_results.get("faithfulness_score"),
                    "per_test_result": {
                        "query": original_query,
                        "expected_answer": expected,
                        "answer": answer,
                        "dataset": dataset,
                        "modality": modality,
                        "deepeval": deepeval_results,
                        "latency": latency,
                        "retrieved_documents": retrieved
                    }
                }
            except Exception as e:
                logger.error(f"Test case {i+1} failed: {e}")
                with completed_count:
                    completed[0] += 1
                return {
                    "success": False,
                    "latency": 0.0,
                    "error": str(e)
                }
        
        # Execute test cases in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_case = {
                executor.submit(evaluate_single_case, (i, test_case)): (i, test_case)
                for i, test_case in enumerate(test_cases, 1)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_case):
                case_result = future.result()
                
                if case_result["success"]:
                    results["latency"].append(case_result["latency"])
                    
                    # Collect generator metrics
                    if case_result["hallucination_score"] is not None:
                        results["hallucination_scores"].append(case_result["hallucination_score"])
                    if case_result["answer_relevancy_score"] is not None:
                        results["answer_relevancy_scores"].append(case_result["answer_relevancy_score"])
                    if case_result["faithfulness_score"] is not None:
                        results["faithfulness_scores"].append(case_result["faithfulness_score"])
                    
                    results["per_test_results"].append(case_result["per_test_result"])
                else:
                    results["latency"].append(case_result["latency"])
        
        return self._aggregate_results(results)
    
    def _aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate evaluation results."""
        # Aggregate results
        aggregated = {
            "total_tests": results["total"],
            "latency": measure_latency(results["latency"]),
            "per_test_results": results["per_test_results"]
        }
        
        # Add Generator metrics (answer quality)
        if results["hallucination_scores"]:
            aggregated["avg_hallucination_score"] = sum(results["hallucination_scores"]) / len(results["hallucination_scores"])
            aggregated["hallucination_rate"] = 1.0 - aggregated["avg_hallucination_score"]
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
    
    def _evaluate_with_automatic_upload(self, query_pipeline, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate using DeepEval's evaluate() function for automatic Confident AI uploads.
        
        This method:
        1. Runs queries through the pipeline
        2. Creates LLMTestCase objects
        3. Uses DeepEval's evaluate() function which automatically uploads to Confident AI
        """
        try:
            from deepeval.test_case import LLMTestCase
            from deepeval.metrics import (
                HallucinationMetric,
                AnswerRelevancyMetric,
                FaithfulnessMetric
            )
            from deepeval import evaluate
            from evals.metrics import evaluate_with_deepeval_automatic_upload
        except ImportError as e:
            logger.error(f"DeepEval not available for automatic upload: {e}")
            logger.info("Falling back to manual evaluation...")
            return self._evaluate_sequential(query_pipeline, test_cases, {
                "total": len(test_cases),
                "latency": [],
                "hallucination_scores": [],
                "answer_relevancy_scores": [],
                "faithfulness_scores": [],
                "contextual_relevancy_scores": [],
                "contextual_precision_scores": [],
                "contextual_recall_scores": [],
                "per_test_results": []
            })
        
        logger.info("Using DeepEval's evaluate() function for automatic Confident AI uploads")
        logger.info("This will automatically upload results when CONFIDENT_API_KEY is set")
        
        # Prepare test cases for DeepEval
        deepeval_test_cases = []
        per_test_results = []
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Preparing test case {i+1}/{len(test_cases)} for DeepEval")
            
            try:
                import time
                from pathlib import Path
                start_time = time.time()
                
                # Enhance query for audio files
                query = test_case["query"]
                if test_case.get("modality") == "audio" and test_case.get("audio_path"):
                    audio_filename = Path(test_case["audio_path"]).name
                    query = f"{query} [Audio file: {audio_filename}]"
                
                # Run query through pipeline
                response = query_pipeline.process(
                    query=query,
                    query_type=test_case.get("query_type"),
                    limit=10
                )
                
                latency = time.time() - start_time
                
                # Extract results
                answer = response.get("answer", "")
                retrieved = response.get("retrieved_documents", [])
                expected = test_case.get("expected_answer", "")
                original_query = test_case.get("query", "")
                
                # Truncate long answers
                max_answer_length = 500
                if len(answer) > max_answer_length:
                    answer = answer[:max_answer_length] + "... [truncated]"
                
                # Extract retrieval context
                retrieval_context = [doc.get("content", "")[:500] for doc in retrieved[:3]]
                
                # Filter out empty contexts
                retrieval_context = [ctx for ctx in retrieval_context if ctx and ctx.strip()]
                
                # Skip test case if no retrieval context (required for RAG metrics)
                if not retrieval_context:
                    logger.warning(f"Skipping test case {i+1}: No retrieval context available")
                    continue
                
                # Create LLMTestCase for DeepEval
                # Note: HallucinationMetric and FaithfulnessMetric require 'context' parameter
                # We use retrieval_context for both 'context' and 'retrieval_context'
                deepeval_test_case = LLMTestCase(
                    input=original_query,
                    actual_output=answer,
                    expected_output=expected if expected else None,
                    context=retrieval_context,  # Required for HallucinationMetric and FaithfulnessMetric
                    retrieval_context=retrieval_context  # Required for RAG triad metrics
                )
                deepeval_test_cases.append(deepeval_test_case)
                
                # Store per-test result for aggregation
                per_test_results.append({
                    "query": original_query,
                    "expected_answer": expected,
                    "answer": answer,
                    "dataset": test_case.get("dataset"),
                    "modality": test_case.get("modality"),
                    "latency": latency,
                    "retrieved_documents": retrieved
                })
                
            except Exception as e:
                logger.error(f"Failed to prepare test case {i+1}: {e}")
                continue
        
        if not deepeval_test_cases:
            logger.error("No test cases prepared for DeepEval")
            return {}
        
        # Define metrics
        metrics = [
            HallucinationMetric(threshold=0.5),
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.7)
        ]
        
        # Run evaluation with automatic upload
        upload_result = evaluate_with_deepeval_automatic_upload(deepeval_test_cases, metrics)
        
        # Aggregate results (similar to manual evaluation)
        # Note: DeepEval's evaluate() doesn't return detailed per-metric scores in the same format
        # So we'll use the per_test_results we collected
        results = {
            "total": len(per_test_results),
            "latency": [r["latency"] for r in per_test_results],
            "hallucination_scores": [],
            "answer_relevancy_scores": [],
            "faithfulness_scores": [],
            "per_test_results": per_test_results,
            "confident_ai_uploaded": upload_result.get("confident_ai_uploaded", False)
        }
        
        return self._aggregate_results(results)

