"""Agentic query pipeline orchestrating validation, triage, retrieval, and answer generation."""

from typing import Dict, Any, Optional, List
import time
from loguru import logger

from agents.query_validation_agent import QueryValidationAgent
from agents.query_triage_agent import QueryTriageAgent
from agents.retrieval_orchestration_agent import RetrievalOrchestrationAgent
from agents.answer_generation_agent import AnswerGenerationAgent
from agents.post_processing_agent import PostProcessingAgent
from utils.errors import ValidationError, APIError
from utils.telemetry import get_telemetry_collector
from utils.langsmith_config import trace_agent_operation

# CrewAI integration (optional - can be enabled via feature flag)
try:
    from pipeline.crewai_orchestrator import CrewAIOrchestrator
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logger.warning("CrewAI not available. Using custom orchestration.")


class AgenticQueryPipeline:
    """
    Agentic query pipeline orchestrating validation, triage, retrieval, answer generation,
    and post-processing stages.
    """
    
    def __init__(self, retrieval_agent: RetrievalOrchestrationAgent, use_crewai: bool = False):
        """
        Initialize agentic query pipeline.
        
        Args:
            retrieval_agent: Retrieval orchestration agent with search tools
            use_crewai: Whether to use CrewAI orchestration (default: False for backward compatibility)
        """
        self.use_crewai = use_crewai and CREWAI_AVAILABLE
        
        if self.use_crewai:
            logger.info("Using CrewAI orchestration")
            self.crewai_orchestrator = CrewAIOrchestrator(retrieval_agent)
        else:
            logger.info("Using custom orchestration")
            self.validation_agent = QueryValidationAgent()
            self.triage_agent = QueryTriageAgent()
            self.retrieval_agent = retrieval_agent
            self.answer_agent = AnswerGenerationAgent()
            self.postprocess_agent = PostProcessingAgent()
        
        self.telemetry = get_telemetry_collector()
    
    @trace_agent_operation("agentic_query_pipeline")
    def process(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        Process query through agentic pipeline.
        
        Args:
            query: User query
            max_iterations: Maximum retrieval attempts
        
        Returns:
            Complete response dictionary:
            {
                "query": str,
                "answer": str,
                "sources": List[Dict],
                "confidence": float,
                "metadata": Dict,
                "iterations": int
            }
        """
        # Use CrewAI if enabled
        if self.use_crewai:
            return self.crewai_orchestrator.execute_pipeline(query, max_iterations)
        
        # Fallback to custom orchestration
        return self._process_custom(query, max_iterations)
    
    def _process_custom(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        Process query using custom orchestration (original implementation).
        
        Args:
            query: User query
            max_iterations: Maximum retrieval attempts
        
        Returns:
            Complete response dictionary
        """
        # Track full pipeline execution
        pipeline_op_id = self.telemetry.start_operation(
            agent_name="AgenticQueryPipeline",
            operation="process",
            input_data={"query": query[:200], "max_iterations": max_iterations},
            metadata={"query_length": len(query)}
        )
        
        start_time = time.time()
        iteration = 0
        current_query = query
        
        try:
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"Processing iteration {iteration}/{max_iterations}")
                
                try:
                    # Stage 1: Validation
                    logger.info("Stage 1/4: Query Validation")
                    validation = self.validation_agent.validate(current_query)
                    
                    if not validation.get("is_valid", False):
                        raise ValidationError(validation.get("error", "Query validation failed"))
                    
                    sanitized_query = validation["sanitized_query"]
                    
                    # Stage 2: Triage
                    logger.info("Stage 2/4: Query Triage")
                    triage = self.triage_agent.triage(sanitized_query)
                    
                    # Stage 3: Retrieval
                    logger.info("Stage 3/4: Retrieval Orchestration")
                    retrieval_start = time.time()
                    retrieval_result = self.retrieval_agent.retrieve(
                        query=triage["expanded_query"],
                        strategy=triage["search_strategy"],
                        limit=10
                    )
                    retrieval_time = time.time() - retrieval_start
                    
                    results = retrieval_result.get("results", [])
                    retrieval_confidence = retrieval_result.get("confidence", 0.0)
                    
                    # Stage 4: Answer Generation
                    logger.info("Stage 4/4: Answer Generation")
                    generation_start = time.time()
                    answer_result = self.answer_agent.generate(
                        query=current_query,
                        context=results,
                        reasoning_steps=retrieval_result.get("reasoning_steps")
                    )
                    generation_time = time.time() - generation_start
                    
                    # Post-Processing
                    logger.info("Post-Processing results")
                    postprocess_start = time.time()
                    final_result = self.postprocess_agent.process(
                        answer=answer_result["answer"],
                        sources=results,
                        query=current_query
                    )
                    postprocess_time = time.time() - postprocess_start
                    
                    # Check if answer quality is acceptable
                    final_confidence = final_result["confidence"]
                    
                    if final_confidence > 0.7 or iteration >= max_iterations:
                        # Good enough or max iterations reached
                        total_time = time.time() - start_time
                        
                        result = {
                            "query": query,
                            "answer": final_result["final_answer"],
                            "sources": answer_result["citations"],
                            "retrieved_documents": results[:5],
                            "confidence": final_confidence,
                            "metadata": {
                                "iterations": iteration,
                                "search_strategy": triage["search_strategy"],
                                "methods_used": retrieval_result.get("methods_used", []),
                                "query_type": triage["query_type"],
                                "complexity": validation.get("complexity", "unknown"),
                                "intent": validation.get("intent", "unknown"),
                                "retrieval_time": retrieval_time,
                                "generation_time": generation_time,
                                "postprocess_time": postprocess_time,
                                "total_time": total_time,
                                "num_retrieved": len(results),
                                "hallucination_score": final_result.get("hallucination_score", 0.0),
                                "citation_verification": final_result.get("citation_verification", {})
                            },
                            "reasoning_steps": answer_result.get("reasoning_steps", [])
                        }
                        
                        # Surface optional LLM-as-judge details alongside other metadata
                        llm_judge = final_result.get("llm_judge")
                        if llm_judge is not None:
                            result["metadata"]["llm_judge"] = llm_judge
                        
                        # Complete telemetry tracking
                        self.telemetry.end_operation(
                            pipeline_op_id,
                            output_data={"confidence": final_confidence, "iterations": iteration},
                            metadata=result["metadata"]
                        )
                        
                        return result
                    else:
                        # Low confidence, try alternative strategy
                        logger.info(f"Low confidence ({final_confidence:.2f}), trying alternative strategy")
                        if iteration < max_iterations:
                            # Try different search strategy
                            triage["search_strategy"]["use_hybrid"] = True
                            continue
                
                except ValidationError as e:
                    logger.error(f"Validation error: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Pipeline error in iteration {iteration}: {e}")
                    if iteration >= max_iterations:
                        raise APIError(f"Query processing failed after {max_iterations} attempts: {e}")
                    # Try again with next iteration
                    continue
        
            # Max iterations reached or escalation
            total_time = time.time() - start_time
            
            result = {
                "query": query,
                "answer": "I couldn't find a confident answer after multiple attempts. Please try rephrasing your question or providing more context.",
                "sources": [],
                "confidence": 0.0,
                "metadata": {
                    "iterations": iteration,
                    "total_time": total_time,
                    "note": "Maximum iterations reached or escalation triggered"
                }
            }
            
            self.telemetry.end_operation(
                pipeline_op_id,
                output_data={"confidence": 0.0, "iterations": iteration},
                error="Maximum iterations reached",
                metadata=result["metadata"]
            )
            
            return result
        except Exception as e:
            self.telemetry.end_operation(pipeline_op_id, error=str(e))
            raise
    
