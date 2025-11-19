"""CrewAI orchestrator for multi-agent pipeline."""

from typing import Dict, Any, Optional, List
import json
import time
from loguru import logger

try:
    from crewai import Crew, Process
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logger.warning("CrewAI not installed. Install with: pip install crewai")

from agents.crewai_agents import CrewAIAgentFactory
from agents.crewai_tools import (
    initialize_tools, validate_query_tool, triage_query_tool,
    retrieve_documents_tool, generate_answer_tool, validate_answer_tool
)
from agents.retrieval_orchestration_agent import RetrievalOrchestrationAgent
from pipeline.crewai_tasks import (
    create_validation_task, create_triage_task, create_retrieval_task,
    create_generation_task, create_qa_task
)
from utils.telemetry import get_telemetry_collector
from utils.errors import ValidationError, APIError


class CrewAIOrchestrator:
    """Orchestrates multi-agent pipeline using CrewAI."""
    
    def __init__(self, retrieval_agent: RetrievalOrchestrationAgent):
        """
        Initialize CrewAI orchestrator.
        
        Args:
            retrieval_agent: Retrieval orchestration agent instance
        """
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI is not installed. Install with: pip install crewai")
        
        self.retrieval_agent = retrieval_agent
        self.factory = CrewAIAgentFactory(retrieval_agent)
        self.telemetry = get_telemetry_collector()
        
        # Initialize tools with agent instances
        initialize_tools(
            validation_agent=self.factory.validation_agent,
            triage_agent=self.factory.triage_agent,
            retrieval_agent=self.factory.retrieval_agent,
            answer_agent=self.factory.answer_agent,
            postprocess_agent=self.factory.postprocess_agent
        )
        
        # Create tools list for each agent
        self.validator_tools = [validate_query_tool]
        self.triage_tools = [triage_query_tool]
        self.retrieval_tools = [retrieve_documents_tool]
        self.generation_tools = [generate_answer_tool]
        self.qa_tools = [validate_answer_tool]
        
        # Create agents
        self.validator_agent = self.factory.create_validator_agent(self.validator_tools)
        self.triage_agent = self.factory.create_triage_agent(self.triage_tools)
        self.retrieval_agent_crew = self.factory.create_retrieval_agent(self.retrieval_tools)
        self.generation_agent = self.factory.create_answer_generator_agent(self.generation_tools)
        self.qa_agent = self.factory.create_qa_agent(self.qa_tools)
    
    def execute_pipeline(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        Execute the full pipeline using CrewAI.
        
        Args:
            query: User query
            max_iterations: Maximum iterations for refinement
        
        Returns:
            Complete response dictionary
        """
        # Track full pipeline execution
        pipeline_op_id = self.telemetry.start_operation(
            agent_name="CrewAIOrchestrator",
            operation="execute_pipeline",
            input_data={"query": query[:200], "max_iterations": max_iterations},
            metadata={"query_length": len(query)}
        )
        
        start_time = time.time()
        iteration = 0
        current_query = query
        
        try:
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"CrewAI Pipeline - Iteration {iteration}/{max_iterations}")
                
                try:
                    # Stage 1: Validation
                    logger.info("Stage 1/5: Query Validation")
                    validation_result = self._execute_validation(current_query)
                    
                    if not validation_result.get("is_valid", False):
                        raise ValidationError(
                            validation_result.get("error", "Query validation failed")
                        )
                    
                    sanitized_query = validation_result["sanitized_query"]
                    
                    # Stage 2: Triage
                    logger.info("Stage 2/5: Query Triage")
                    triage_result = self._execute_triage(sanitized_query)
                    
                    # Stage 3: Retrieval
                    logger.info("Stage 3/5: Retrieval")
                    retrieval_start = time.time()
                    retrieval_result = self._execute_retrieval(
                        triage_result["expanded_query"],
                        triage_result["search_strategy"]
                    )
                    retrieval_time = time.time() - retrieval_start
                    
                    results = retrieval_result.get("results", [])
                    retrieval_confidence = retrieval_result.get("confidence", 0.0)
                    
                    # Stage 4: Answer Generation
                    logger.info("Stage 4/5: Answer Generation")
                    generation_start = time.time()
                    answer_result = self._execute_generation(
                        current_query, results, retrieval_result.get("reasoning_steps")
                    )
                    generation_time = time.time() - generation_start
                    
                    # Stage 5: Quality Assurance
                    logger.info("Stage 5/5: Quality Assurance")
                    qa_start = time.time()
                    final_result = self._execute_qa(
                        answer_result["answer"], results, current_query
                    )
                    qa_time = time.time() - qa_start
                    
                    # Check if answer quality is acceptable
                    final_confidence = final_result["confidence"]
                    
                    if final_confidence > 0.7 or iteration >= max_iterations:
                        total_time = time.time() - start_time
                        
                        result = {
                            "query": query,
                            "answer": final_result["final_answer"],
                            "sources": answer_result["citations"],
                            "retrieved_documents": results[:5],
                            "confidence": final_confidence,
                            "metadata": {
                                "iterations": iteration,
                                "search_strategy": triage_result["search_strategy"],
                                "methods_used": retrieval_result.get("methods_used", []),
                                "query_type": triage_result["query_type"],
                                "complexity": validation_result.get("complexity", "unknown"),
                                "intent": validation_result.get("intent", "unknown"),
                                "retrieval_time": retrieval_time,
                                "generation_time": generation_time,
                                "qa_time": qa_time,
                                "total_time": total_time,
                                "num_retrieved": len(results),
                                "hallucination_score": final_result.get("hallucination_score", 0.0),
                                "citation_verification": final_result.get("citation_verification", {})
                            },
                            "reasoning_steps": answer_result.get("reasoning_steps", [])
                        }
                        
                        self.telemetry.end_operation(
                            pipeline_op_id,
                            output_data={"confidence": final_confidence, "iterations": iteration},
                            metadata=result["metadata"]
                        )
                        
                        return result
                    else:
                        logger.info(f"Low confidence ({final_confidence:.2f}), trying alternative strategy")
                        if iteration < max_iterations:
                            # Try different search strategy
                            triage_result["search_strategy"]["use_hybrid"] = True
                            continue
                
                except ValidationError as e:
                    logger.error(f"Validation error: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Pipeline error in iteration {iteration}: {e}")
                    if iteration >= max_iterations:
                        raise APIError(f"Query processing failed after {max_iterations} attempts: {e}")
                    continue
            
            # Max iterations reached
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
    
    def _execute_validation(self, query: str) -> Dict[str, Any]:
        """Execute validation stage."""
        # Use direct agent call for now (faster and more reliable)
        # CrewAI task execution can be added later if needed
        try:
            return self.factory.validation_agent.validate(query)
        except Exception as e:
            logger.warning(f"Validation via CrewAI failed, using direct call: {e}")
            return self.factory.validation_agent.validate(query)
    
    def _execute_triage(self, query: str) -> Dict[str, Any]:
        """Execute triage stage."""
        # Use direct agent call for now
        try:
            return self.factory.triage_agent.triage(query)
        except Exception as e:
            logger.warning(f"Triage via CrewAI failed, using direct call: {e}")
            return self.factory.triage_agent.triage(query)
    
    def _execute_retrieval(self, query: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute retrieval stage."""
        # Use direct retrieval for now (faster than CrewAI agent)
        return self.retrieval_agent.retrieve(query, strategy, limit=10)
    
    def _execute_generation(self, query: str, context: list, reasoning_steps: Optional[List[str]]) -> Dict[str, Any]:
        """Execute answer generation stage."""
        # Use direct agent call for now
        try:
            return self.factory.answer_agent.generate(query, context, reasoning_steps)
        except Exception as e:
            logger.warning(f"Answer generation via CrewAI failed, using direct call: {e}")
            return self.factory.answer_agent.generate(query, context, reasoning_steps)
    
    def _execute_qa(self, answer: str, sources: list, query: str) -> Dict[str, Any]:
        """Execute quality assurance stage."""
        # Use direct agent call for now
        try:
            return self.factory.postprocess_agent.process(answer, sources, query)
        except Exception as e:
            logger.warning(f"QA via CrewAI failed, using direct call: {e}")
            return self.factory.postprocess_agent.process(answer, sources, query)
    
    def _parse_agent_output(self, output: Any, default: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse CrewAI agent output.
        
        Args:
            output: CrewAI task output
            default: Default dictionary if parsing fails
        
        Returns:
            Parsed dictionary
        """
        if isinstance(output, dict):
            return output
        
        if isinstance(output, str):
            # Try to extract JSON from string
            try:
                # Look for JSON in the output
                import re
                json_match = re.search(r'\{.*\}', output, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except (json.JSONDecodeError, AttributeError):
                pass
            
            # If no JSON found, try to use tool output directly
            # For now, return default and log
            logger.warning(f"Could not parse agent output: {output[:200]}")
            return default
        
        return default
    
