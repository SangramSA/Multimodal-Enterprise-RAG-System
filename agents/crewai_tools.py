"""CrewAI tools wrapping existing agent methods."""

from typing import Dict, Any, Optional, List
from loguru import logger

# Use LangChain's tool decorator which is compatible with CrewAI
try:
    from langchain_core.tools import tool
    TOOL_DECORATOR_AVAILABLE = True
except ImportError:
    TOOL_DECORATOR_AVAILABLE = False
    # Fallback decorator (won't work but prevents import errors)
    def tool(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func

from agents.query_validation_agent import QueryValidationAgent
from agents.query_triage_agent import QueryTriageAgent
from agents.retrieval_orchestration_agent import RetrievalOrchestrationAgent
from agents.answer_generation_agent import AnswerGenerationAgent
from agents.post_processing_agent import PostProcessingAgent
from utils.telemetry import get_telemetry_collector


# Global agent instances (will be initialized by factory)
_validation_agent: Optional[QueryValidationAgent] = None
_triage_agent: Optional[QueryTriageAgent] = None
_retrieval_agent: Optional[RetrievalOrchestrationAgent] = None
_answer_agent: Optional[AnswerGenerationAgent] = None
_postprocess_agent: Optional[PostProcessingAgent] = None


def initialize_tools(validation_agent: QueryValidationAgent,
                     triage_agent: QueryTriageAgent,
                     retrieval_agent: RetrievalOrchestrationAgent,
                     answer_agent: AnswerGenerationAgent,
                     postprocess_agent: PostProcessingAgent):
    """Initialize global agent instances for tools."""
    global _validation_agent, _triage_agent, _retrieval_agent
    global _answer_agent, _postprocess_agent
    
    _validation_agent = validation_agent
    _triage_agent = triage_agent
    _retrieval_agent = retrieval_agent
    _answer_agent = answer_agent
    _postprocess_agent = postprocess_agent


@tool
def validate_query_tool(query: str) -> Dict[str, Any]:
    """
    Validate a user query for security, complexity, and intent.
    
    Args:
        query: User query string to validate
    
    Returns:
        Dictionary with validation results including:
        - is_valid: bool
        - sanitized_query: str
        - complexity: str (simple/moderate/complex)
        - intent: str (factual/analytical/exploratory/comparative)
        - requires_clarification: bool
        - clarification_questions: List[str]
        - security_checks: Dict[str, bool]
    """
    if not _validation_agent:
        raise ValueError("Validation agent not initialized. Call initialize_tools() first.")
    
    telemetry = get_telemetry_collector()
    op_id = telemetry.start_operation("QueryValidationAgent", "validate", {"query_length": len(query)})
    
    try:
        result = _validation_agent.validate(query)
        telemetry.end_operation(op_id, output_data={"is_valid": result.get("is_valid")})
        return result
    except Exception as e:
        telemetry.end_operation(op_id, error=str(e))
        raise


@tool
def triage_query_tool(query: str) -> Dict[str, Any]:
    """
    Classify a query and select the optimal search strategy.
    
    Args:
        query: User query string
    
    Returns:
        Dictionary with triage results including:
        - query_type: str (factual_lookup/visual_qa/audio_qa/reasoning/summarization/semantic_linkage)
        - expanded_query: str
        - search_strategy: Dict[str, bool] (use_keyword, use_vector, use_graph, use_hybrid)
        - confidence: float
        - reasoning: str
    """
    if not _triage_agent:
        raise ValueError("Triage agent not initialized. Call initialize_tools() first.")
    
    telemetry = get_telemetry_collector()
    op_id = telemetry.start_operation("QueryTriageAgent", "triage", {"query_length": len(query)})
    
    try:
        result = _triage_agent.triage(query)
        telemetry.end_operation(op_id, output_data={"query_type": result.get("query_type")})
        return result
    except Exception as e:
        telemetry.end_operation(op_id, error=str(e))
        raise


@tool
def retrieve_documents_tool(query: str, strategy: Optional[Dict[str, Any]] = None, 
                           limit: int = 10) -> Dict[str, Any]:
    """
    Retrieve relevant documents using keyword, vector, graph, or hybrid search.
    
    Args:
        query: Search query
        strategy: Optional search strategy dictionary (if None, agent decides)
        limit: Maximum number of results
    
    Returns:
        Dictionary with retrieval results including:
        - results: List[Dict] - Retrieved documents
        - methods_used: List[str] - Search methods used
        - reasoning: str - Explanation of method selection
        - confidence: float - Confidence score
    """
    """
    Retrieve relevant documents using keyword, vector, graph, or hybrid search.
    
    Args:
        query: Search query
        strategy: Optional search strategy dictionary (if None, agent decides)
        limit: Maximum number of results
    
    Returns:
        Dictionary with retrieval results including:
        - results: List[Dict] - Retrieved documents
        - methods_used: List[str] - Search methods used
        - reasoning: str - Explanation of method selection
        - confidence: float - Confidence score
    """
    if not _retrieval_agent:
        raise ValueError("Retrieval agent not initialized. Call initialize_tools() first.")
    
    telemetry = get_telemetry_collector()
    op_id = telemetry.start_operation(
        "RetrievalOrchestrationAgent", 
        "retrieve", 
        {"query_length": len(query), "limit": limit}
    )
    
    try:
        result = _retrieval_agent.retrieve(query, strategy, limit)
        telemetry.end_operation(
            op_id, 
            output_data={
                "num_results": len(result.get("results", [])),
                "methods_used": result.get("methods_used", [])
            }
        )
        return result
    except Exception as e:
        telemetry.end_operation(op_id, error=str(e))
        raise


@tool
def generate_answer_tool(query: str, context: List[Dict], 
                        reasoning_steps: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive answer from retrieved context.
    
    Args:
        query: User query
        context: List of retrieved documents
        reasoning_steps: Optional reasoning steps from previous processing
    
    Returns:
        Dictionary with answer results:
        - answer: str - Generated answer
        - reasoning_steps: List[str] - Reasoning steps (if multi-step)
        - citations: List[Dict] - Source citations
        - confidence: float - Confidence score
    """
    if not _answer_agent:
        raise ValueError("Answer agent not initialized. Call initialize_tools() first.")
    
    telemetry = get_telemetry_collector()
    op_id = telemetry.start_operation(
        "AnswerGenerationAgent",
        "generate",
        {"query_length": len(query), "num_context_docs": len(context)}
    )
    
    try:
        result = _answer_agent.generate(query, context, reasoning_steps)
        telemetry.end_operation(
            op_id,
            output_data={
                "answer_length": len(result.get("answer", "")),
                "num_citations": len(result.get("citations", []))
            }
        )
        return result
    except Exception as e:
        telemetry.end_operation(op_id, error=str(e))
        raise


@tool
def validate_answer_tool(answer: str, sources: List[Dict], 
                        query: str) -> Dict[str, Any]:
    """
    Validate answer quality, detect hallucinations, and verify citations.
    
    Args:
        answer: Generated answer
        sources: Source documents
        query: Original query
    
    Returns:
        Dictionary with validation results:
        - final_answer: str - Validated answer
        - confidence: float - Overall confidence
        - hallucination_score: float - Hallucination detection score
        - citation_verification: Dict - Citation verification results
        - validation: Dict - Validation details
    """
    if not _postprocess_agent:
        raise ValueError("Post-processing agent not initialized. Call initialize_tools() first.")
    
    telemetry = get_telemetry_collector()
    op_id = telemetry.start_operation(
        "PostProcessingAgent",
        "process",
        {"answer_length": len(answer), "num_sources": len(sources)}
    )
    
    try:
        result = _postprocess_agent.process(answer, sources, query)
        telemetry.end_operation(
            op_id,
            output_data={
                "confidence": result.get("confidence", 0.0),
                "hallucination_score": result.get("hallucination_score", 0.0)
            }
        )
        return result
    except Exception as e:
        telemetry.end_operation(op_id, error=str(e))
        raise

