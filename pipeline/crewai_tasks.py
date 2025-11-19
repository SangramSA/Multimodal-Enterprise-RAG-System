"""CrewAI tasks for the multi-agent pipeline."""

from typing import Dict, Any, Optional
from loguru import logger

try:
    from crewai import Task
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logger.warning("CrewAI not installed. Install with: pip install crewai")
    # Dummy Task class for type hints
    class Task:
        pass


def create_validation_task(agent, query: str) -> Task:
    """
    Create validation task.
    
    Args:
        agent: CrewAI validation agent
        query: User query to validate
    
    Returns:
        CrewAI Task for validation
    """
    return Task(
        description=f"""Validate and sanitize the following user query:
        
Query: "{query}"

Perform the following:
1. Validate the query for security threats (SQL injection, XSS, etc.)
2. Assess query complexity (simple/moderate/complex)
3. Detect user intent (factual/analytical/exploratory/comparative)
4. Sanitize the query

Return a JSON object with:
- is_valid: boolean
- sanitized_query: string
- complexity: string
- intent: string
- security_checks: object with security check results""",
        agent=agent,
        expected_output="JSON object with validation results including is_valid, sanitized_query, complexity, intent, and security_checks"
    )


def create_triage_task(agent, sanitized_query: str) -> Task:
    """
    Create triage task.
    
    Args:
        agent: CrewAI triage agent
        sanitized_query: Sanitized query from validation
    
    Returns:
        CrewAI Task for triage
    """
    return Task(
        description=f"""Classify the following query and select the optimal search strategy:

Query: "{sanitized_query}"

Perform the following:
1. Classify the query type (factual_lookup, visual_qa, audio_qa, reasoning, summarization, semantic_linkage)
2. Expand the query with synonyms and related terms
3. Select the optimal search strategy (which methods to use: keyword, vector, graph, hybrid)

Return a JSON object with:
- query_type: string
- expanded_query: string
- search_strategy: object with use_keyword, use_vector, use_graph, use_hybrid booleans
- confidence: float (0-1)
- reasoning: string explaining the classification and strategy selection""",
        agent=agent,
        expected_output="JSON object with query_type, expanded_query, search_strategy, confidence, and reasoning"
    )


def create_retrieval_task(agent, expanded_query: str, strategy: Dict[str, Any]) -> Task:
    """
    Create retrieval task.
    
    Args:
        agent: CrewAI retrieval agent
        expanded_query: Expanded query from triage
        strategy: Search strategy dictionary
    
    Returns:
        CrewAI Task for retrieval
    """
    return Task(
        description=f"""Retrieve relevant documents for the following query:

Query: "{expanded_query}"
Search Strategy: {strategy}

Use the appropriate search methods (keyword, vector, graph, or hybrid) based on the strategy.
Retrieve up to 10 relevant documents.

Return a JSON object with:
- results: array of document objects (each with content, chunk_id, metadata, score)
- methods_used: array of strings indicating which search methods were used
- reasoning: string explaining the retrieval approach
- confidence: float (0-1) indicating confidence in the results""",
        agent=agent,
        expected_output="JSON object with results array, methods_used array, reasoning string, and confidence float"
    )


def create_generation_task(agent, query: str, context: list) -> Task:
    """
    Create answer generation task.
    
    Args:
        agent: CrewAI answer generation agent
        query: User query
        context: Retrieved documents
    
    Returns:
        CrewAI Task for answer generation
    """
    return Task(
        description=f"""Generate a comprehensive answer to the following query using the provided context:

Query: "{query}"

Context Documents ({len(context)} documents):
{_format_context_for_task(context)}

Instructions:
1. Use only information from the provided context
2. If the context doesn't contain enough information, say so
3. For complex queries, use multi-step reasoning
4. Always cite sources using [Source X] format
5. Be comprehensive but concise

Return a JSON object with:
- answer: string - The generated answer
- reasoning_steps: array of strings - Reasoning steps (if multi-step)
- citations: array of objects with source_id, chunk_id, file_name, modality, content_preview, score
- confidence: float (0-1) - Confidence in the answer""",
        agent=agent,
        expected_output="JSON object with answer string, reasoning_steps array, citations array, and confidence float"
    )


def create_qa_task(agent, answer: str, sources: list, query: str) -> Task:
    """
    Create quality assurance task.
    
    Args:
        agent: CrewAI QA agent
        answer: Generated answer
        sources: Source documents
        query: Original query
    
    Returns:
        CrewAI Task for quality assurance
    """
    return Task(
        description=f"""Validate the quality of the following answer:

Query: "{query}"
Answer: "{answer[:500]}..." (truncated)
Number of Sources: {len(sources)}

Perform the following validations:
1. Check if answer references sources correctly
2. Detect potential hallucinations
3. Verify citations are valid
4. Calculate overall confidence score
5. Validate answer completeness

Return a JSON object with:
- final_answer: string - The validated answer
- confidence: float (0-1) - Overall confidence
- hallucination_score: float (0-1) - Hallucination detection score (higher = more likely hallucination)
- citation_verification: object with total_citations, valid_citations, invalid_citations, missing_citations
- validation: object with is_valid boolean and issues array""",
        agent=agent,
        expected_output="JSON object with final_answer, confidence, hallucination_score, citation_verification, and validation"
    )


def _format_context_for_task(context: list, max_docs: int = 10) -> str:
    """Format context documents for task description."""
    formatted = []
    for i, doc in enumerate(context[:max_docs], 1):
        content_preview = doc.get("content", "")[:200]
        file_name = doc.get("metadata", {}).get("file_name", "Unknown")
        score = doc.get("rrf_score", doc.get("score", 0))
        formatted.append(
            f"[Source {i}] {file_name} (Score: {score:.3f})\n"
            f"Content: {content_preview}...\n"
        )
    return "\n".join(formatted)

