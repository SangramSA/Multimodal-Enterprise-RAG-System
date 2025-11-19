"""End-to-end query processing pipeline."""

from typing import Dict, Any, Optional, List
import time
import openai
from loguru import logger

from utils.config import OPENAI_API_KEY, OPENAI_MODEL
from utils.errors import ValidationError, APIError
from pipeline.validation import InputValidator
from agents.retrieval_agent import RetrievalAgent
from agents.query_rewriter import QueryRewriter


class QueryPipeline:
    """End-to-end query processing pipeline."""
    
    def __init__(self, retrieval_agent: RetrievalAgent):
        self.validator = InputValidator()
        self.retrieval_agent = retrieval_agent
        self.query_rewriter = QueryRewriter()
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    
    def process(self, query: str, query_type: Optional[str] = None,
               limit: int = 10, include_evaluation: bool = False) -> Dict[str, Any]:
        """
        Process a query through the full pipeline.
        
        Args:
            query: User query
            query_type: Optional query type override
            limit: Maximum number of retrieval results
            include_evaluation: Whether to include evaluation metrics
        
        Returns:
            Complete response with answer, sources, and metadata
        """
        start_time = time.time()
        
        try:
            # 1. Input validation
            validation_result = self.validator.validate_query(query)
            sanitized_query = validation_result["sanitized_query"]
            
            # 2. Query triage/rewriting
            rewritten = self.query_rewriter.rewrite_query(sanitized_query, query_type)
            
            # 3. Agent-based retrieval orchestration
            retrieval_start = time.time()
            retrieved_docs = self.retrieval_agent.retrieve(rewritten["expanded_query"], limit=limit)
            retrieval_time = time.time() - retrieval_start
            
            if not retrieved_docs:
                return {
                    "query": query,
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "confidence": 0.0,
                    "metadata": {
                        "retrieval_time": retrieval_time,
                        "total_time": time.time() - start_time,
                        "query_type": rewritten["query_type"]
                    }
                }
            
            # 4. Context assembly
            context = self._assemble_context(retrieved_docs)
            
            # 5. Answer generation
            generation_start = time.time()
            answer_result = self._generate_answer(sanitized_query, context, retrieved_docs)
            generation_time = time.time() - generation_start
            
            # 6. Post-processing
            answer = answer_result["answer"]
            confidence = answer_result.get("confidence", 0.5)
            citations = self._extract_citations(answer, retrieved_docs)
            
            # 7. Evaluation logging (if enabled)
            eval_metrics = None
            if include_evaluation:
                eval_metrics = self._evaluate_response(query, answer, retrieved_docs)
            
            total_time = time.time() - start_time
            
            return {
                "query": query,
                "answer": answer,
                "sources": citations,
                "retrieved_documents": retrieved_docs[:5],  # Top 5 for display
                "confidence": confidence,
                "metadata": {
                    "query_type": rewritten["query_type"],
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "total_time": total_time,
                    "num_retrieved": len(retrieved_docs)
                },
                "evaluation": eval_metrics
            }
            
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise APIError(f"Query processing failed: {e}")
    
    def _assemble_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Assemble context from retrieved documents."""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs[:5], 1):  # Use top 5 for context
            content = doc.get("content", "")
            chunk_id = doc.get("chunk_id", f"doc_{i}")
            modality = doc.get("metadata", {}).get("modality", "unknown")
            
            context_parts.append(f"[Source {i} - {modality} - {chunk_id}]\n{content[:500]}\n")
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, query: str, context: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer using GPT-4."""
        if not self.client:
            raise APIError("OpenAI client not initialized")
        
        prompt = f"""You are a helpful assistant that answers questions based on provided context.
Use only the information from the context to answer. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers based on the given context. Always cite your sources."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Estimate confidence based on source scores
            avg_score = sum(doc.get("rrf_score", doc.get("score", 0)) for doc in sources[:3]) / min(len(sources), 3)
            confidence = min(avg_score * 1.2, 1.0)  # Scale to 0-1
            
            return {
                "answer": answer,
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise APIError(f"Failed to generate answer: {e}")
    
    def _extract_citations(self, answer: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract citations from answer and map to sources."""
        citations = []
        
        for i, source in enumerate(sources[:5], 1):
            citations.append({
                "source_id": i,
                "chunk_id": source.get("chunk_id"),
                "file_id": source.get("metadata", {}).get("file_id"),
                "modality": source.get("metadata", {}).get("modality"),
                "content_preview": source.get("content", "")[:200],
                "score": source.get("rrf_score", source.get("score", 0))
            })
        
        return citations
    
    def _evaluate_response(self, query: str, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate response quality (simplified)."""
        return {
            "num_sources": len(sources),
            "avg_source_score": sum(s.get("rrf_score", s.get("score", 0)) for s in sources) / len(sources) if sources else 0,
            "answer_length": len(answer),
            "has_citations": len(sources) > 0
        }

