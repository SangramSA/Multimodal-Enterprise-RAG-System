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
from middleware.pipeline_cache import PipelineCache


class QueryPipeline:
    """End-to-end query processing pipeline."""
    
    def __init__(self, retrieval_agent: RetrievalAgent):
        self.validator = InputValidator()
        self.retrieval_agent = retrieval_agent
        self.query_rewriter = QueryRewriter()
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.pipeline_cache = PipelineCache()
    
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
        validation_time = 0.0
        triage_time = 0.0
        cache_lookup_ms = 0.0
        
        logger.info(
            "QueryPipeline.process | query_len={} | type_override={} | limit={} | include_evaluation={}",
            len(query),
            query_type,
            limit,
            include_evaluation,
        )
        
        try:
            # 1. Input validation
            validation_start = time.time()
            validation_result = self.validator.validate_query(query)
            validation_time = time.time() - validation_start
            sanitized_query = validation_result["sanitized_query"]

            # 1b. Pipeline-level semantic cache lookup
            cache_lookup_start = time.time()
            cache_hit, matched_query, cache_age_ms, cached_response = self.pipeline_cache.lookup(
                sanitized_query
            )
            cache_lookup_ms = (time.time() - cache_lookup_start) * 1000.0
            if cache_hit and cached_response:
                # Shallow copy to avoid mutating cached object
                response = dict(cached_response)
                metadata = dict(response.get("metadata", {}))
                metadata["pipeline_cache_hit"] = True
                metadata["pipeline_cache_matched_query"] = matched_query
                metadata["pipeline_cache_age_ms"] = cache_age_ms
                metadata["pipeline_cache_lookup_ms"] = cache_lookup_ms
                # Preserve original total time as cold_total_time if present
                if "cold_total_time" not in metadata and "total_time" in metadata:
                    metadata["cold_total_time"] = metadata["total_time"]
                # Record the fast cached total time for this request
                metadata["cached_total_time"] = time.time() - start_time
                metadata["total_time"] = metadata["cached_total_time"]
                response["metadata"] = metadata
                logger.info(
                    "QueryPipeline.process | PIPELINE CACHE HIT | query_len={} | matched_query_prefix='{}' | age_ms={:.1f}",
                    len(query),
                    (matched_query or "")[:80],
                    cache_age_ms or 0.0,
                )
                return response
            
            # 2. Query triage/rewriting
            triage_start = time.time()
            rewritten = self.query_rewriter.rewrite_query(sanitized_query, query_type)
            triage_time = time.time() - triage_start
            
            # 3. Agent-based retrieval orchestration
            retrieval_start = time.time()
            retrieved_docs = self.retrieval_agent.retrieve(rewritten["expanded_query"], limit=limit)
            retrieval_time = time.time() - retrieval_start
            
            if not retrieved_docs:
                total_time = time.time() - start_time
                logger.info(
                    "QueryPipeline.process | NO RESULTS | query_type={} | validation_s={:.3f} | triage_s={:.3f} | retrieval_s={:.3f}",
                    rewritten.get("query_type"),
                    validation_time,
                    triage_time,
                    retrieval_time,
                )
                return {
                    "query": query,
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "confidence": 0.0,
                        "metadata": {
                            "query_type": rewritten["query_type"],
                            "retrieval_time": retrieval_time,
                            "generation_time": 0.0,
                            "validation_time": validation_time,
                            "triage_time": triage_time,
                            "pipeline_cache_lookup_ms": cache_lookup_ms,
                            "total_time": total_time,
                            "cold_total_time": total_time,
                            "num_retrieved": 0,
                            "pipeline_cache_hit": False,
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

            response = {
                "query": query,
                "answer": answer,
                "sources": citations,
                "retrieved_documents": retrieved_docs[:5],  # Top 5 for display
                "confidence": confidence,
                "metadata": {
                    "query_type": rewritten["query_type"],
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "validation_time": validation_time,
                    "triage_time": triage_time,
                    "pipeline_cache_lookup_ms": cache_lookup_ms,
                    "total_time": total_time,
                    "cold_total_time": total_time,
                    "num_retrieved": len(retrieved_docs),
                    "pipeline_cache_hit": False,
                },
                "evaluation": eval_metrics
            }
            # Store in pipeline cache for future semantic hits
            self.pipeline_cache.store(sanitized_query, response)

            logger.info(
                "QueryPipeline.process | DONE | query_type={} | num_retrieved={} | validation_s={:.3f} | triage_s={:.3f} | retrieval_s={:.3f} | generation_s={:.3f} | total_s={:.3f}",
                rewritten.get("query_type"),
                len(retrieved_docs),
                validation_time,
                triage_time,
                retrieval_time,
                generation_time,
                total_time,
            )

            return response
            
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

