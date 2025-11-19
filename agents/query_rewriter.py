"""Query triage and rewriting."""

from typing import Dict, Any, Optional
import openai
from loguru import logger

from utils.config import OPENAI_API_KEY, OPENAI_MODEL
from utils.errors import APIError


class QueryRewriter:
    """Rewrite and classify queries."""
    
    QUERY_TYPES = {
        "factual_lookup": "Factual lookup - finding specific information",
        "visual_qa": "Visual question answering - questions about images",
        "audio_qa": "Audio question answering - questions about audio content",
        "reasoning": "Reasoning - requires inference or analysis",
        "summarization": "Summarization - summarizing content",
        "semantic_linkage": "Semantic linkage - finding relationships"
    }
    
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """Classify query type."""
        prompt = f"""Classify the following query into one of these types:
- factual_lookup: Finding specific facts or information
- visual_qa: Questions about images or visual content
- audio_qa: Questions about audio content
- reasoning: Requires inference, analysis, or reasoning
- summarization: Asking for summaries
- semantic_linkage: Finding relationships or connections

Query: {query}

Return JSON with:
- query_type: one of the types above
- confidence: 0-1
- reasoning: brief explanation"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at classifying queries. Return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return {
                "query_type": result.get("query_type", "factual_lookup"),
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", "")
            }
        except Exception as e:
            logger.warning(f"Query classification failed: {e}, defaulting to factual_lookup")
            return {
                "query_type": "factual_lookup",
                "confidence": 0.5,
                "reasoning": "Classification failed, using default"
            }
    
    def expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms."""
        prompt = f"""Expand the following search query with synonyms and related terms to improve retrieval.
Return only the expanded query, not explanations.

Original query: {query}
Expanded query:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at expanding search queries. Return only the expanded query."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            expanded = response.choices[0].message.content.strip()
            return expanded if expanded else query
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}, using original query")
            return query
    
    def rewrite_query(self, query: str, query_type: Optional[str] = None) -> Dict[str, Any]:
        """Rewrite query for better retrieval."""
        if not query_type:
            classification = self.classify_query(query)
            query_type = classification["query_type"]
        
        # Expand query
        expanded_query = self.expand_query(query)
        
        return {
            "original_query": query,
            "expanded_query": expanded_query,
            "query_type": query_type,
            "use_keyword": query_type in ["factual_lookup"],
            "use_vector": True,  # Always use vector search
            "use_graph": query_type in ["semantic_linkage", "reasoning"]
        }

