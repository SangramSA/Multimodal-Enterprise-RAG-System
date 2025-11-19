"""Query triage agent for classification and strategy selection."""

from typing import Dict, Any, Optional
import json
import openai
from loguru import logger

from agents.base_agent import BaseAgent
from utils.config import OPENAI_API_KEY, OPENAI_MODEL
from utils.errors import APIError


class QueryTriageAgent(BaseAgent):
    """Agent for query classification and search strategy selection."""
    
    QUERY_TYPES = {
        "factual_lookup": "Factual lookup - finding specific information",
        "visual_qa": "Visual question answering - questions about images",
        "audio_qa": "Audio question answering - questions about audio content",
        "reasoning": "Reasoning - requires inference or analysis",
        "summarization": "Summarization - summarizing content",
        "semantic_linkage": "Semantic linkage - finding relationships"
    }
    
    def __init__(self):
        super().__init__("QueryTriageAgent")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        Main triage processing method.
        
        Args:
            query: User query
        
        Returns:
            Triage result dictionary
        """
        return self.triage(query)
    
    def triage(self, query: str) -> Dict[str, Any]:
        """
        Perform query triage: classify and select search strategy.
        
        Args:
            query: User query string
        
        Returns:
            Dictionary with triage results:
            {
                "query_type": str,
                "expanded_query": str,
                "search_strategy": Dict[str, bool],
                "confidence": float,
                "reasoning": str
            }
        """
        try:
            # Classify query
            classification = self.classify_query(query)
            query_type = classification["query_type"]
            
            # Expand query
            expanded_query = self.expand_query(query)
            
            # Select search strategy
            search_strategy = self.select_search_strategy(query_type)
            
            return {
                "query_type": query_type,
                "expanded_query": expanded_query,
                "search_strategy": search_strategy,
                "confidence": classification.get("confidence", 0.7),
                "reasoning": classification.get("reasoning", "")
            }
        except Exception as e:
            self.log_error(f"Triage failed: {e}")
            # Fallback to default strategy
            return {
                "query_type": "factual_lookup",
                "expanded_query": query,
                "search_strategy": {
                    "use_keyword": True,
                    "use_vector": True,
                    "use_graph": False,
                    "use_hybrid": True
                },
                "confidence": 0.5,
                "reasoning": "Triage failed, using default strategy"
            }
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify query into one of the predefined types.
        
        Args:
            query: Query string
        
        Returns:
            Classification result:
            {
                "query_type": str,
                "confidence": float,
                "reasoning": str
            }
        """
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
            
            result = json.loads(response.choices[0].message.content)
            return {
                "query_type": result.get("query_type", "factual_lookup"),
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", "")
            }
        except Exception as e:
            self.log_warning(f"Query classification failed: {e}, defaulting to factual_lookup")
            return {
                "query_type": "factual_lookup",
                "confidence": 0.5,
                "reasoning": "Classification failed, using default"
            }
    
    def expand_query(self, query: str) -> str:
        """
        Expand query with synonyms and related terms.
        
        Args:
            query: Original query
        
        Returns:
            Expanded query string
        """
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
            self.log_warning(f"Query expansion failed: {e}, using original query")
            return query
    
    def select_search_strategy(self, query_type: str) -> Dict[str, bool]:
        """
        Select search strategy based on query type.
        
        Args:
            query_type: Classified query type
        
        Returns:
            Dictionary with strategy flags:
            {
                "use_keyword": bool,
                "use_vector": bool,
                "use_graph": bool,
                "use_hybrid": bool
            }
        """
        # Default strategy
        strategy = {
            "use_keyword": False,
            "use_vector": True,  # Always use vector search
            "use_graph": False,
            "use_hybrid": False
        }
        
        # Adjust based on query type
        if query_type == "factual_lookup":
            strategy["use_keyword"] = True
            strategy["use_vector"] = True
            strategy["use_hybrid"] = True
        
        elif query_type == "semantic_linkage" or query_type == "reasoning":
            strategy["use_graph"] = True
            strategy["use_vector"] = True
            strategy["use_hybrid"] = True
        
        elif query_type == "visual_qa" or query_type == "audio_qa":
            strategy["use_vector"] = True
            strategy["use_keyword"] = True  # For metadata matching
            strategy["use_hybrid"] = True
        
        elif query_type == "summarization":
            strategy["use_vector"] = True
            strategy["use_hybrid"] = True
        
        else:
            # Default: use hybrid for unknown types
            strategy["use_hybrid"] = True
            strategy["use_vector"] = True
        
        return strategy

