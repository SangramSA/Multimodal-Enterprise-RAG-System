"""Retrieval orchestration agent with LangChain tools for direct data store access."""

from typing import List, Dict, Any, Optional
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from loguru import logger

from agents.base_agent import BaseAgent
from utils.config import OPENAI_API_KEY, OPENAI_MODEL
from utils.langsmith_config import trace_agent_operation
from search.hybrid_search import HybridSearch
from search.graph_search import GraphSearch
from search.keyword_search import KeywordSearch
from search.vector_search import VectorSearch
from agents.utils import format_search_results, format_graph_results


class RetrievalOrchestrationAgent(BaseAgent):
    """Agent for orchestrating retrieval using LangChain with direct data store access."""
    
    def __init__(self, hybrid_search: HybridSearch, graph_search: GraphSearch,
                 keyword_search: KeywordSearch, vector_search: VectorSearch):
        super().__init__("RetrievalOrchestrationAgent")
        
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        
        self.hybrid_search = hybrid_search
        self.graph_search = graph_search
        self.keyword_search = keyword_search
        self.vector_search = vector_search
        
        self.llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1, api_key=OPENAI_API_KEY)
        self.agent = self._create_agent()
    
    def process(self, query: str, strategy: Optional[Dict[str, bool]] = None, 
                limit: int = 10) -> Dict[str, Any]:
        """
        Main retrieval processing method.
        
        Args:
            query: User query
            strategy: Optional search strategy flags
            limit: Maximum number of results
        
        Returns:
            Retrieval result dictionary
        """
        return self.retrieve(query, strategy, limit)
    
    def _create_agent(self):
        """Create LangChain agent with retrieval tools."""
        
        # Graph search tool (consolidated)
        def graph_search_tool_func(query: str, search_type: str = "auto", 
                                  entity_names: Optional[str] = None,
                                  relationship_type: Optional[str] = None) -> str:
            """
            Perform comprehensive graph search in Neo4j.
            
            This tool handles all graph operations:
            - Entity search: Find entities in the knowledge graph
            - Relationship traversal: Find relationships between entities
            - Path finding: Find paths between two entities
            - Content search: Find documents/images/audio related to entities
            
            Use this tool when:
            - Query mentions specific entities, people, organizations, or concepts
            - You need to find relationships or connections
            - Query asks about "what is X related to" or "how are X and Y connected"
            
            Args:
                query: Natural language query or entity name(s)
                search_type: "auto" (default), "entity", "relationship", "path", or "content"
                entity_names: Optional comma-separated list of entity names
                relationship_type: Optional specific relationship type (e.g., "WORKS_FOR", "LOCATED_IN")
            
            Returns:
                Formatted string with search results
            """
            try:
                entity_list = None
                if entity_names:
                    entity_list = [e.strip() for e in entity_names.split(",")]
                
                results = self.graph_search.search_comprehensive(
                    query=query,
                    search_type=search_type,
                    entity_names=entity_list,
                    relationship_type=relationship_type,
                    limit=10
                )
                
                return format_graph_results(results, search_type)
            except Exception as e:
                logger.error(f"Graph search tool failed: {e}")
                return f"Graph search failed: {str(e)}"
        
        # Keyword search tool
        def keyword_search_tool_func(query: str, filters: Optional[str] = None) -> str:
            """
            Perform keyword-based search using BM25 ranking.
            
            Use this tool for:
            - Exact term matching
            - Specific phrases or keywords
            - When you know the exact terms to search for
            
            Args:
                query: Search query with keywords
                filters: Optional JSON string with filters (e.g., '{"modality": "text"}')
            
            Returns:
                Formatted string with search results
            """
            try:
                filter_dict = None
                if filters:
                    import json
                    filter_dict = json.loads(filters)
                
                results = self.keyword_search.search(query, limit=10, filters=filter_dict)
                return format_search_results(results, "keyword")
            except Exception as e:
                logger.error(f"Keyword search tool failed: {e}")
                return f"Keyword search failed: {str(e)}"
        
        # Semantic vector search tool
        def semantic_vector_search_tool_func(query: str, filters: Optional[str] = None) -> str:
            """
            Perform semantic similarity search using embeddings.
            
            Use this tool for:
            - Conceptual queries
            - When exact terms don't match but meaning is similar
            - Abstract or high-level questions
            
            Args:
                query: Natural language query
                filters: Optional JSON string with filters
            
            Returns:
                Formatted string with search results
            """
            try:
                filter_dict = None
                if filters:
                    import json
                    filter_dict = json.loads(filters)
                
                results = self.vector_search.search(query, limit=10, filters=filter_dict)
                return format_search_results(results, "vector")
            except Exception as e:
                logger.error(f"Vector search tool failed: {e}")
                return f"Vector search failed: {str(e)}"
        
        # Hybrid search tool
        def hybrid_search_tool_func(query: str, strategy: str = "auto") -> str:
            """
            Perform hybrid search combining all methods with RRF reranking.
            
            Use this tool for:
            - Complex queries that may benefit from multiple search methods
            - When unsure which method to use
            - Comprehensive search across all data sources
            
            Args:
                query: Natural language query
                strategy: "auto" (default), "keyword_heavy", "vector_heavy", or "graph_heavy"
            
            Returns:
                Formatted string with search results
            """
            try:
                # Map strategy to search flags
                use_keyword = strategy in ["auto", "keyword_heavy"]
                use_vector = strategy in ["auto", "vector_heavy"]
                use_graph = strategy in ["auto", "graph_heavy"]
                
                results = self.hybrid_search.search(
                    query=query,
                    limit=10,
                    use_keyword=use_keyword,
                    use_vector=use_vector,
                    use_graph=use_graph
                )
                return format_search_results(results, "hybrid")
            except Exception as e:
                logger.error(f"Hybrid search tool failed: {e}")
                return f"Hybrid search failed: {str(e)}"
        
        # Wrap functions as tools
        graph_search_tool = tool(graph_search_tool_func)
        keyword_search_tool = tool(keyword_search_tool_func)
        semantic_vector_search_tool = tool(semantic_vector_search_tool_func)
        hybrid_search_tool = tool(hybrid_search_tool_func)
        
        tools = [graph_search_tool, keyword_search_tool, semantic_vector_search_tool, hybrid_search_tool]
        
        # Create agent using LangChain 1.0 API
        system_prompt = """You are an intelligent retrieval agent with access to multiple search methods.

Your capabilities:
1. Graph Search: Find entities and their relationships in the knowledge graph (Neo4j)
2. Keyword Search: Find exact term matches using BM25 ranking (Qdrant)
3. Semantic Vector Search: Find conceptually similar content using embeddings (Qdrant)
4. Hybrid Search: Combine all methods for comprehensive results

Decision Framework:
- If query mentions specific entities/people/orgs → Use graph_search_tool first
- If query has exact terms/phrases → Use keyword_search_tool
- If query is conceptual/abstract → Use semantic_vector_search_tool
- If query is complex/multi-faceted → Use hybrid_search_tool
- If unsure → Start with hybrid_search_tool, then refine based on results

Always:
- Use multiple tools if needed to get comprehensive results
- Combine results intelligently
- Provide reasoning for your tool choices
- If results are insufficient, try alternative methods
- Return formatted results with scores and metadata"""
        
        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt
        )
        
        return agent
    
    @trace_agent_operation("retrieval_orchestration")
    def retrieve(self, query: str, strategy: Optional[Dict[str, bool]] = None, 
                limit: int = 10) -> Dict[str, Any]:
        """
        Retrieve relevant documents using the agent.
        
        Args:
            query: User query
            strategy: Optional search strategy flags (if None, agent decides)
            limit: Maximum number of results
        
        Returns:
            Dictionary with retrieval results:
            {
                "results": List[Dict],
                "methods_used": List[str],
                "reasoning": str,
                "confidence": float
            }
        """
        try:
            # If strategy is provided, use direct search (faster)
            if strategy:
                return self._direct_retrieval(query, strategy, limit)
            
            # Otherwise, use agent for intelligent tool selection
            # For now, use direct hybrid search as agent invocation is complex
            # In production, you would invoke the agent here
            self.log_info("Using direct hybrid search (agent invocation to be implemented)")
            results = self.hybrid_search.search(query, limit=limit)
            
            return {
                "results": results,
                "methods_used": ["hybrid"],
                "reasoning": "Used hybrid search combining all methods",
                "confidence": 0.8 if results else 0.0
            }
        except Exception as e:
            self.log_error(f"Retrieval failed: {e}")
            # Fallback to simple vector search
            try:
                results = self.vector_search.search(query, limit=limit)
                return {
                    "results": results,
                    "methods_used": ["vector"],
                    "reasoning": "Fallback to vector search",
                    "confidence": 0.5 if results else 0.0
                }
            except Exception as fallback_error:
                return {
                    "results": [],
                    "methods_used": [],
                    "reasoning": f"All retrieval methods failed: {str(fallback_error)}",
                    "confidence": 0.0
                }
    
    def _direct_retrieval(self, query: str, strategy: Dict[str, bool], limit: int) -> Dict[str, Any]:
        """
        Direct retrieval using strategy flags (bypasses agent for speed).
        
        Args:
            query: User query
            strategy: Search strategy flags
            limit: Maximum number of results
        
        Returns:
            Retrieval result dictionary
        """
        methods_used = []
        all_results = []
        
        # Use hybrid search if strategy indicates it
        if strategy.get("use_hybrid", False):
            results = self.hybrid_search.search(
                query=query,
                limit=limit,
                use_keyword=strategy.get("use_keyword", True),
                use_vector=strategy.get("use_vector", True),
                use_graph=strategy.get("use_graph", False)
            )
            all_results.extend(results)
            methods_used.append("hybrid")
        
        else:
            # Use individual methods
            if strategy.get("use_keyword", False):
                keyword_results = self.keyword_search.search(query, limit=limit)
                all_results.extend(keyword_results)
                methods_used.append("keyword")
            
            if strategy.get("use_vector", False):
                vector_results = self.vector_search.search(query, limit=limit)
                all_results.extend(vector_results)
                methods_used.append("vector")
            
            if strategy.get("use_graph", False):
                graph_results = self.graph_search.search_comprehensive(query, limit=limit)
                all_results.extend(graph_results)
                methods_used.append("graph")
        
        # Deduplicate results
        seen = set()
        unique_results = []
        for result in all_results:
            result_id = result.get("chunk_id") or result.get("file_id") or str(id(result))
            if result_id not in seen:
                seen.add(result_id)
                unique_results.append(result)
                if len(unique_results) >= limit:
                    break
        
        return {
            "results": unique_results,
            "methods_used": methods_used,
            "reasoning": f"Used {', '.join(methods_used)} search methods",
            "confidence": 0.8 if unique_results else 0.0
        }

