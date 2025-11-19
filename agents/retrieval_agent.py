"""LangChain-based retrieval agent."""

from typing import List, Dict, Any, Optional
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from loguru import logger

from utils.config import OPENAI_API_KEY, OPENAI_MODEL
from search.hybrid_search import HybridSearch
from agents.query_rewriter import QueryRewriter


class RetrievalAgent:
    """Agent for orchestrating retrieval using LangChain."""
    
    def __init__(self, hybrid_search: HybridSearch):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        
        self.hybrid_search = hybrid_search
        self.query_rewriter = QueryRewriter()
        self.llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1, api_key=OPENAI_API_KEY)
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create LangChain agent with retrieval tools using LangChain 1.0 API."""
        # Create tool functions that have access to self
        def hybrid_search_tool_func(query: str) -> str:
            """Search across documents using hybrid search (keyword, vector, and graph). Use this for most queries."""
            return self._search_tool(query)
        
        def keyword_search_tool_func(query: str) -> str:
            """Search using keyword matching. Use for exact term matching."""
            return self._keyword_search_tool(query)
        
        def vector_search_tool_func(query: str) -> str:
            """Search using semantic similarity. Use for conceptual queries."""
            return self._vector_search_tool(query)
        
        def graph_search_tool_func(query: str) -> str:
            """Search using knowledge graph. Use for finding relationships between entities."""
            return self._graph_search_tool(query)
        
        # Wrap functions as tools
        hybrid_search_tool = tool(hybrid_search_tool_func)
        keyword_search_tool = tool(keyword_search_tool_func)
        vector_search_tool = tool(vector_search_tool_func)
        graph_search_tool = tool(graph_search_tool_func)
        
        tools = [hybrid_search_tool, keyword_search_tool, vector_search_tool, graph_search_tool]
        
        # Create agent using LangChain 1.0 API
        system_prompt = """You are a retrieval agent that helps find relevant information from a knowledge base.
You have access to multiple search tools. Choose the appropriate tool(s) based on the query.
- Use hybrid_search_tool for most queries (combines all methods)
- Use keyword_search_tool for exact term matching
- Use vector_search_tool for semantic/conceptual queries
- Use graph_search_tool for relationship queries

Always provide clear, concise results with citations."""
        
        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt
        )
        
        return agent
    
    def _search_tool(self, query: str) -> str:
        """Tool for hybrid search."""
        try:
            results = self.hybrid_search.search(query, limit=5)
            if not results:
                return "No results found."
            
            formatted = []
            for i, result in enumerate(results, 1):
                content = result.get("content", "")[:200]  # Truncate
                formatted.append(f"{i}. {content} (score: {result.get('rrf_score', 0):.3f})")
            
            return "\n".join(formatted)
        except Exception as e:
            logger.error(f"Search tool failed: {e}")
            return f"Search failed: {e}"
    
    def _keyword_search_tool(self, query: str) -> str:
        """Tool for keyword search."""
        try:
            results = self.hybrid_search.keyword_search.search(query, limit=5)
            if not results:
                return "No results found."
            
            formatted = []
            for i, result in enumerate(results, 1):
                content = result.get("content", "")[:200]
                formatted.append(f"{i}. {content} (keyword score: {result.get('keyword_score', 0):.3f})")
            
            return "\n".join(formatted)
        except Exception as e:
            logger.error(f"Keyword search tool failed: {e}")
            return f"Keyword search failed: {e}"
    
    def _vector_search_tool(self, query: str) -> str:
        """Tool for vector search."""
        try:
            results = self.hybrid_search.vector_search.search(query, limit=5)
            if not results:
                return "No results found."
            
            formatted = []
            for i, result in enumerate(results, 1):
                content = result.get("content", "")[:200]
                formatted.append(f"{i}. {content} (similarity: {result.get('score', 0):.3f})")
            
            return "\n".join(formatted)
        except Exception as e:
            logger.error(f"Vector search tool failed: {e}")
            return f"Vector search failed: {e}"
    
    def _graph_search_tool(self, query: str) -> str:
        """Tool for graph search."""
        try:
            # Extract entity names from query (simplified)
            words = query.split()
            results = []
            for word in words:
                if len(word) > 3:
                    entity_results = self.hybrid_search.graph_search.search_by_entity(word, limit=3)
                    results.extend(entity_results)
            
            if not results:
                return "No results found in knowledge graph."
            
            formatted = []
            for i, result in enumerate(results[:5], 1):
                file_name = result.get("file_name", "Unknown")
                formatted.append(f"{i}. Found in: {file_name} (match type: {result.get('match_type')})")
            
            return "\n".join(formatted)
        except Exception as e:
            logger.error(f"Graph search tool failed: {e}")
            return f"Graph search failed: {e}"
    
    def retrieve(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using the agent.
        
        Args:
            query: User query
            limit: Maximum number of results
        
        Returns:
            List of retrieved documents
        """
        try:
            # First, rewrite query
            rewritten = self.query_rewriter.rewrite_query(query)
            
            # Use hybrid search directly (simpler than full agent for now)
            results = self.hybrid_search.search(
                query=rewritten["expanded_query"],
                limit=limit,
                use_keyword=rewritten["use_keyword"],
                use_vector=rewritten["use_vector"],
                use_graph=rewritten["use_graph"]
            )
            
            return results
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            # Fallback to simple vector search
            return self.hybrid_search.vector_search.search(query, limit=limit)

