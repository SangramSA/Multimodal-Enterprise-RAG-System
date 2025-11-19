"""CrewAI agent wrappers for the multi-agent pipeline."""

from typing import Dict, Any, Optional
from loguru import logger

try:
    from crewai import Agent
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logger.warning("CrewAI not installed. Install with: pip install crewai")
    # Dummy Agent class for type hints
    class Agent:
        pass

from agents.query_validation_agent import QueryValidationAgent
from agents.query_triage_agent import QueryTriageAgent
from agents.retrieval_orchestration_agent import RetrievalOrchestrationAgent
from agents.answer_generation_agent import AnswerGenerationAgent
from agents.post_processing_agent import PostProcessingAgent
from utils.config import OPENAI_MODEL
from utils.telemetry import get_telemetry_collector


class CrewAIAgentFactory:
    """Factory for creating CrewAI agents from existing agent implementations."""
    
    def __init__(self, retrieval_agent: RetrievalOrchestrationAgent):
        """
        Initialize factory with existing agents.
        
        Args:
            retrieval_agent: Retrieval orchestration agent instance
        """
        self.validation_agent = QueryValidationAgent()
        self.triage_agent = QueryTriageAgent()
        self.retrieval_agent = retrieval_agent
        self.answer_agent = AnswerGenerationAgent()
        self.postprocess_agent = PostProcessingAgent()
        self.telemetry = get_telemetry_collector()
    
    def create_validator_agent(self, tools: list) -> Agent:
        """Create CrewAI agent for query validation."""
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI is not installed")
        return Agent(
            role="Query Security Validator",
            goal="Validate and sanitize user queries for security, complexity, and intent",
            backstory="""You are an expert security analyst and query validator. 
            Your role is to ensure all user queries are safe, properly formatted, and 
            assessed for complexity. You validate queries for SQL injection, XSS attacks, 
            and other security threats. You also assess query complexity and detect user intent.""",
            verbose=True,
            allow_delegation=False,
            tools=[],  # Empty tools list - we'll use direct agent calls
            llm=self._get_llm()
        )
    
    def create_triage_agent(self, tools: list) -> Agent:
        """Create CrewAI agent for query triage."""
        return Agent(
            role="Query Classifier and Strategy Selector",
            goal="Classify queries into types and select optimal search strategies",
            backstory="""You are an expert at understanding user queries and determining 
            the best approach to answer them. You classify queries into categories like 
            factual lookup, reasoning, summarization, or semantic linkage. Based on the 
            classification, you select the optimal search strategy combining keyword, 
            vector, and graph search methods.""",
            verbose=True,
            allow_delegation=False,
            tools=[],  # Empty tools list - we'll use direct agent calls
            llm=self._get_llm()
        )
    
    def create_retrieval_agent(self, tools: list) -> Agent:
        """Create CrewAI agent for document retrieval."""
        return Agent(
            role="Information Retrieval Specialist",
            goal="Retrieve relevant documents using multiple search methods (keyword, vector, graph, hybrid)",
            backstory="""You are an expert information retrieval specialist with access to 
            multiple search methods. You can use keyword search for exact matches, semantic 
            vector search for conceptual similarity, graph search for entity relationships, 
            and hybrid search that combines all methods. You intelligently select and combine 
            these methods to find the most relevant documents for a given query.""",
            verbose=True,
            allow_delegation=False,
            tools=[],  # Empty tools list - we'll use direct agent calls
            llm=self._get_llm()
        )
    
    def create_answer_generator_agent(self, tools: list) -> Agent:
        """Create CrewAI agent for answer generation."""
        return Agent(
            role="Answer Synthesis Expert",
            goal="Generate comprehensive, accurate answers from retrieved context using multi-step reasoning when needed",
            backstory="""You are an expert at synthesizing information from multiple sources 
            into coherent, accurate answers. You can perform simple factual lookups or complex 
            multi-step reasoning depending on the query complexity. You always cite your sources 
            and ensure answers are grounded in the provided context. For complex queries, you 
            break down the reasoning into clear steps.""",
            verbose=True,
            allow_delegation=False,
            tools=[],  # Empty tools list - we'll use direct agent calls
            llm=self._get_llm()
        )
    
    def create_qa_agent(self, tools: list) -> Agent:
        """Create CrewAI agent for quality assurance."""
        return Agent(
            role="Answer Quality Validator",
            goal="Validate answers, detect hallucinations, verify citations, and calculate confidence scores",
            backstory="""You are a quality assurance expert specializing in validating AI-generated 
            answers. You check answers against source documents to detect hallucinations, verify 
            that citations are correct, and assess overall answer quality. You calculate confidence 
            scores based on source coverage, citation accuracy, and answer completeness. Your goal 
            is to ensure only high-quality, well-sourced answers are returned to users.""",
            verbose=True,
            allow_delegation=False,
            tools=[],  # Empty tools list - we'll use direct agent calls
            llm=self._get_llm()
        )
    
    def _get_llm(self):
        """Get LLM configuration for CrewAI agents."""
        try:
            from langchain_openai import ChatOpenAI
            from utils.config import OPENAI_API_KEY
            return ChatOpenAI(model=OPENAI_MODEL, temperature=0.1, api_key=OPENAI_API_KEY)
        except ImportError:
            logger.warning("LangChain OpenAI not available, using default LLM")
            return None

