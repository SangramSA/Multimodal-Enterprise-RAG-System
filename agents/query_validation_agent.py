"""Query validation agent with security checks and complexity assessment."""

from typing import Dict, Any, Optional
import re
import openai
from loguru import logger

from agents.base_agent import BaseAgent
from utils.config import OPENAI_API_KEY, OPENAI_MODEL
from utils.errors import ValidationError, APIError
from pipeline.validation import InputValidator
from utils.langsmith_config import trace_agent_operation


class QueryValidationAgent(BaseAgent):
    """Agent for validating and preprocessing queries."""
    
    COMPLEXITY_KEYWORDS = {
        "simple": ["what", "who", "when", "where", "which"],
        "moderate": ["how", "why", "explain", "describe", "compare"],
        "complex": ["analyze", "evaluate", "synthesize", "relationship", "connection", "correlation"]
    }
    
    def __init__(self):
        super().__init__("QueryValidationAgent")
        self.validator = InputValidator()
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        Main validation processing method.
        
        Args:
            query: User query to validate
        
        Returns:
            Validation result dictionary
        """
        return self.validate(query)
    
    @trace_agent_operation("query_validation")
    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate query with security checks and complexity assessment.
        
        Args:
            query: User query string
        
        Returns:
            Dictionary with validation results containing:
            - is_valid: bool
            - sanitized_query: str
            - complexity: str
            - intent: str
            - security_checks: Dict[str, bool]
        """
        with self.track_operation("validate", input_data={"query_length": len(query)}):
            try:
                # Basic validation using existing validator
                validation_result = self.validator.validate_query(query)
                sanitized_query = validation_result["sanitized_query"]
                
                # Assess complexity
                complexity = self.assess_complexity(sanitized_query)
                
                # Detect intent
                intent = self.detect_intent(sanitized_query)
                
                # Security checks
                security_checks = self._perform_security_checks(sanitized_query)
                
                return {
                    "is_valid": True,
                    "sanitized_query": sanitized_query,
                    "complexity": complexity,
                    "intent": intent,
                    "security_checks": security_checks
                }
            except ValidationError as e:
                return {
                    "is_valid": False,
                    "error": str(e),
                    "sanitized_query": query,
                    "complexity": "unknown",
                    "intent": "unknown",
                    "security_checks": {}
                }
            except Exception as e:
                self.log_error(f"Validation failed: {e}")
                return self.handle_error(e, "validate")
    
    def assess_complexity(self, query: str) -> str:
        """
        Assess query complexity level.
        
        Args:
            query: Query string
        
        Returns:
            Complexity level: "simple", "moderate", or "complex"
        """
        query_lower = query.lower()
        
        # Check for complex keywords
        for keyword in self.COMPLEXITY_KEYWORDS["complex"]:
            if keyword in query_lower:
                return "complex"
        
        # Check for moderate keywords
        for keyword in self.COMPLEXITY_KEYWORDS["moderate"]:
            if keyword in query_lower:
                return "moderate"
        
        # Check query length and structure
        word_count = len(query.split())
        if word_count > 20:
            return "complex"
        elif word_count > 10:
            return "moderate"
        
        # Default to simple
        return "simple"
    
    def detect_intent(self, query: str) -> str:
        """
        Detect user intent from query.
        
        Args:
            query: Query string
        
        Returns:
            Intent type: "factual", "analytical", "exploratory", "comparative", "unknown"
        """
        query_lower = query.lower()
        
        # Factual intent
        if any(word in query_lower for word in ["what", "who", "when", "where", "which"]):
            return "factual"
        
        # Analytical intent
        if any(word in query_lower for word in ["why", "how", "explain", "analyze"]):
            return "analytical"
        
        # Exploratory intent
        if any(word in query_lower for word in ["find", "search", "show", "list"]):
            return "exploratory"
        
        # Comparative intent
        if any(word in query_lower for word in ["compare", "difference", "versus", "vs", "better"]):
            return "comparative"
        
        return "unknown"
    
    def _perform_security_checks(self, query: str) -> Dict[str, bool]:
        """
        Perform security checks on query.
        
        Args:
            query: Query string
        
        Returns:
            Dictionary with security check results
        """
        checks = {
            "sql_injection": False,
            "xss": False,
            "command_injection": False
        }
        
        # SQL injection patterns (already checked by InputValidator, but double-check)
        sql_patterns = [
            r"(?i)(union|select|insert|update|delete|drop|create|alter|exec|execute)",
            r"(?i)(--|;|/\*|\*/)",
        ]
        for pattern in sql_patterns:
            if re.search(pattern, query):
                checks["sql_injection"] = True
                break
        
        # XSS patterns
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*="
        ]
        for pattern in xss_patterns:
            if re.search(pattern, query, re.IGNORECASE | re.DOTALL):
                checks["xss"] = True
                break
        
        # Command injection patterns
        cmd_patterns = [
            r"[;&|`$()]",
            r"\$\{",
        ]
        for pattern in cmd_patterns:
            if re.search(pattern, query):
                checks["command_injection"] = True
                break
        
        return checks

