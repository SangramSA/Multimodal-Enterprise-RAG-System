"""Post-processing agent for answer validation and quality assurance."""

from typing import List, Dict, Any, Optional
import openai
from loguru import logger

from agents.base_agent import BaseAgent
from utils.config import OPENAI_API_KEY, OPENAI_MODEL
from utils.errors import APIError
from agents.utils import calculate_confidence_from_scores


class PostProcessingAgent(BaseAgent):
    """Agent for post-processing and validating answers."""
    
    HALLUCINATION_THRESHOLD = 0.3  # Above this, likely hallucination
    
    def __init__(self):
        super().__init__("PostProcessingAgent")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
    
    def process(self, answer: str, sources: List[Dict], query: str) -> Dict[str, Any]:
        """
        Main post-processing method.
        
        Args:
            answer: Generated answer
            sources: Source documents
            query: Original query
        
        Returns:
            Post-processed result dictionary
        """
        # Validate answer
        validation = self.validate_answer(answer, sources)
        
        # Detect hallucinations
        hallucination_score = self.detect_hallucinations(answer, sources)
        
        # Verify citations
        citation_verification = self.verify_citations(answer, sources)
        
        # Calculate confidence
        confidence = self.calculate_confidence(answer, sources, validation, hallucination_score)
        
        # Format response
        formatted_response = self._format_response(answer, sources, validation)
        
        return {
            "final_answer": formatted_response,
            "confidence": confidence,
            "hallucination_score": hallucination_score,
            "citation_verification": citation_verification,
            "validation": validation
        }
    
    def validate_answer(self, answer: str, sources: List[Dict]) -> Dict[str, Any]:
        """
        Validate answer against sources.
        
        Args:
            answer: Generated answer
            sources: Source documents
        
        Returns:
            Validation result dictionary
        """
        validation_results = {
            "is_valid": True,
            "issues": [],
            "source_coverage": 0.0
        }
        
        # Check if answer is empty
        if not answer or not answer.strip():
            validation_results["is_valid"] = False
            validation_results["issues"].append("Answer is empty")
            return validation_results
        
        # Check if answer references sources
        import re
        citation_pattern = r"\[Source\s+\d+\]"
        citations = re.findall(citation_pattern, answer, re.IGNORECASE)
        
        if not citations:
            validation_results["issues"].append("Answer lacks source citations")
        
        # Check source coverage
        if sources:
            validation_results["source_coverage"] = len(citations) / min(len(sources), 5)  # Normalize to top 5 sources
        
        # Check for "I don't know" or similar phrases
        uncertainty_phrases = ["i don't know", "i cannot", "i'm not sure", "unable to", "no information"]
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in uncertainty_phrases):
            validation_results["issues"].append("Answer indicates uncertainty")
        
        return validation_results
    
    def detect_hallucinations(self, answer: str, sources: List[Dict]) -> float:
        """
        Detect potential hallucinations in answer.
        
        Args:
            answer: Generated answer
            sources: Source documents
        
        Returns:
            Hallucination score (0-1, higher = more likely hallucination)
        """
        if not sources:
            return 0.5  # Can't verify without sources
        
        # Extract key claims from answer
        claims = self._extract_claims(answer)
        
        if not claims:
            return 0.0  # No specific claims to verify
        
        # Check each claim against sources
        verification_scores = []
        for claim in claims[:5]:  # Limit to 5 claims
            score = self._verify_claim_against_sources(claim, sources)
            verification_scores.append(score)
        
        if not verification_scores:
            return 0.5  # Can't verify
        
        # Hallucination score is inverse of average verification
        avg_verification = sum(verification_scores) / len(verification_scores)
        hallucination_score = 1.0 - avg_verification
        
        return min(hallucination_score, 1.0)
    
    def verify_citations(self, answer: str, sources: List[Dict]) -> Dict[str, Any]:
        """
        Verify citations in answer.
        
        Args:
            answer: Generated answer
            sources: Source documents
        
        Returns:
            Citation verification result
        """
        import re
        
        citation_pattern = r"\[Source\s+(\d+)\]"
        cited_indices = [int(match) - 1 for match in re.findall(citation_pattern, answer, re.IGNORECASE)]
        
        verification = {
            "total_citations": len(cited_indices),
            "valid_citations": 0,
            "invalid_citations": [],
            "missing_citations": []
        }
        
        # Check if cited sources exist
        for idx in cited_indices:
            if 0 <= idx < len(sources):
                verification["valid_citations"] += 1
            else:
                verification["invalid_citations"].append(idx + 1)
        
        # Check if important sources are missing citations
        top_sources = sources[:3]  # Top 3 sources should be cited
        for i, source in enumerate(top_sources):
            if i not in cited_indices:
                verification["missing_citations"].append(i + 1)
        
        return verification
    
    def calculate_confidence(self, answer: str, sources: List[Dict], 
                           validation: Dict[str, Any], hallucination_score: float) -> float:
        """
        Calculate overall confidence score.
        
        Args:
            answer: Generated answer
            sources: Source documents
            validation: Validation results
            hallucination_score: Hallucination detection score
        
        Returns:
            Confidence score (0-1)
        """
        confidence_factors = []
        
        # Factor 1: Answer length (longer = more confident, up to a point)
        answer_length_score = min(len(answer) / 200.0, 1.0)
        confidence_factors.append(answer_length_score * 0.2)
        
        # Factor 2: Source count
        source_count_score = min(len(sources) / 5.0, 1.0)
        confidence_factors.append(source_count_score * 0.2)
        
        # Factor 3: Source coverage
        coverage_score = validation.get("source_coverage", 0.0)
        confidence_factors.append(coverage_score * 0.2)
        
        # Factor 4: Low hallucination
        hallucination_factor = (1.0 - hallucination_score) * 0.2
        confidence_factors.append(hallucination_factor)
        
        # Factor 5: Validation status
        validation_score = 1.0 if validation.get("is_valid", False) and len(validation.get("issues", [])) == 0 else 0.5
        confidence_factors.append(validation_score * 0.2)
        
        # Calculate weighted average
        confidence = sum(confidence_factors)
        
        return min(max(confidence, 0.0), 1.0)
    
    def _format_response(self, answer: str, sources: List[Dict], 
                        validation: Dict[str, Any]) -> str:
        """
        Format final response for display.
        
        Args:
            answer: Generated answer
            sources: Source documents
            validation: Validation results
        
        Returns:
            Formatted response string
        """
        # For now, return answer as-is
        # In production, could add formatting, markdown, etc.
        return answer
    
    def _extract_claims(self, answer: str) -> List[str]:
        """
        Extract key claims from answer.
        
        Args:
            answer: Answer text
        
        Returns:
            List of extracted claims
        """
        # Simple extraction: sentences that make factual statements
        import re
        
        sentences = re.split(r'[.!?]+', answer)
        claims = []
        
        # Look for sentences with factual indicators
        factual_indicators = ["is", "are", "was", "were", "has", "have", "contains", "includes"]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Skip very short sentences
                # Check if sentence contains factual indicators
                if any(indicator in sentence.lower() for indicator in factual_indicators):
                    claims.append(sentence[:200])  # Truncate long claims
        
        return claims[:5]  # Limit to 5 claims
    
    def _verify_claim_against_sources(self, claim: str, sources: List[Dict]) -> float:
        """
        Verify a claim against source documents.
        
        Args:
            claim: Claim to verify
            sources: Source documents
        
        Returns:
            Verification score (0-1, higher = more verified)
        """
        # Simple keyword matching
        claim_words = set(claim.lower().split())
        
        max_match = 0.0
        for source in sources[:5]:  # Check top 5 sources
            content = source.get("content", "").lower()
            content_words = set(content.split())
            
            # Calculate word overlap
            overlap = len(claim_words & content_words)
            total_claim_words = len(claim_words)
            
            if total_claim_words > 0:
                match_score = overlap / total_claim_words
                max_match = max(max_match, match_score)
        
        return max_match

