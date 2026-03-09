"""Post-processing agent for answer validation and quality assurance."""

from typing import List, Dict, Any, Optional
import json
import openai
from loguru import logger

from agents.base_agent import BaseAgent
from utils.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    USE_LLM_JUDGE,
    LLM_JUDGE_MODEL,
)
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
        self.use_llm_judge = USE_LLM_JUDGE
        self.judge_model = LLM_JUDGE_MODEL or self.model
    
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
        # Optional LLM-as-judge evaluation
        judge_result: Optional[Dict[str, Any]] = None
        if self.use_llm_judge:
            try:
                judge_result = self.llm_judge(query=query, answer=answer, sources=sources)
            except Exception as e:
                logger.warning(f"LLM judge evaluation failed, falling back to heuristic-only scores: {e}")
                judge_result = None
        
        # Validate answer
        validation = self.validate_answer(answer, sources)
        
        # Detect hallucinations (LLM judge result takes precedence when available)
        hallucination_score = self.detect_hallucinations(answer, sources, judge_result=judge_result)
        
        # Verify citations
        citation_verification = self.verify_citations(answer, sources)
        
        # Calculate confidence (optionally blended with LLM judge confidence)
        confidence = self.calculate_confidence(
            answer,
            sources,
            validation,
            hallucination_score,
            judge_result=judge_result,
        )
        
        # Format response
        formatted_response = self._format_response(answer, sources, validation)
        
        result: Dict[str, Any] = {
            "final_answer": formatted_response,
            "confidence": confidence,
            "hallucination_score": hallucination_score,
            "citation_verification": citation_verification,
            "validation": validation,
        }
        
        # Attach raw judge output for downstream telemetry / UI when available
        if judge_result is not None:
            result["llm_judge"] = judge_result
        
        return result
    
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
    
    def detect_hallucinations(
        self,
        answer: str,
        sources: List[Dict],
        judge_result: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Detect potential hallucinations in answer.
        
        Args:
            answer: Generated answer
            sources: Source documents
        
        Returns:
            Hallucination score (0-1, higher = more likely hallucination)
        """
        # If an LLM judge result is available, prefer its hallucination / faithfulness signal
        if judge_result:
            # Direct hallucination score from judge (0-1, higher = more hallucination)
            raw_hallucination = judge_result.get("hallucination_score")
            if isinstance(raw_hallucination, (int, float)):
                try:
                    score = float(raw_hallucination)
                    return max(0.0, min(score, 1.0))
                except (TypeError, ValueError):
                    pass
            
            # Derive hallucination score from faithfulness when provided
            faithfulness = judge_result.get("faithfulness_score")
            if isinstance(faithfulness, (int, float)):
                try:
                    faithfulness_score = float(faithfulness)
                    hallucination_score = 1.0 - max(0.0, min(faithfulness_score, 1.0))
                    return max(0.0, min(hallucination_score, 1.0))
                except (TypeError, ValueError):
                    pass
        
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
    
    def calculate_confidence(
        self,
        answer: str,
        sources: List[Dict],
        validation: Dict[str, Any],
        hallucination_score: float,
        judge_result: Optional[Dict[str, Any]] = None,
    ) -> float:
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
        
        # Base heuristic confidence
        base_confidence = sum(confidence_factors)
        base_confidence = min(max(base_confidence, 0.0), 1.0)
        
        # Optionally blend with LLM judge confidence when available
        judge_confidence: Optional[float] = None
        if judge_result:
            raw_conf = judge_result.get("confidence_score")
            if isinstance(raw_conf, (int, float)):
                try:
                    judge_confidence = float(raw_conf)
                    judge_confidence = min(max(judge_confidence, 0.0), 1.0)
                except (TypeError, ValueError):
                    judge_confidence = None
        
        if judge_confidence is not None:
            # Simple blend: heuristic (60%) + judge confidence (40%)
            blended = 0.6 * base_confidence + 0.4 * judge_confidence
            return min(max(blended, 0.0), 1.0)
        
        return base_confidence
    
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

    def llm_judge(
        self,
        query: str,
        answer: str,
        sources: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Use an LLM-as-judge to evaluate answer faithfulness and confidence.
        
        The judge is asked to return a compact JSON object with fields:
          - faithfulness_score: float (0-1, higher = more grounded in sources)
          - hallucination_score: float (0-1, higher = more hallucinated)
          - confidence_score: float (0-1, higher = more confident)
          - rationale: str (short explanation)
        """
        # Prepare a compact list of source snippets for the judge
        source_snippets = []
        for i, src in enumerate(sources[:5]):
            content = src.get("content") or src.get("content_preview") or ""
            if not content:
                continue
            source_snippets.append(
                {
                    "index": i,
                    "file_name": src.get("file_name"),
                    "modality": src.get("modality"),
                    "snippet": content[:500],
                }
            )
        
        system_prompt = (
            "You are an expert QA judge evaluating whether an answer is grounded in the provided sources.\n"
            "Carefully compare the answer to the source snippets. Pay attention to factual claims, numbers, and entities.\n"
            "Respond ONLY with a JSON object and no additional text."
        )
        
        user_prompt = {
            "query": query,
            "answer": answer,
            "sources": source_snippets,
            "instructions": (
                "Return a JSON object with the following keys:\n"
                "  faithfulness_score: number between 0 and 1 (1 = fully grounded in sources).\n"
                "  hallucination_score: number between 0 and 1 (1 = mostly hallucinated).\n"
                "  confidence_score: number between 0 and 1 (your confidence in the answer quality).\n"
                "  rationale: short explanation (1-3 sentences) of your judgment."
            ),
        }
        
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_prompt)},
                ],
            )
            content = response.choices[0].message.content
            if not content:
                logger.warning("LLM judge returned empty content")
                return None
            
            judge_json = json.loads(content)
            if not isinstance(judge_json, dict):
                logger.warning(f"LLM judge response is not a JSON object: {content[:200]}")
                return None
            
            # Ensure expected keys exist (fill with defaults if missing)
            result: Dict[str, Any] = {}
            for key in ("faithfulness_score", "hallucination_score", "confidence_score"):
                value = judge_json.get(key)
                if isinstance(value, (int, float)):
                    try:
                        v = float(value)
                        result[key] = max(0.0, min(v, 1.0))
                    except (TypeError, ValueError):
                        continue
            rationale = judge_json.get("rationale")
            if isinstance(rationale, str):
                result["rationale"] = rationale
            
            # Return None if nothing useful was parsed
            if not result:
                logger.warning(f"LLM judge returned JSON without usable scores: {judge_json}")
                return None
            
            return result
        except Exception as e:
            logger.warning(f"LLM judge call failed: {e}")
            return None

