"""Answer generation agent with multi-step reasoning support."""

from typing import List, Dict, Any, Optional
import openai
from loguru import logger

from agents.base_agent import BaseAgent
from utils.config import OPENAI_API_KEY, OPENAI_MODEL
from utils.errors import APIError


class AnswerGenerationAgent(BaseAgent):
    """Agent for generating answers from retrieved context."""
    
    MAX_CONTEXT_LENGTH = 4000  # Maximum characters for context
    MAX_REASONING_STEPS = 20  # Maximum reasoning steps (increased to show full reasoning)
    
    def __init__(self):
        super().__init__("AnswerGenerationAgent")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
    
    def process(self, query: str, context: List[Dict], 
                reasoning_steps: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main answer generation processing method.
        
        Args:
            query: User query
            context: Retrieved context documents
            reasoning_steps: Optional reasoning steps from previous processing
        
        Returns:
            Answer generation result
        """
        return self.generate(query, context, reasoning_steps)
    
    def generate(self, query: str, context: List[Dict], 
                 reasoning_steps: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate answer from retrieved context.
        
        Args:
            query: User query
            context: List of retrieved documents
            reasoning_steps: Optional list of reasoning steps
        
        Returns:
            Dictionary with answer and metadata:
            {
                "answer": str,
                "reasoning_steps": List[str],
                "citations": List[Dict],
                "confidence": float
            }
        """
        try:
            # Assemble context
            context_text = self._assemble_context(context)
            
            # Determine if multi-step reasoning is needed
            needs_reasoning = self._needs_reasoning(query, context_text)
            
            if needs_reasoning and not reasoning_steps:
                # Generate with multi-step reasoning
                result = self._generate_with_reasoning(query, context_text)
            else:
                # Generate simple answer
                result = self._generate_simple_answer(query, context_text, reasoning_steps)
            
            # Extract citations
            citations = self._extract_citations(result["answer"], context)
            
            return {
                "answer": result["answer"],
                "reasoning_steps": result.get("reasoning_steps", []),
                "citations": citations,
                "confidence": result.get("confidence", 0.7)
            }
        except Exception as e:
            self.log_error(f"Answer generation failed: {e}")
            return self.handle_error(e, "generate")
    
    def _assemble_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Assemble context from retrieved documents.
        
        Args:
            retrieved_docs: List of retrieved documents
        
        Returns:
            Assembled context string
        """
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(retrieved_docs[:10], 1):  # Use top 10
            content = doc.get("content", "")
            chunk_id = doc.get("chunk_id", f"doc_{i}")
            modality = doc.get("metadata", {}).get("modality", "unknown")
            file_name = doc.get("metadata", {}).get("file_name", "Unknown")
            score = doc.get("rrf_score", doc.get("score", doc.get("keyword_score", 0)))
            
            # Truncate content if needed
            max_content_length = (self.MAX_CONTEXT_LENGTH - total_length) // (10 - i + 1)
            content_preview = content[:max_content_length] if len(content) > max_content_length else content
            
            context_parts.append(
                f"[Source {i} - {modality} - {file_name} - Score: {score:.3f}]\n"
                f"Chunk ID: {chunk_id}\n"
                f"{content_preview}\n"
            )
            
            total_length += len(context_parts[-1])
            
            if total_length >= self.MAX_CONTEXT_LENGTH:
                break
        
        return "\n\n".join(context_parts)
    
    def _needs_reasoning(self, query: str, context: str) -> bool:
        """
        Determine if query needs multi-step reasoning.
        
        Args:
            query: User query
            context: Retrieved context
        
        Returns:
            True if reasoning is needed, False otherwise
        """
        reasoning_keywords = ["why", "how", "explain", "analyze", "compare", "evaluate", "synthesize"]
        query_lower = query.lower()
        
        return any(keyword in query_lower for keyword in reasoning_keywords)
    
    def _generate_simple_answer(self, query: str, context: str, 
                                reasoning_steps: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate simple answer without multi-step reasoning.
        
        Args:
            query: User query
            context: Assembled context
            reasoning_steps: Optional existing reasoning steps
        
        Returns:
            Answer result dictionary
        """
        prompt = f"""You are a helpful assistant that answers questions based on provided context.
Use only the information from the context to answer. If the context doesn't contain enough information, say so.
Always cite your sources using [Source X] format.

Context:
{context}

Question: {query}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers based on the given context. Always cite your sources using [Source X] format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Estimate confidence based on answer length and context usage
            confidence = min(len(answer) / 200.0, 1.0)  # Longer answers = more confident
            
            return {
                "answer": answer,
                "reasoning_steps": reasoning_steps or [],
                "confidence": confidence
            }
        except Exception as e:
            self.log_error(f"Simple answer generation failed: {e}")
            raise APIError(f"Failed to generate answer: {e}")
    
    def _generate_with_reasoning(self, query: str, context: str) -> Dict[str, Any]:
        """
        Generate answer with multi-step reasoning.
        
        Args:
            query: User query
            context: Assembled context
        
        Returns:
            Answer result with reasoning steps
        """
        prompt = f"""You are a helpful assistant that answers complex questions using multi-step reasoning.
Break down the question into steps, reason through each step, and then provide a comprehensive answer.
Always cite your sources using [Source X] format.

Context:
{context}

Question: {query}

Provide your answer in the following format:
Step 1: [reasoning step]
Step 2: [reasoning step]
...
Final Answer: [comprehensive answer with citations]"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at multi-step reasoning. Break down complex questions and provide step-by-step analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000  # Increased to allow longer reasoning steps
            )
            
            answer_text = response.choices[0].message.content.strip()
            
            # Parse reasoning steps - capture full content of each step (including multi-line content)
            reasoning_steps = []
            final_answer = ""
            
            # Use regex to find all step patterns and final answer
            import re
            
            # Pattern to match "Step N:" or "Step N" at the start of a line
            step_pattern = re.compile(r'^Step\s+\d+[:\s]', re.IGNORECASE | re.MULTILINE)
            final_answer_pattern = re.compile(r'Final\s+Answer\s*:', re.IGNORECASE)
            
            if step_pattern.search(answer_text):
                # Find all step positions
                step_matches = list(step_pattern.finditer(answer_text))
                final_answer_match = final_answer_pattern.search(answer_text)
                
                # Extract each step's full content
                for i, match in enumerate(step_matches):
                    step_start = match.start()
                    # Find where this step ends (next step or final answer)
                    if i + 1 < len(step_matches):
                        step_end = step_matches[i + 1].start()
                    elif final_answer_match:
                        step_end = final_answer_match.start()
                    else:
                        step_end = len(answer_text)
                    
                    # Extract the full step content
                    step_content = answer_text[step_start:step_end].strip()
                    if step_content:
                        reasoning_steps.append(step_content)
                
                # Extract final answer if present
                if final_answer_match:
                    final_answer = answer_text[final_answer_match.end():].strip()
                else:
                    # If no "Final Answer:" marker, check if there's content after last step
                    if step_matches:
                        last_step_end = step_matches[-1].end()
                        remaining = answer_text[last_step_end:].strip()
                        # If remaining text is substantial and doesn't look like a step, use it as answer
                        if remaining and not step_pattern.match(remaining):
                            final_answer = remaining
            else:
                # No step markers found - check if it's structured differently
                # Look for numbered lists or other patterns
                numbered_pattern = re.compile(r'^\d+[\.\)]\s+', re.MULTILINE)
                if numbered_pattern.search(answer_text):
                    # Extract numbered items
                    matches = list(numbered_pattern.finditer(answer_text))
                    for i, match in enumerate(matches):
                        item_start = match.start()
                        if i + 1 < len(matches):
                            item_end = matches[i + 1].start()
                        else:
                            item_end = len(answer_text)
                        item_content = answer_text[item_start:item_end].strip()
                        if item_content:
                            reasoning_steps.append(item_content)
            
            # If no steps found, use entire answer as single reasoning step (don't truncate)
            if not reasoning_steps:
                reasoning_steps = [answer_text]
            
            # If no final answer extracted, use the last reasoning step or full text
            if not final_answer:
                if reasoning_steps:
                    # Check if last step looks like an answer
                    last_step = reasoning_steps[-1]
                    if "answer" in last_step.lower()[:50] or len(last_step) > 200:
                        final_answer = last_step
                    else:
                        final_answer = answer_text
                else:
                    final_answer = answer_text
            
            confidence = min(len(answer_text) / 300.0, 1.0)  # Longer reasoning = more confident
            
            return {
                "answer": final_answer if final_answer else answer_text,
                "reasoning_steps": reasoning_steps[:self.MAX_REASONING_STEPS],
                "confidence": confidence
            }
        except Exception as e:
            self.log_error(f"Reasoning answer generation failed: {e}")
            # Fallback to simple answer
            return self._generate_simple_answer(query, context)
    
    def _extract_citations(self, answer: str, sources: List[Dict]) -> List[Dict]:
        """
        Extract citations from answer and map to sources.
        
        Args:
            answer: Generated answer text
            sources: List of source documents
        
        Returns:
            List of citation dictionaries
        """
        citations = []
        
        # Look for [Source X] patterns in answer
        import re
        citation_pattern = r"\[Source\s+(\d+)\]"
        matches = re.findall(citation_pattern, answer, re.IGNORECASE)
        
        cited_indices = set(int(match) - 1 for match in matches)  # Convert to 0-based
        
        # Create citation entries for cited sources
        for i, source in enumerate(sources[:10]):
            if i in cited_indices or len(citations) < 3:  # Always include top 3
                citations.append({
                    "source_id": i + 1,
                    "chunk_id": source.get("chunk_id"),
                    "file_id": source.get("metadata", {}).get("file_id"),
                    "file_name": source.get("metadata", {}).get("file_name", "Unknown"),
                    "modality": source.get("metadata", {}).get("modality", "unknown"),
                    "content_preview": source.get("content", "")[:200],
                    "score": source.get("rrf_score", source.get("score", source.get("keyword_score", 0)))
                })
        
        return citations

