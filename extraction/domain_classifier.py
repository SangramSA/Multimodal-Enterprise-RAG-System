"""Domain classification for documents and chunks."""

from typing import List, Dict, Any, Optional
import json
import openai

from utils.logging import logger
from utils.config import OPENAI_API_KEY, OPENAI_MODEL, DOMAIN_TAGS
from utils.errors import APIError


class DomainClassifier:
    """Classify documents/chunks into domain categories."""
    
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.domain_tags = DOMAIN_TAGS
    
    def classify(self, text: str, entities: Optional[List[Dict[str, Any]]] = None, 
                 document_structure: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Classify text into domain categories.
        
        Args:
            text: Text content to classify
            entities: Optional list of extracted entities
            document_structure: Optional document structure info (headers, titles)
        
        Returns:
            List of domain tags with confidence scores
        """
        try:
            # Build context
            context_parts = []
            if entities:
                entity_types = [e.get("type") for e in entities if e.get("type")]
                context_parts.append(f"Entity types found: {', '.join(set(entity_types))}")
            
            if document_structure:
                if document_structure.get("title"):
                    context_parts.append(f"Title: {document_structure.get('title')}")
                if document_structure.get("headers"):
                    context_parts.append(f"Headers: {', '.join(document_structure.get('headers', [])[:5])}")
            
            context = "\n".join(context_parts) if context_parts else "No additional context"
            
            prompt = f"""Classify the following text into one or more domain categories from this list:
{', '.join(self.domain_tags)}

Consider:
- The content and topics discussed
- Entity types mentioned (e.g., medical entities suggest 'medical' domain)
- Document structure (titles, headers)
- Terminology and vocabulary used

Text:
{text[:2000]}  # Limit text length

Context:
{context}

Return a JSON object with:
- domains: list of {{tag, confidence}} where tag is one of the domain categories and confidence is 0-1
- reasoning: brief explanation

Only include domains with confidence >= 0.6."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at classifying documents into domain categories. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            result_json = json.loads(response.choices[0].message.content)
            
            # Parse domains
            domains = []
            for domain_data in result_json.get("domains", []):
                tag = domain_data.get("tag")
                confidence = domain_data.get("confidence", 0.0)
                
                if tag in self.domain_tags and confidence >= 0.6:
                    domains.append({
                        "tag": tag,
                        "confidence": confidence
                    })
            
            # Sort by confidence
            domains.sort(key=lambda x: x["confidence"], reverse=True)
            
            logger.info(f"Classified into domains: {[d['tag'] for d in domains]}")
            return domains
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Fallback: try to infer from entities
            return self._fallback_classify(entities)
        except Exception as e:
            logger.error(f"Domain classification failed: {e}")
            return self._fallback_classify(entities)
    
    def _fallback_classify(self, entities: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Fallback classification based on entity types."""
        if not entities:
            return []
        
        entity_types = [e.get("type") for e in entities if e.get("type")]
        type_counts = {}
        for et in entity_types:
            type_counts[et] = type_counts.get(et, 0) + 1
        
        domains = []
        
        # Simple heuristics
        if "Medical" in entity_types or "medical" in str(entity_types).lower():
            domains.append({"tag": "medical", "confidence": 0.7})
        if "Organization" in entity_types and type_counts.get("Organization", 0) > 2:
            domains.append({"tag": "finance", "confidence": 0.6})
        
        return domains
    
    def classify_chunk(self, chunk: Dict[str, Any], entities: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Classify a chunk and return list of domain tags.
        
        Returns:
            List of domain tag strings
        """
        text = chunk.get("content", "")
        
        # Extract document structure if available
        metadata = chunk.get("metadata", {})
        document_structure = {
            "title": metadata.get("title"),
            "headers": metadata.get("headers", [])
        }
        
        domains = self.classify(text, entities, document_structure)
        return [d["tag"] for d in domains]

