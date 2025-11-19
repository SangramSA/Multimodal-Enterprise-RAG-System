"""Entity extraction using GPT-4 with structured output."""

from typing import List, Dict, Any, Optional
import json
import openai
from pydantic import BaseModel, Field

from utils.logging import logger
from utils.config import OPENAI_API_KEY, OPENAI_MODEL
from utils.errors import APIError


class Entity(BaseModel):
    """Entity model."""
    name: str = Field(description="Name of the entity")
    type: str = Field(description="Type: Person, Organization, Location, Concept, Date, or Other")
    description: Optional[str] = Field(None, description="Brief description or context")
    confidence: float = Field(0.0, description="Confidence score 0-1")


class Relationship(BaseModel):
    """Relationship model."""
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relationship_type: str = Field(description="Type of relationship")
    description: Optional[str] = Field(None, description="Description of the relationship")
    confidence: float = Field(0.0, description="Confidence score 0-1")


class EntityExtractionResult(BaseModel):
    """Result of entity extraction."""
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)


class EntityExtractor:
    """Extract entities and relationships from text using GPT-4."""
    
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
    
    def extract(self, text: str, context: Optional[Dict[str, Any]] = None) -> EntityExtractionResult:
        """
        Extract entities and relationships from text.
        
        Args:
            text: Text to extract entities from
            context: Optional context metadata (file_id, modality, etc.)
        
        Returns:
            EntityExtractionResult with entities and relationships
        """
        try:
            context_str = ""
            if context:
                context_str = f"\nContext: File ID: {context.get('file_id')}, Modality: {context.get('modality')}"
            
            prompt = f"""Extract entities and relationships from the following text. 
Focus on:
- Persons (names of people)
- Organizations (companies, institutions, groups)
- Locations (places, cities, countries)
- Concepts (important ideas, topics, technologies)
- Dates (specific dates mentioned)
- Relationships between entities (works_for, located_in, mentions, related_to, etc.)

Text:{context_str}
{text}

Return a JSON object with:
- entities: list of {{name, type, description, confidence}}
- relationships: list of {{source, target, relationship_type, description, confidence}}

Be thorough but accurate. Only extract entities you're confident about (confidence > 0.7)."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured information from text. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result_json = json.loads(response.choices[0].message.content)
            
            # Parse entities
            entities = []
            for entity_data in result_json.get("entities", []):
                try:
                    entity = Entity(**entity_data)
                    # Filter by confidence
                    if entity.confidence >= 0.7:
                        entities.append(entity)
                except Exception as e:
                    logger.warning(f"Failed to parse entity: {entity_data}, error: {e}")
            
            # Parse relationships
            relationships = []
            for rel_data in result_json.get("relationships", []):
                try:
                    relationship = Relationship(**rel_data)
                    # Filter by confidence
                    if relationship.confidence >= 0.7:
                        relationships.append(relationship)
                except Exception as e:
                    logger.warning(f"Failed to parse relationship: {rel_data}, error: {e}")
            
            logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
            return EntityExtractionResult(entities=entities, relationships=relationships)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise APIError(f"Invalid JSON response from API: {e}")
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            raise APIError(f"Failed to extract entities: {e}")
    
    def extract_from_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract entities from multiple chunks."""
        results = []
        for chunk in chunks:
            try:
                context = {
                    "file_id": chunk.get("file_id"),
                    "modality": chunk.get("metadata", {}).get("modality"),
                    "chunk_id": chunk.get("chunk_id")
                }
                extraction_result = self.extract(chunk.get("content", ""), context)
                
                results.append({
                    "chunk_id": chunk.get("chunk_id"),
                    "entities": [e.dict() for e in extraction_result.entities],
                    "relationships": [r.dict() for r in extraction_result.relationships]
                })
            except Exception as e:
                logger.error(f"Failed to extract from chunk {chunk.get('chunk_id')}: {e}")
                results.append({
                    "chunk_id": chunk.get("chunk_id"),
                    "entities": [],
                    "relationships": [],
                    "error": str(e)
                })
        
        return results
    
    def link_entities_across_modalities(
        self,
        extraction_results: List[Dict[str, Any]],
        chunks: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Link the same entity across different modalities.
        
        Returns:
            Dictionary mapping entity names to list of occurrence metadata
        """
        entity_occurrences: Dict[str, List[Dict[str, Any]]] = {}
        # Map modality to proper label (Document, Image, Audio)
        # PDF and TXT files should both be "Document" nodes
        modality_to_label = {
            "text": "Document",
            "document": "Document",
            "image": "Image",
            "audio": "Audio"
        }
        chunk_metadata_map = {
            chunk.get("chunk_id"): {
                "file_id": chunk.get("metadata", {}).get("file_id"),
                "modality": chunk.get("metadata", {}).get("modality", "document"),
                "content_label": modality_to_label.get(
                    chunk.get("metadata", {}).get("modality", "document"),
                    "Document"
                ),
            }
            for chunk in (chunks or [])
        }
        
        for result in extraction_results:
            chunk_id = result.get("chunk_id")
            metadata = chunk_metadata_map.get(chunk_id, {})
            for entity in result.get("entities", []):
                entity_name = entity.get("name", "").lower().strip()
                if entity_name:
                    if entity_name not in entity_occurrences:
                        entity_occurrences[entity_name] = []
                    entity_occurrences[entity_name].append({
                        "chunk_id": chunk_id,
                        "file_id": metadata.get("file_id"),
                        "modality": metadata.get("modality", "document"),
                        "content_label": metadata.get("content_label", "Document")
                    })
        
        return entity_occurrences

