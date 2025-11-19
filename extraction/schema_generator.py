"""Dynamic schema generation for knowledge graph."""

from typing import Dict, List, Set, Any
from utils.logging import logger


class SchemaGenerator:
    """Generate and manage graph database schema."""
    
    # Node types
    NODE_TYPES = {
        "Person": "Person",
        "Organization": "Organization",
        "Location": "Location",
        "Concept": "Concept",
        "Date": "Date",
        "Document": "Document",
        "Image": "Image",
        "Audio": "Audio"
    }
    
    # Relationship types
    RELATIONSHIP_TYPES = {
        "MENTIONS": "MENTIONS",
        "CONTAINS": "CONTAINS",
        "RELATED_TO": "RELATED_TO",
        "WORKS_FOR": "WORKS_FOR",
        "LOCATED_IN": "LOCATED_IN",
        "OCCURS_IN": "OCCURS_IN",
        "REFERENCES": "REFERENCES",
        "CROSS_MODAL_LINK": "CROSS_MODAL_LINK"
    }
    
    def __init__(self):
        self.observed_node_types: Set[str] = set()
        self.observed_relationship_types: Set[str] = set()
    
    def update_schema(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]):
        """Update schema based on observed entities and relationships."""
        for entity in entities:
            entity_type = entity.get("type")
            if entity_type:
                # Map to standard node type
                if entity_type in ["Person", "person"]:
                    self.observed_node_types.add("Person")
                elif entity_type in ["Organization", "organization", "Company", "company"]:
                    self.observed_node_types.add("Organization")
                elif entity_type in ["Location", "location", "Place", "place"]:
                    self.observed_node_types.add("Location")
                elif entity_type in ["Concept", "concept", "Topic", "topic"]:
                    self.observed_node_types.add("Concept")
                elif entity_type in ["Date", "date"]:
                    self.observed_node_types.add("Date")
        
        for relationship in relationships:
            rel_type = relationship.get("relationship_type", "").upper()
            if rel_type:
                # Map to standard relationship type
                if rel_type in ["WORKS_FOR", "EMPLOYED_BY", "WORKS_AT"]:
                    self.observed_relationship_types.add("WORKS_FOR")
                elif rel_type in ["LOCATED_IN", "LOCATION", "IN"]:
                    self.observed_relationship_types.add("LOCATED_IN")
                elif rel_type in ["MENTIONS", "MENTIONED_IN", "APPEARS_IN"]:
                    self.observed_relationship_types.add("MENTIONS")
                elif rel_type in ["RELATED_TO", "RELATED", "ASSOCIATED_WITH"]:
                    self.observed_relationship_types.add("RELATED_TO")
                else:
                    self.observed_relationship_types.add(rel_type)
    
    def get_node_type(self, entity_type: str) -> str:
        """Get standardized node type."""
        type_mapping = {
            "Person": "Person",
            "person": "Person",
            "Organization": "Organization",
            "organization": "Organization",
            "Company": "Organization",
            "company": "Organization",
            "Location": "Location",
            "location": "Location",
            "Place": "Location",
            "place": "Location",
            "Concept": "Concept",
            "concept": "Concept",
            "Topic": "Concept",
            "topic": "Concept",
            "Date": "Date",
            "date": "Date"
        }
        return type_mapping.get(entity_type, "Concept")
    
    def get_relationship_type(self, rel_type: str) -> str:
        """Get standardized relationship type."""
        rel_mapping = {
            "works_for": "WORKS_FOR",
            "works_at": "WORKS_FOR",
            "employed_by": "WORKS_FOR",
            "located_in": "LOCATED_IN",
            "location": "LOCATED_IN",
            "in": "LOCATED_IN",
            "mentions": "MENTIONS",
            "mentioned_in": "MENTIONS",
            "appears_in": "MENTIONS",
            "related_to": "RELATED_TO",
            "related": "RELATED_TO",
            "associated_with": "RELATED_TO",
            "contains": "CONTAINS",
            "references": "REFERENCES"
        }
        normalized = rel_type.lower().replace(" ", "_")
        return rel_mapping.get(normalized, rel_type.upper())
    
    def get_schema_summary(self) -> Dict[str, Any]:
        """Get summary of current schema."""
        return {
            "node_types": sorted(list(self.observed_node_types)),
            "relationship_types": sorted(list(self.observed_relationship_types)),
            "total_node_types": len(self.observed_node_types),
            "total_relationship_types": len(self.observed_relationship_types)
        }

