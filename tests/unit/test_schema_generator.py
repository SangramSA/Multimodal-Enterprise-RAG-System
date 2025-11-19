"""Unit tests for schema generator."""

import pytest
from extraction.schema_generator import SchemaGenerator


class TestSchemaGenerator:
    """Test suite for SchemaGenerator."""
    
    def test_schema_generator_initialization(self):
        """Test schema generator initialization."""
        generator = SchemaGenerator()
        assert len(generator.observed_node_types) == 0
        assert len(generator.observed_relationship_types) == 0
    
    def test_get_node_type_person(self):
        """Test node type mapping for Person."""
        generator = SchemaGenerator()
        assert generator.get_node_type("Person") == "Person"
        assert generator.get_node_type("person") == "Person"
    
    def test_get_node_type_organization(self):
        """Test node type mapping for Organization."""
        generator = SchemaGenerator()
        assert generator.get_node_type("Organization") == "Organization"
        assert generator.get_node_type("organization") == "Organization"
        assert generator.get_node_type("Company") == "Organization"
        assert generator.get_node_type("company") == "Organization"
    
    def test_get_node_type_location(self):
        """Test node type mapping for Location."""
        generator = SchemaGenerator()
        assert generator.get_node_type("Location") == "Location"
        assert generator.get_node_type("location") == "Location"
        assert generator.get_node_type("Place") == "Location"
    
    def test_get_node_type_concept(self):
        """Test node type mapping for Concept."""
        generator = SchemaGenerator()
        assert generator.get_node_type("Concept") == "Concept"
        assert generator.get_node_type("concept") == "Concept"
        assert generator.get_node_type("Topic") == "Concept"
        assert generator.get_node_type("Unknown") == "Concept"  # Default
    
    def test_get_relationship_type_works_for(self):
        """Test relationship type mapping for WORKS_FOR."""
        generator = SchemaGenerator()
        assert generator.get_relationship_type("works_for") == "WORKS_FOR"
        assert generator.get_relationship_type("works_at") == "WORKS_FOR"
        assert generator.get_relationship_type("employed_by") == "WORKS_FOR"
    
    def test_get_relationship_type_located_in(self):
        """Test relationship type mapping for LOCATED_IN."""
        generator = SchemaGenerator()
        assert generator.get_relationship_type("located_in") == "LOCATED_IN"
        assert generator.get_relationship_type("location") == "LOCATED_IN"
        assert generator.get_relationship_type("in") == "LOCATED_IN"
    
    def test_get_relationship_type_mentions(self):
        """Test relationship type mapping for MENTIONS."""
        generator = SchemaGenerator()
        assert generator.get_relationship_type("mentions") == "MENTIONS"
        assert generator.get_relationship_type("mentioned_in") == "MENTIONS"
        assert generator.get_relationship_type("appears_in") == "MENTIONS"
    
    def test_get_relationship_type_normalizes_spaces(self):
        """Test that relationship types normalize spaces."""
        generator = SchemaGenerator()
        assert generator.get_relationship_type("works for") == "WORKS_FOR"
        assert generator.get_relationship_type("related to") == "RELATED_TO"
    
    def test_update_schema_with_entities(self):
        """Test schema update with entities."""
        generator = SchemaGenerator()
        entities = [
            {"type": "Person", "name": "John"},
            {"type": "Organization", "name": "Company"},
            {"type": "Location", "name": "City"}
        ]
        
        generator.update_schema(entities, [])
        
        assert "Person" in generator.observed_node_types
        assert "Organization" in generator.observed_node_types
        assert "Location" in generator.observed_node_types
    
    def test_update_schema_with_relationships(self):
        """Test schema update with relationships."""
        generator = SchemaGenerator()
        relationships = [
            {"relationship_type": "works_for", "source": "A", "target": "B"},
            {"relationship_type": "located_in", "source": "A", "target": "B"}
        ]
        
        generator.update_schema([], relationships)
        
        assert "WORKS_FOR" in generator.observed_relationship_types
        assert "LOCATED_IN" in generator.observed_relationship_types
    
    def test_update_schema_maps_variants(self):
        """Test that schema update maps entity type variants."""
        generator = SchemaGenerator()
        entities = [
            {"type": "person", "name": "John"},  # lowercase
            {"type": "Company", "name": "Corp"}  # Company -> Organization
        ]
        
        generator.update_schema(entities, [])
        
        assert "Person" in generator.observed_node_types
        assert "Organization" in generator.observed_node_types
    
    def test_get_schema_summary(self):
        """Test getting schema summary."""
        generator = SchemaGenerator()
        generator.update_schema(
            [{"type": "Person"}, {"type": "Organization"}],
            [{"relationship_type": "works_for"}]
        )
        
        summary = generator.get_schema_summary()
        
        assert summary["total_node_types"] == 2
        assert summary["total_relationship_types"] == 1
        assert "Person" in summary["node_types"]
        assert "Organization" in summary["node_types"]
        assert "WORKS_FOR" in summary["relationship_types"]

