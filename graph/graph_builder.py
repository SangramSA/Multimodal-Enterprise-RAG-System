"""Build knowledge graph from extracted entities and relationships."""

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from utils.logging import logger
from graph.neo4j_client import Neo4jClient
from extraction.schema_generator import SchemaGenerator


class GraphBuilder:
    """Build knowledge graph from extracted data."""
    
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        self.client = neo4j_client or Neo4jClient()
        self.schema_generator = SchemaGenerator()
    
    def generate_entity_id(self, entity_name: str, entity_type: str) -> str:
        """Generate unique ID for entity."""
        # Normalize name for ID
        normalized_name = entity_name.lower().strip().replace(" ", "_")
        return f"{entity_type.lower()}_{normalized_name}"
    
    def build_from_extraction(self, extraction_results: List[Dict[str, Any]], 
                             file_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build graph from entity extraction results.
        
        Args:
            extraction_results: List of extraction results with entities and relationships
            file_metadata: Metadata about the source file
        
        Returns:
            Summary of graph construction
        """
        nodes_created = 0
        relationships_created = 0
        content_node_id = None
        
        # Create content node (Document, Image, or Audio)
        modality = file_metadata.get("modality", "document")
        
        # Map modality to proper Neo4j label
        # Schema defines: Document, Image, Audio (not "Text")
        # PDF and TXT files should both be "Document" nodes
        modality_to_label = {
            "text": "Document",      # PDF and TXT files → Document
            "document": "Document",   # Explicit document modality → Document
            "image": "Image",         # Image files → Image
            "audio": "Audio"          # Audio files → Audio
        }
        content_label = modality_to_label.get(modality, "Document")
        content_node_id = file_metadata.get("file_id")
        
        content_properties = {
            "id": content_node_id,
            "file_id": content_node_id,
            "file_name": file_metadata.get("file_name"),
            "modality": modality,
            "domain_tags": file_metadata.get("domain_tags", []),
            "upload_timestamp": file_metadata.get("upload_timestamp"),
            "created_at": datetime.utcnow().isoformat()
        }
        
        if self.client.create_node(content_label, content_properties):
            nodes_created += 1
            logger.info(f"Created {content_label} node: {content_node_id}")
        
        # Process each chunk's entities and relationships
        entity_id_map = {}  # Map entity names to node IDs
        
        for result in extraction_results:
            chunk_id = result.get("chunk_id")
            entities = result.get("entities", [])
            relationships = result.get("relationships", [])
            
            # Create entity nodes
            for entity in entities:
                entity_name = entity.get("name", "").strip()
                entity_type = entity.get("type", "Concept")
                
                if not entity_name:
                    continue
                
                # Get standardized node type
                node_type = self.schema_generator.get_node_type(entity_type)
                entity_id = self.generate_entity_id(entity_name, node_type)
                entity_id_map[entity_name] = entity_id
                
                entity_properties = {
                    "id": entity_id,
                    "name": entity_name,
                    "type": node_type,
                    "description": entity.get("description"),
                    "confidence": entity.get("confidence", 0.0),
                    "first_seen": datetime.utcnow().isoformat()
                }
                
                if self.client.create_node(node_type, entity_properties):
                    nodes_created += 1
                
                # Link entity to content node
                rel_type = "MENTIONS"
                if self.client.create_relationship(
                    source_label=content_label,
                    source_id=content_node_id,
                    source_id_key="id",
                    target_label=node_type,
                    target_id=entity_id,
                    target_id_key="id",
                    rel_type=rel_type,
                    properties={"chunk_id": chunk_id}
                ):
                    relationships_created += 1
            
            # Create relationships between entities
            for relationship in relationships:
                source_name = relationship.get("source", "").strip()
                target_name = relationship.get("target", "").strip()
                rel_type = relationship.get("relationship_type", "RELATED_TO")
                
                if not source_name or not target_name:
                    continue
                
                # Get entity IDs
                source_id = entity_id_map.get(source_name)
                target_id = entity_id_map.get(target_name)
                
                if not source_id or not target_id:
                    # Try to find entity IDs
                    source_entity = next((e for e in entities if e.get("name") == source_name), None)
                    target_entity = next((e for e in entities if e.get("name") == target_name), None)
                    
                    if source_entity:
                        source_node_type = self.schema_generator.get_node_type(source_entity.get("type"))
                        source_id = self.generate_entity_id(source_name, source_node_type)
                    if target_entity:
                        target_node_type = self.schema_generator.get_node_type(target_entity.get("type"))
                        target_id = self.generate_entity_id(target_name, target_node_type)
                
                if source_id and target_id:
                    standardized_rel_type = self.schema_generator.get_relationship_type(rel_type)
                    
                    if self.client.create_relationship(
                        source_label="Entity",  # Will match any entity type
                        source_id=source_id,
                        source_id_key="id",
                        target_label="Entity",
                        target_id=target_id,
                        target_id_key="id",
                        rel_type=standardized_rel_type,
                        properties={
                            "description": relationship.get("description"),
                            "confidence": relationship.get("confidence", 0.0)
                        }
                    ):
                        relationships_created += 1
        
        logger.success(f"Graph built: {nodes_created} nodes, {relationships_created} relationships")
        
        return {
            "nodes_created": nodes_created,
            "relationships_created": relationships_created,
            "content_node_id": content_node_id
        }
    
    def _find_existing_content_nodes_for_entity(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        Find existing content nodes in Neo4j that mention a specific entity.
        
        Args:
            entity_name: Normalized entity name (lowercase, stripped)
        
        Returns:
            List of existing content node metadata (file_id, label, modality)
        """
        try:
            return self.client.find_content_nodes_by_entity(entity_name)
        except Exception as e:
            logger.warning(f"Failed to query existing content nodes for entity '{entity_name}': {e}")
            return []
    
    def link_cross_modal_entities(self, entity_links: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
        """
        Link the same entity across different modalities, including cross-session linking.
        
        Args:
            entity_links: Dictionary mapping entity names to list of occurrence metadata from current batch
        
        Returns:
            Dictionary with:
            - same_session_links: Links created within current batch
            - cross_session_links: Links created between new and existing nodes
            - total_links: Total links created
        """
        same_session_links = 0
        cross_session_links = 0
        
        for entity_name, occurrences in entity_links.items():
            if not occurrences:
                continue
            
            # Group new occurrences by content node (file) and modality
            new_content_nodes = {}
            # Map modality to label (same as in build_from_extraction)
            modality_to_label = {
                "text": "Document",
                "document": "Document",
                "image": "Image",
                "audio": "Audio"
            }
            for occurrence in occurrences:
                file_id = occurrence.get("file_id")
                if not file_id:
                    continue
                if file_id not in new_content_nodes:
                    modality = occurrence.get("modality", "document")
                    label = modality_to_label.get(modality, "Document")
                    new_content_nodes[file_id] = {
                        "label": label,
                        "modality": modality
                    }
            
            # Query Neo4j for existing content nodes that mention this entity
            existing_content_nodes = {}
            try:
                existing_nodes = self._find_existing_content_nodes_for_entity(entity_name)
                for node in existing_nodes:
                    file_id = node.get("file_id")
                    if file_id and file_id not in new_content_nodes:  # Exclude nodes from current batch
                        existing_content_nodes[file_id] = {
                            "label": node.get("label", "Document"),
                            "modality": node.get("modality", "document")
                        }
            except Exception as e:
                logger.warning(f"Failed to query existing nodes for entity '{entity_name}': {e}")
                existing_content_nodes = {}
            
            # Combine all content nodes (new + existing)
            all_content_nodes = {**new_content_nodes, **existing_content_nodes}
            
            if len(all_content_nodes) < 2:
                continue  # Need at least two distinct content nodes to link
            
            # Create links between all pairs
            file_items = list(all_content_nodes.items())
            for i in range(len(file_items)):
                source_id, source_meta = file_items[i]
                source_is_new = source_id in new_content_nodes
                
                for j in range(i + 1, len(file_items)):
                    target_id, target_meta = file_items[j]
                    target_is_new = target_id in new_content_nodes
                    
                    if source_id == target_id:
                        continue
                    
                    # Determine if this is a cross-session link
                    is_cross_session = (source_is_new and target_id in existing_content_nodes) or \
                                     (target_is_new and source_id in existing_content_nodes)
                    
                    if self.client.create_relationship(
                        source_label=source_meta.get("label", "Document"),
                        source_id=source_id,
                        source_id_key="id",
                        target_label=target_meta.get("label", "Document"),
                        target_id=target_id,
                        target_id_key="id",
                        rel_type="CROSS_MODAL_LINK",
                        properties={
                            "entity_name": entity_name,
                            "source_modality": source_meta.get("modality"),
                            "target_modality": target_meta.get("modality")
                        }
                    ):
                        if is_cross_session:
                            cross_session_links += 1
                            logger.debug(f"Created cross-session link: {source_id} ↔ {target_id} (entity: {entity_name})")
                        else:
                            same_session_links += 1
                            logger.debug(f"Created same-session link: {source_id} ↔ {target_id} (entity: {entity_name})")
        
        total_links = same_session_links + cross_session_links
        
        if total_links > 0:
            logger.info(
                f"Cross-modal linking: {same_session_links} same-session, "
                f"{cross_session_links} cross-session, {total_links} total"
            )
        
        return {
            "same_session_links": same_session_links,
            "cross_session_links": cross_session_links,
            "total_links": total_links
        }

