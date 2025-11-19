"""Build knowledge graph from extracted entities and relationships."""

from typing import List, Dict, Any, Optional, Tuple
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
        
        # STEP 1: Build entity_id_map across ALL chunks first
        # Map entity names to (entity_id, node_type) tuples
        entity_id_map = {}  # Map entity names to (entity_id, node_type)
        all_entities = []  # Collect all entities from all chunks
        
        # Helper function to normalize entity names for matching
        def normalize_entity_name(name: str) -> str:
            """Normalize entity name for comparison (lowercase, strip, remove extra spaces)."""
            return " ".join(name.lower().strip().split())
        
        # Helper function to find entity in map with fuzzy matching
        def find_entity_in_map(name: str) -> Optional[tuple]:
            """Find entity in map with case-insensitive matching."""
            # Try exact match first
            if name in entity_id_map:
                return entity_id_map[name]
            
            # Try normalized match
            normalized = normalize_entity_name(name)
            if normalized in normalized_entity_map:
                original_name = normalized_entity_map[normalized]
                return entity_id_map.get(original_name)
            
            return None
        
        # Helper function to find entity in Neo4j if not in map
        def find_entity_in_neo4j(entity_name: str) -> Optional[tuple]:
            """Find entity node in Neo4j by name (case-insensitive). Returns (entity_id, node_type) or None."""
            # Try exact match first
            for node_type in ["Person", "Organization", "Location", "Concept", "Date"]:
                query = f"""
                MATCH (n:{node_type} {{name: $name}})
                RETURN n.id as id, labels(n)[0] as label
                LIMIT 1
                """
                try:
                    results = self.client.execute_query(query, {"name": entity_name})
                    if results:
                        return (results[0].get("id"), results[0].get("label"))
                except Exception:
                    continue
            
            # Try case-insensitive match
            normalized_name = normalize_entity_name(entity_name)
            for node_type in ["Person", "Organization", "Location", "Concept", "Date"]:
                query = f"""
                MATCH (n:{node_type})
                WHERE toLower(trim(n.name)) = $normalized_name
                RETURN n.id as id, labels(n)[0] as label, n.name as original_name
                LIMIT 1
                """
                try:
                    results = self.client.execute_query(query, {"normalized_name": normalized_name})
                    if results:
                        return (results[0].get("id"), results[0].get("label"))
                except Exception:
                    continue
            
            return None
        
        # First pass: Create all entity nodes and build the map
        for result in extraction_results:
            chunk_id = result.get("chunk_id")
            entities = result.get("entities", [])
            all_entities.extend(entities)
            
            # Create entity nodes and build the map
            for entity in entities:
                entity_name = entity.get("name", "").strip()
                entity_type = entity.get("type", "Concept")
                
                if not entity_name:
                    continue
                
                # Get standardized node type
                node_type = self.schema_generator.get_node_type(entity_type)
                entity_id = self.generate_entity_id(entity_name, node_type)
                
                # Store in map: name -> (id, node_type)
                entity_id_map[entity_name] = (entity_id, node_type)
                
                # Create entity node if it doesn't exist (MERGE handles duplicates)
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
                if self.client.create_relationship(
                    source_label=content_label,
                    source_id=content_node_id,
                    source_id_key="id",
                    target_label=node_type,
                    target_id=entity_id,
                    target_id_key="id",
                    rel_type="MENTIONS",
                    properties={"chunk_id": chunk_id}
                ):
                    relationships_created += 1
        
        # Build normalized name map for fuzzy matching (after entities are created)
        normalized_entity_map = {}  # normalized_name -> original_name
        for entity_name in entity_id_map.keys():
            normalized = normalize_entity_name(entity_name)
            if normalized not in normalized_entity_map:
                normalized_entity_map[normalized] = entity_name
            # If we have multiple entities with same normalized name, prefer the first one
        
        # STEP 2: Create relationships between entities (now with full entity map)
        for result in extraction_results:
            relationships = result.get("relationships", [])
            
            for relationship in relationships:
                source_name = relationship.get("source", "").strip()
                target_name = relationship.get("target", "").strip()
                rel_type = relationship.get("relationship_type", "RELATED_TO")
                
                if not source_name or not target_name:
                    continue
                
                # Get entity info from map (with fuzzy matching)
                source_info = find_entity_in_map(source_name)
                target_info = find_entity_in_map(target_name)
                
                # If not in map, try to find in Neo4j (might be from previous chunks or files)
                if not source_info:
                    source_info = find_entity_in_neo4j(source_name)
                    if source_info:
                        entity_id_map[source_name] = source_info
                        normalized_entity_map[normalize_entity_name(source_name)] = source_name
                        logger.debug(f"Found existing entity in Neo4j: {source_name} -> {source_info[0]}")
                
                if not target_info:
                    target_info = find_entity_in_neo4j(target_name)
                    if target_info:
                        entity_id_map[target_name] = target_info
                        normalized_entity_map[normalize_entity_name(target_name)] = target_name
                        logger.debug(f"Found existing entity in Neo4j: {target_name} -> {target_info[0]}")
                
                # If still not found, skip this relationship
                if not source_info or not target_info:
                    logger.warning(
                        f"Could not find entities for relationship: "
                        f"{source_name} -> {target_name}. Skipping."
                    )
                    continue
                
                source_id, source_node_type = source_info
                target_id, target_node_type = target_info
                
                # Use correct node labels (not "Entity")
                standardized_rel_type = self.schema_generator.get_relationship_type(rel_type)
                
                if self.client.create_relationship(
                    source_label=source_node_type,  # Use actual node type (Person, Organization, etc.)
                    source_id=source_id,
                    source_id_key="id",
                    target_label=target_node_type,  # Use actual node type
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

