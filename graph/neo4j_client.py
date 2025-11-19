"""Neo4j client for graph database operations."""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver
from loguru import logger

from utils.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from utils.errors import DatabaseError
from utils.errors import retry_with_backoff


class Neo4jClient:
    """Client for Neo4j graph database operations."""
    
    def __init__(self):
        self.uri = NEO4J_URI
        self.user = NEO4J_USER
        self.password = NEO4J_PASSWORD
        self.driver: Optional[Driver] = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Verify connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.success("Connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise DatabaseError(f"Neo4j connection failed: {e}")
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query."""
        if not self.driver:
            self._connect()
        
        def _execute():
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        
        try:
            return retry_with_backoff(_execute, max_retries=3)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise DatabaseError(f"Query failed: {e}")
    
    def create_node(self, label: str, properties: Dict[str, Any], unique_id: str = "id") -> bool:
        """Create or merge a node."""
        if unique_id not in properties:
            raise ValueError(f"Property '{unique_id}' is required for unique identification")
        
        query = f"""
        MERGE (n:{label} {{{unique_id}: $id}})
        SET n += $props
        RETURN n
        """
        
        try:
            self.execute_query(query, {
                "id": properties[unique_id],
                "props": properties
            })
            return True
        except Exception as e:
            logger.error(f"Failed to create node: {e}")
            return False
    
    def create_relationship(self, source_label: str, source_id: str, source_id_key: str,
                           target_label: str, target_id: str, target_id_key: str,
                           rel_type: str, properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create or merge a relationship."""
        query = f"""
        MATCH (a:{source_label} {{{source_id_key}: $source_id}})
        MATCH (b:{target_label} {{{target_id_key}: $target_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += $props
        RETURN r
        """
        
        try:
            self.execute_query(query, {
                "source_id": source_id,
                "target_id": target_id,
                "props": properties or {}
            })
            return True
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return False
    
    def find_nodes(self, label: str, filters: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Find nodes by label and optional filters."""
        where_clauses = []
        params = {}
        
        if filters:
            for key, value in filters.items():
                where_clauses.append(f"n.{key} = ${key}")
                params[key] = value
        
        where_clause = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        query = f"MATCH (n:{label}){where_clause} RETURN n LIMIT $limit"
        params["limit"] = limit
        
        try:
            results = self.execute_query(query, params)
            return [record["n"] for record in results]
        except Exception as e:
            logger.error(f"Failed to find nodes: {e}")
            return []
    
    def find_related_nodes(self, node_id: str, node_label: str, relationship_type: Optional[str] = None,
                          depth: int = 1, limit: int = 50) -> List[Dict[str, Any]]:
        """Find nodes related to a given node."""
        rel_pattern = f"-[:{relationship_type}*1..{depth}]->" if relationship_type else f"-[*1..{depth}]->"
        
        query = f"""
        MATCH (n:{node_label} {{id: $node_id}}){rel_pattern}(related)
        RETURN DISTINCT related, labels(related) as labels
        LIMIT $limit
        """
        
        try:
            results = self.execute_query(query, {"node_id": node_id, "limit": limit})
            return results
        except Exception as e:
            logger.error(f"Failed to find related nodes: {e}")
            return []
    
    def delete_node(self, label: str, node_id: str, id_key: str = "id") -> bool:
        """Delete a node and its relationships."""
        query = f"""
        MATCH (n:{label} {{{id_key}: $id}})
        DETACH DELETE n
        RETURN count(n) as deleted
        """
        
        try:
            result = self.execute_query(query, {"id": node_id})
            return result[0].get("deleted", 0) > 0
        except Exception as e:
            logger.error(f"Failed to delete node: {e}")
            return False
    
    def health_check(self) -> bool:
        """Check if database is accessible."""
        try:
            self.execute_query("RETURN 1")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def find_content_node_by_file_id(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a content node (Document/Image/Audio) by file_id.
        
        Args:
            file_id: The file ID to search for
        
        Returns:
            Dictionary with node data if found, None otherwise
        """
        query = """
        MATCH (content:Document|Image|Audio {id: $file_id})
        RETURN content, labels(content)[0] as label
        LIMIT 1
        """
        
        try:
            results = self.execute_query(query, {"file_id": file_id})
            if results:
                node_data = results[0].get("content", {})
                return {
                    "file_id": node_data.get("id"),
                    "file_name": node_data.get("file_name"),
                    "modality": node_data.get("modality"),
                    "domain_tags": node_data.get("domain_tags", []),
                    "upload_timestamp": node_data.get("upload_timestamp"),
                    "label": results[0].get("label", "Document")
                }
            return None
        except Exception as e:
            logger.warning(f"Failed to find content node for file_id '{file_id}': {e}")
            return None
    
    def find_content_nodes_by_entity(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        Find all content nodes (Document/Image/Audio) that mention a specific entity.
        
        Args:
            entity_name: Normalized entity name (lowercase, stripped)
        
        Returns:
            List of dictionaries with file_id, label, and modality
        """
        # Generate entity ID to find the entity node
        # Entity IDs are in format: {type}_{normalized_name}
        # We need to search for entities with matching name
        query = """
        MATCH (content:Document|Image|Audio)-[:MENTIONS]->(entity)
        WHERE toLower(entity.name) = $entity_name
        RETURN DISTINCT content.id as file_id, 
               labels(content)[0] as label, 
               content.modality as modality
        """
        
        try:
            results = self.execute_query(query, {"entity_name": entity_name.lower().strip()})
            return [
                {
                    "file_id": record.get("file_id"),
                    "label": record.get("label", "Document"),
                    "modality": record.get("modality", "document")
                }
                for record in results
                if record.get("file_id")
            ]
        except Exception as e:
            logger.warning(f"Failed to find content nodes for entity '{entity_name}': {e}")
            return []

