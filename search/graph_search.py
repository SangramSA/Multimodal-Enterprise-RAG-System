"""Graph search using Neo4j Cypher queries."""

from typing import List, Dict, Any, Optional
from loguru import logger

from graph.neo4j_client import Neo4jClient


class GraphSearch:
    """Search using graph traversal."""
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.client = neo4j_client
    
    def search_by_entity(self, entity_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for content related to an entity."""
        query = """
        MATCH (e)-[:MENTIONS|RELATED_TO*1..2]-(content:Document|Image|Audio)
        WHERE toLower(e.name) CONTAINS toLower($entity_name)
        RETURN DISTINCT content, labels(content) as labels
        LIMIT $limit
        """
        
        try:
            results = self.client.execute_query(query, {
                "entity_name": entity_name,
                "limit": limit
            })
            
            formatted_results = []
            for record in results:
                content = record.get("content", {})
                formatted_results.append({
                    "file_id": content.get("file_id"),
                    "file_name": content.get("file_name"),
                    "modality": content.get("modality"),
                    "domain_tags": content.get("domain_tags", []),
                    "score": 1.0,  # Graph matches get full score
                    "match_type": "entity"
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []
    
    def search_relationships(self, source_entity: str, relationship_type: Optional[str] = None,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Search for entities related to a source entity."""
        rel_pattern = f"-[:{relationship_type}]->" if relationship_type else "-->"
        
        query = f"""
        MATCH (source)-{rel_pattern}(target)
        WHERE toLower(source.name) CONTAINS toLower($source_entity)
        RETURN target, labels(target) as labels, type(relationships(source)[0]) as rel_type
        LIMIT $limit
        """
        
        try:
            results = self.client.execute_query(query, {
                "source_entity": source_entity,
                "limit": limit
            })
            
            formatted_results = []
            for record in results:
                target = record.get("target", {})
                formatted_results.append({
                    "entity_name": target.get("name"),
                    "entity_type": target.get("type"),
                    "relationship_type": record.get("rel_type"),
                    "score": 1.0,
                    "match_type": "relationship"
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Relationship search failed: {e}")
            return []
    
    def find_path(self, source_entity: str, target_entity: str, max_depth: int = 3) -> Optional[List[Dict[str, Any]]]:
        """Find shortest path between two entities."""
        query = """
        MATCH path = shortestPath((source)-[*1..$max_depth]-(target))
        WHERE toLower(source.name) CONTAINS toLower($source_entity)
          AND toLower(target.name) CONTAINS toLower($target_entity)
        RETURN path, length(path) as path_length
        LIMIT 1
        """
        
        try:
            results = self.client.execute_query(query, {
                "source_entity": source_entity,
                "target_entity": target_entity,
                "max_depth": max_depth
            })
            
            if results:
                return results[0]
            return None
        except Exception as e:
            logger.error(f"Path finding failed: {e}")
            return None
    
    def get_related_content(self, entity_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get content nodes related to an entity."""
        return self.search_by_entity(entity_name, limit)

