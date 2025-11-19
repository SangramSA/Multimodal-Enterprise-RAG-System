"""Graph search using Neo4j Cypher queries."""

from typing import List, Dict, Any, Optional
from loguru import logger

from graph.neo4j_client import Neo4jClient
from agents.utils import extract_entities_from_query, format_graph_results


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
    
    def search_comprehensive(self, query: str, search_type: str = "auto", 
                            entity_names: Optional[List[str]] = None,
                            relationship_type: Optional[str] = None,
                            max_depth: int = 2, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Comprehensive graph search that handles all graph operations.
        
        This method can:
        - Extract entities from natural language queries
        - Search for entities in the knowledge graph
        - Traverse relationships between entities
        - Find paths between entities
        - Find content (documents/images/audio) related to entities
        
        Args:
            query: Natural language query or entity name(s)
            search_type: 
                - "auto": Automatically determine search type from query
                - "entity": Search for specific entities
                - "relationship": Find relationships from entities
                - "path": Find path between two entities
                - "content": Find content nodes related to entities
            entity_names: Optional explicit list of entity names (if not provided, extracted from query)
            relationship_type: Optional specific relationship type to traverse
            max_depth: Maximum depth for graph traversal (default: 2)
            limit: Maximum number of results
        
        Returns:
            List of search results
        """
        # Extract entities if not provided
        if not entity_names:
            entities = extract_entities_from_query(query)
            entity_names = [e["name"] for e in entities] if entities else []
        
        # Auto-detect search type if needed
        if search_type == "auto":
            search_type = self._auto_detect_search_type(query)
        
        # Execute appropriate search
        if search_type == "entity":
            if entity_names:
                results = self.search_by_entity(entity_names[0], limit=limit)
            else:
                logger.warning("No entities found for entity search")
                results = []
        
        elif search_type == "relationship":
            if entity_names:
                results = self.search_relationships(
                    entity_names[0],
                    relationship_type=relationship_type,
                    limit=limit
                )
            else:
                logger.warning("No entities found for relationship search")
                results = []
        
        elif search_type == "path":
            if len(entity_names) >= 2:
                path = self.find_path(
                    entity_names[0],
                    entity_names[1],
                    max_depth=max_depth
                )
                results = [path] if path else []
            else:
                logger.warning("Path search requires at least 2 entities")
                results = []
        
        elif search_type == "content":
            if entity_names:
                results = self.get_related_content(entity_names[0], limit=limit)
            else:
                logger.warning("No entities found for content search")
                results = []
        
        else:
            # Default: comprehensive search across all entities
            all_results = []
            for entity in entity_names:
                entity_results = self.search_by_entity(entity, limit=limit)
                all_results.extend(entity_results)
            
            # Deduplicate by file_id
            seen = set()
            results = []
            for result in all_results:
                file_id = result.get("file_id")
                if file_id and file_id not in seen:
                    seen.add(file_id)
                    results.append(result)
                    if len(results) >= limit:
                        break
        
        return results
    
    def _auto_detect_search_type(self, query: str) -> str:
        """
        Auto-detect search type from query text.
        
        Args:
            query: Query string
        
        Returns:
            Detected search type
        """
        query_lower = query.lower()
        
        # Check for path-related keywords
        if any(word in query_lower for word in ["path", "connect", "link between", "route", "connection"]):
            return "path"
        
        # Check for relationship-related keywords
        if any(word in query_lower for word in ["relationship", "related", "connected", "associate", "link"]):
            return "relationship"
        
        # Check for content-related keywords
        if any(word in query_lower for word in ["document", "file", "content", "image", "audio", "about"]):
            return "content"
        
        # Default to entity search
        return "entity"

