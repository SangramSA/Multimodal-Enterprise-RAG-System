#!/usr/bin/env python3
"""Script to clear all data from Neo4j and Qdrant databases."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import List, Dict, Any
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "multimodal_rag")


def clear_neo4j():
    """Delete all nodes and relationships from Neo4j."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Delete all relationships first
            result = session.run("MATCH ()-[r]->() DELETE r RETURN count(r) as deleted")
            rels_deleted = result.single()["deleted"]
            
            # Delete all nodes
            result = session.run("MATCH (n) DELETE n RETURN count(n) as deleted")
            nodes_deleted = result.single()["deleted"]
            
            print(f"✅ Neo4j cleared: {nodes_deleted} nodes, {rels_deleted} relationships deleted")
            return True
    except Exception as e:
        print(f"❌ Failed to clear Neo4j: {e}")
        return False
    finally:
        driver.close()


def clear_qdrant():
    """Delete all points from Qdrant collection."""
    client = QdrantClient(url=QDRANT_URL)
    
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_exists = any(c.name == QDRANT_COLLECTION_NAME for c in collections.collections)
        
        if not collection_exists:
            print(f"⚠️  Collection '{QDRANT_COLLECTION_NAME}' does not exist in Qdrant")
            return True
        
        # Delete all points using scroll and delete
        # First, get all point IDs
        points_deleted = 0
        offset = None
        
        while True:
            result, next_offset = client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False
            )
            
            if not result:
                break
            
            point_ids = [point.id for point in result]
            if point_ids:
                client.delete(
                    collection_name=QDRANT_COLLECTION_NAME,
                    points_selector=point_ids
                )
                points_deleted += len(point_ids)
            
            if next_offset is None:
                break
            offset = next_offset
        
        print(f"✅ Qdrant cleared: {points_deleted} points deleted from '{QDRANT_COLLECTION_NAME}'")
        return True
    except Exception as e:
        print(f"❌ Failed to clear Qdrant: {e}")
        return False


def main():
    """Main function to clear both databases."""
    print("="*80)
    print("CLEARING DATABASES")
    print("="*80)
    print()
    
    # Confirm action
    response = input("⚠️  This will DELETE ALL DATA from Neo4j and Qdrant. Continue? (yes/no): ")
    if response.lower() != "yes":
        print("Cancelled.")
        return
    
    print()
    print("Clearing Neo4j...")
    neo4j_success = clear_neo4j()
    
    print()
    print("Clearing Qdrant...")
    qdrant_success = clear_qdrant()
    
    print()
    print("="*80)
    if neo4j_success and qdrant_success:
        print("✅ Both databases cleared successfully!")
    else:
        print("⚠️  Some operations may have failed. Check the output above.")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

