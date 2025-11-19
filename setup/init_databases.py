"""Initialize Neo4j and Qdrant databases with required indexes and collections."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from loguru import logger
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# Import from centralized config
from utils.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    QDRANT_URL,
    EMBEDDING_DIMENSION
)


def init_neo4j():
    """Initialize Neo4j database with indexes and constraints."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        with driver.session() as session:
            # Create constraints for unique entities
            logger.info("Creating Neo4j constraints...")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (o:Organization) REQUIRE o.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Image) REQUIRE i.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Audio) REQUIRE a.id IS UNIQUE")
            
            # Create indexes for faster lookups
            logger.info("Creating Neo4j indexes...")
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.file_id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.domain_tags)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (o:Organization) ON (o.name)")
            
        driver.close()
        logger.success("Neo4j initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j: {e}")
        return False


def init_qdrant():
    """Initialize Qdrant database with collections."""
    try:
        client = QdrantClient(url=QDRANT_URL)
        
        collection_name = "multimodal_rag"
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)
        
        if not collection_exists:
            logger.info(f"Creating Qdrant collection: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            logger.success(f"Qdrant collection '{collection_name}' created successfully")
        else:
            logger.info(f"Qdrant collection '{collection_name}' already exists")
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {e}")
        return False


def main():
    """Main initialization function."""
    logger.info("Initializing databases...")
    
    neo4j_success = init_neo4j()
    qdrant_success = init_qdrant()
    
    if neo4j_success and qdrant_success:
        logger.success("All databases initialized successfully")
        return 0
    else:
        logger.error("Database initialization failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

