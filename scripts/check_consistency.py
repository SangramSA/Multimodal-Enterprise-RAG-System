#!/usr/bin/env python3
"""Script to check data consistency between Neo4j and Qdrant."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import directly to avoid circular import issues
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Import config directly
import os
from dotenv import load_dotenv
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "multimodal_rag")


def get_all_content_nodes() -> List[Dict[str, Any]]:
    """Get all content nodes (Document/Image/Audio) from Neo4j."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    query = """
    MATCH (content:Document|Image|Audio)
    RETURN content.id as file_id, 
           labels(content)[0] as label,
           content.file_name as file_name,
           content.modality as modality,
           content.upload_timestamp as upload_timestamp
    ORDER BY content.upload_timestamp DESC
    """
    
    try:
        with driver.session() as session:
            result = session.run(query)
            nodes = []
            for record in result:
                nodes.append({
                    "file_id": record["file_id"],
                    "label": record["label"],
                    "file_name": record["file_name"],
                    "modality": record["modality"],
                    "upload_timestamp": record["upload_timestamp"]
                })
            return nodes
    except Exception as e:
        print(f"ERROR: Failed to query Neo4j: {e}")
        return []
    finally:
        driver.close()


def check_chunks_in_qdrant(file_id: str) -> Dict[str, Any]:
    """Check if chunks exist in Qdrant for a given file_id."""
    client = QdrantClient(url=QDRANT_URL)
    
    try:
        # Create filter for file_id
        qdrant_filter = Filter(
            must=[
                FieldCondition(key="file_id", match=MatchValue(value=file_id))
            ]
        )
        
        # Use query_points instead of deprecated search method
        # We need to use scroll to get all points with the filter
        try:
            results, _ = client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                scroll_filter=qdrant_filter,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
        except Exception:
            # Fallback to search if scroll doesn't work
            dummy_vector = [0.0] * 1536  # Standard embedding dimension
            results = client.search(
                collection_name=QDRANT_COLLECTION_NAME,
                query_vector=dummy_vector,
                limit=1000,
                query_filter=qdrant_filter,
                score_threshold=0.0
            )
        
        chunks = []
        for result in results:
            # Handle both scroll and search result formats
            if hasattr(result, 'payload'):
                payload = result.payload
            elif isinstance(result, dict):
                payload = result.get("payload", {})
            else:
                payload = result if isinstance(result, dict) else {}
            
            content = payload.get("content", "")
            content_preview = content[:50] + "..." if len(content) > 50 else content
            
            chunks.append({
                "chunk_id": payload.get("chunk_id"),
                "chunk_index": payload.get("chunk_index", 0),
                "content": content_preview
            })
        
        return {
            "exists": len(chunks) > 0,
            "chunk_count": len(chunks),
            "chunks": chunks
        }
    except Exception as e:
        return {
            "exists": False,
            "chunk_count": 0,
            "error": str(e)
        }


def main():
    """Main function to check consistency."""
    print("="*80)
    print("DATA CONSISTENCY CHECK: Neo4j vs Qdrant")
    print("="*80)
    print()
    
    # Get all content nodes from Neo4j
    print("Querying Neo4j for all content nodes...")
    content_nodes = get_all_content_nodes()
    
    if not content_nodes:
        print("WARNING: No content nodes found in Neo4j")
        return
    
    print(f"Found {len(content_nodes)} content nodes in Neo4j\n")
    
    # Check each node
    results = {
        "total_files": len(content_nodes),
        "files_with_chunks": 0,
        "files_missing_chunks": 0,
        "missing_files": [],
        "total_chunks": 0
    }
    
    print("-" * 80)
    for i, node in enumerate(content_nodes, 1):
        file_id = node.get("file_id")
        file_name = node.get("file_name", "Unknown")
        modality = node.get("modality", "unknown")
        
        print(f"[{i}/{len(content_nodes)}] {file_name} ({modality})")
        print(f"  File ID: {file_id}")
        
        # Check Qdrant
        qdrant_result = check_chunks_in_qdrant(file_id)
        
        if qdrant_result["exists"]:
            chunk_count = qdrant_result["chunk_count"]
            results["files_with_chunks"] += 1
            results["total_chunks"] += chunk_count
            print(f"  ✅ Found {chunk_count} chunk(s) in Qdrant")
        else:
            results["files_missing_chunks"] += 1
            results["missing_files"].append({
                "file_id": file_id,
                "file_name": file_name,
                "modality": modality,
                "error": qdrant_result.get("error")
            })
            print(f"  ❌ No chunks found in Qdrant")
            if qdrant_result.get("error"):
                print(f"     Error: {qdrant_result['error']}")
        
        print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total files in Neo4j: {results['total_files']}")
    print(f"Files with chunks in Qdrant: {results['files_with_chunks']} ✅")
    print(f"Files missing chunks in Qdrant: {results['files_missing_chunks']} ❌")
    print(f"Total chunks in Qdrant: {results['total_chunks']}")
    
    if results["missing_files"]:
        print("\n" + "="*80)
        print("FILES MISSING CHUNKS IN QDRANT:")
        print("="*80)
        for missing in results["missing_files"]:
            print(f"  - {missing['file_name']} ({missing['modality']})")
            print(f"    File ID: {missing['file_id']}")
            if missing.get("error"):
                print(f"    Error: {missing['error']}")
            print()
    else:
        print("\n✅ All files in Neo4j have corresponding chunks in Qdrant!")
    
    print("\n" + "="*80)


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

