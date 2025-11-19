"""Script to check data consistency between Neo4j and Qdrant."""

from pathlib import Path
import sys
import os

# Add project root to path BEFORE any imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Avoid importing utils.logging to prevent circular import
# Use print statements for output instead

# Now import project modules
from graph.neo4j_client import Neo4jClient
from vector.qdrant_client import QdrantClientWrapper
from vector.vector_store import VectorStore
from vector.embedding_service import EmbeddingService


def get_all_content_nodes(neo4j_client: Neo4jClient) -> list:
    """Get all content nodes (Document/Image/Audio) from Neo4j."""
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
        results = neo4j_client.execute_query(query)
        return results
    except Exception as e:
        print(f"ERROR: Failed to query Neo4j: {e}")
        return []


def check_chunks_in_qdrant(vector_store: VectorStore, file_id: str) -> dict:
    """Check if chunks exist in Qdrant for a given file_id."""
    try:
        chunks = vector_store.get_chunks_by_file_id(file_id)
        return {
            "exists": len(chunks) > 0,
            "chunk_count": len(chunks),
            "chunks": chunks
        }
    except Exception as e:
        print(f"WARNING: Error checking Qdrant for {file_id}: {e}")
        return {
            "exists": False,
            "chunk_count": 0,
            "error": str(e)
        }


def check_consistency():
    """Check consistency between Neo4j and Qdrant."""
    print("Starting data consistency check between Neo4j and Qdrant...")
    
    # Initialize clients
    try:
        neo4j_client = Neo4jClient()
    except Exception as e:
        print(f"ERROR: Failed to connect to Neo4j: {e}")
        return None
    
    try:
        qdrant_client = QdrantClientWrapper()
        embedding_service = EmbeddingService()
        vector_store = VectorStore(qdrant_client, embedding_service)
    except Exception as e:
        print(f"ERROR: Failed to connect to Qdrant: {e}")
        return None
    
    # Get all content nodes from Neo4j
    print("Querying Neo4j for all content nodes...")
    content_nodes = get_all_content_nodes(neo4j_client)
    
    if not content_nodes:
        print("WARNING: No content nodes found in Neo4j")
        return
    
    print(f"Found {len(content_nodes)} content nodes in Neo4j")
    
    # Check each node
    results = {
        "total_files": len(content_nodes),
        "files_with_chunks": 0,
        "files_missing_chunks": 0,
        "missing_files": [],
        "total_chunks": 0
    }
    
    print("\n" + "="*80)
    print("DATA CONSISTENCY REPORT")
    print("="*80)
    print(f"\nTotal files in Neo4j: {len(content_nodes)}\n")
    
    for i, node in enumerate(content_nodes, 1):
        file_id = node.get("file_id")
        file_name = node.get("file_name", "Unknown")
        modality = node.get("modality", "unknown")
        label = node.get("label", "Document")
        
        print(f"[{i}/{len(content_nodes)}] Checking: {file_name} ({modality})")
        print(f"  File ID: {file_id}")
        
        # Check Qdrant
        qdrant_result = check_chunks_in_qdrant(vector_store, file_id)
        
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
                "label": label,
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
    
    # Check reverse: Are there orphaned chunks in Qdrant?
    print("\n" + "="*80)
    print("CHECKING FOR ORPHANED CHUNKS IN QDRANT...")
    print("="*80)
    
    # Get all file_ids from Neo4j
    neo4j_file_ids = {node.get("file_id") for node in content_nodes}
    
    # Note: We can't easily list all chunks in Qdrant without scanning,
    # but we can check if any chunks reference non-existent file_ids
    # This is a simplified check - full scan would require more complex querying
    
    print("Note: Full orphan detection would require scanning all Qdrant points.")
    print("This check verifies that Neo4j files have corresponding chunks.")
    
    return results


if __name__ == "__main__":
    try:
        results = check_consistency()
        if results:
            print("\n✅ Consistency check completed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

