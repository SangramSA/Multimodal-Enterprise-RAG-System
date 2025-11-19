"""Streamlit UI for the multimodal RAG system."""

import streamlit as st
import sys
from pathlib import Path
import time
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import validate_config
from utils.logging import logger
from utils.langsmith_config import configure_langsmith
from utils.telemetry import get_telemetry_collector
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.query_pipeline import QueryPipeline
# from pipeline.agentic_query_pipeline import AgenticQueryPipeline  # Commented out - using legacy or CrewAI directly
from agents.retrieval_agent import RetrievalAgent
from agents.retrieval_orchestration_agent import RetrievalOrchestrationAgent
from search.hybrid_search import HybridSearch
from search.keyword_search import KeywordSearch
from search.vector_search import VectorSearch
from search.graph_search import GraphSearch
from graph.neo4j_client import Neo4jClient
from vector.vector_store import VectorStore
from vector.qdrant_client import QdrantClientWrapper
from vector.embedding_service import EmbeddingService
from extraction.entity_extractor import EntityExtractor
from extraction.domain_classifier import DomainClassifier
from graph.graph_builder import GraphBuilder


# Page configuration
st.set_page_config(
    page_title="Multimodal Enterprise RAG",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
# if "agentic_pipeline" not in st.session_state:  # Commented out
#     st.session_state.agentic_pipeline = None
if "ingestion_pipeline" not in st.session_state:
    st.session_state.ingestion_pipeline = None


@st.cache_resource
def initialize_system():
    """Initialize the RAG system components."""
    try:
        # Validate config
        is_valid, error = validate_config()
        if not is_valid:
            st.error(f"Configuration error: {error}")
            st.stop()
        
        # Initialize components
        logger.info("Initializing system components...")
        
        # Database clients
        neo4j_client = Neo4jClient()
        qdrant_client = QdrantClientWrapper()
        
        # Services
        embedding_service = EmbeddingService()
        vector_store = VectorStore(qdrant_client, embedding_service)
        
        # Search components
        keyword_search = KeywordSearch(vector_store)
        vector_search = VectorSearch(vector_store)
        graph_search = GraphSearch(neo4j_client)
        hybrid_search = HybridSearch(keyword_search, vector_search, graph_search)
        
        # Agents
        retrieval_agent = RetrievalAgent(hybrid_search)
        retrieval_orchestration_agent = RetrievalOrchestrationAgent(
            hybrid_search, graph_search, keyword_search, vector_search
        )
        
        # Extraction and classification components
        entity_extractor = EntityExtractor()
        domain_classifier = DomainClassifier()
        graph_builder = GraphBuilder(neo4j_client)
        
        # Pipelines
        ingestion_pipeline = IngestionPipeline(
            entity_extractor=entity_extractor,
            domain_classifier=domain_classifier,
            graph_builder=graph_builder,
            vector_store=vector_store
        )
        query_pipeline = QueryPipeline(retrieval_agent)
        # Initialize with CrewAI disabled by default (can be enabled via UI)
        # agentic_query_pipeline = AgenticQueryPipeline(retrieval_orchestration_agent, use_crewai=False)  # Commented out
        crewai_orchestrator = None  # Will be initialized if CrewAI is selected
        
        # Configure LangSmith for observability (only if API key is valid)
        # Check if API key exists and is valid before enabling
        import os
        langsmith_api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
        if langsmith_api_key and len(langsmith_api_key) > 10:
            configure_langsmith(enabled=True)
        else:
            configure_langsmith(enabled=False)
            logger.debug("LangSmith tracing disabled (no valid API key)")
        
        logger.success("System initialized successfully")
        return {
            "ingestion_pipeline": ingestion_pipeline,
            "query_pipeline": query_pipeline,
            # "agentic_query_pipeline": agentic_query_pipeline,  # Commented out
            "retrieval_orchestration_agent": retrieval_orchestration_agent,  # For CrewAI direct access
            "crewai_orchestrator": crewai_orchestrator,
            "neo4j_client": neo4j_client,
            "vector_store": vector_store
        }
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        st.error(f"Failed to initialize system: {e}")
        st.stop()


# Initialize system
system = initialize_system()
ingestion_pipeline = system["ingestion_pipeline"]
query_pipeline = system["query_pipeline"]
# agentic_query_pipeline = system["agentic_query_pipeline"]  # Commented out - using legacy or CrewAI directly


# Main UI
st.title("🔍 Multimodal Enterprise RAG System")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select Page", ["File Upload", "Query", "Graph Explorer", "Telemetry", "Evaluation"])
    
    st.header("System Status")
    st.success("✅ System Ready")
    
    if st.button("Clear Cache"):
        st.cache_resource.clear()


# File Upload Page
if page == "File Upload":
    st.header("Upload Files")
    st.write("Upload PDF, TXT, JPG, PNG, or MP3 files for processing")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "jpg", "jpeg", "png", "mp3", "wav"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Files"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save uploaded file
                upload_dir = Path("uploads")
                upload_dir.mkdir(exist_ok=True)
                file_path = upload_dir / uploaded_file.name
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process and index file (complete pipeline)
                try:
                    result = ingestion_pipeline.process_and_index_file(file_path)
                    results.append(result)
                    
                    # Show detailed results
                    is_cached = result.get("cached", False)
                    vector_indexed = result.get("vector_indexed", 0)
                    graph_summary = result.get("graph_summary", {})
                    nodes_created = graph_summary.get("nodes_created", 0)
                    relationships_created = graph_summary.get("relationships_created", 0)
                    cross_modal_info = result.get("cross_modal_links", {})
                    cross_session_links = cross_modal_info.get("cross_session_links", 0) if cross_modal_info else 0
                    
                    if is_cached:
                        success_msg = (
                            f"✅ File already processed (using cached data) - {uploaded_file.name}\n"
                            f"   • {vector_indexed} chunks retrieved from cache\n"
                            f"   • Skipped OpenAI API calls (saved costs!)"
                        )
                        st.info(success_msg)
                    else:
                        success_msg = (
                            f"✅ Processed and indexed {uploaded_file.name}\n"
                            f"   • {vector_indexed} chunks indexed in Qdrant\n"
                            f"   • {nodes_created} nodes, {relationships_created} relationships in Neo4j"
                        )
                        if cross_session_links > 0:
                            success_msg += f"\n   • {cross_session_links} cross-session links created 🔗"
                        
                        st.success(success_msg)
                except Exception as e:
                    st.error(f"❌ Failed to process {uploaded_file.name}: {e}")
                    logger.error(f"Ingestion error: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("Processing complete!")
            
            # Show results summary
            if results:
                st.subheader("Processing Results")
                
                for result in results:
                    with st.expander(result.get("file_name", "Unknown")):
                        st.write(f"**File ID:** {result.get('file_id')}")
                        st.write(f"**Modality:** {result.get('modality')}")
                        st.write(f"**Chunks:** {len(result.get('chunks', []))}")
                        st.write(f"**Status:** {result.get('processing_status')}")
                        
                        # Show indexing results if available
                        if result.get('processing_status') == 'complete':
                            st.write("---")
                            st.write("**Indexing Results:**")
                            st.write(f"  • Vector Store: {result.get('vector_indexed', 0)} chunks indexed")
                            graph_summary = result.get('graph_summary', {})
                            st.write(f"  • Knowledge Graph: {graph_summary.get('nodes_created', 0)} nodes, "
                                   f"{graph_summary.get('relationships_created', 0)} relationships")
                            
                            # Show extraction summary
                            extraction_results = result.get('extraction_results', [])
                            total_entities = sum(len(er.get('entities', [])) for er in extraction_results)
                            total_relationships = sum(len(er.get('relationships', [])) for er in extraction_results)
                            st.write(f"  • Entities Extracted: {total_entities}")
                            st.write(f"  • Relationships Extracted: {total_relationships}")
                            
                            # Show cross-modal linking results
                            cross_modal_info = result.get('cross_modal_links', {})
                            if cross_modal_info:
                                total_links = cross_modal_info.get('total_links', 0)
                                same_session = cross_modal_info.get('same_session_links', 0)
                                cross_session = cross_modal_info.get('cross_session_links', 0)
                                if total_links > 0:
                                    st.write("---")
                                    st.write("**Cross-Modal Linking:**")
                                    st.write(f"  • Total Links: {total_links}")
                                    st.write(f"  • Same-Session: {same_session}")
                                    if cross_session > 0:
                                        st.write(f"  • Cross-Session: {cross_session} 🔗")


# Query Page
elif page == "Query":
    st.header("Query the Knowledge Base")
    
    # Pipeline selection
    # Legacy agentic pipeline checkbox removed (system now uses CrewAI or legacy pipeline)
    use_crewai = st.checkbox("Use CrewAI Orchestration (Experimental)", value=False)
    
    if use_crewai:
        st.info("🤖 CrewAI mode: Using CrewAI framework for multi-agent orchestration.")
    else:
        st.info("📚 Legacy mode: Using standard query pipeline.")
    
    
    
    query = st.text_area("Enter your query", height=100)
    
    if st.button("Search"):
        if not query:
            st.warning("Please enter a query")
        else:
            with st.spinner("Processing query..."):
                try:
                    start_time = time.time()
                    
                    if use_crewai:
                        # Use CrewAI orchestrator directly
                        from pipeline.crewai_orchestrator import CrewAIOrchestrator
                        
                        # Initialize CrewAI orchestrator if not already done
                        if system.get("crewai_orchestrator") is None:
                            try:
                                crewai_orchestrator = CrewAIOrchestrator(system["retrieval_orchestration_agent"])
                                system["crewai_orchestrator"] = crewai_orchestrator
                            except ImportError as e:
                                st.error(f"❌ CrewAI not available: {e}. Please install with: pip install crewai")
                                st.stop()
                        
                        response = system["crewai_orchestrator"].execute_pipeline(
                            query=query,
                            max_iterations=3
                        )
                    else:
                        # Use legacy query pipeline
                        response = query_pipeline.process(
                            query=query,
                            limit=10
                        )
                    
                    elapsed_time = time.time() - start_time
                    
                    # Display answer
                    st.subheader("Answer")
                    st.write(response.get("answer", "No answer generated"))
                    
                    # Display reasoning steps if available (agentic pipeline)
                    if response.get("reasoning_steps"):
                        st.subheader("🧠 Reasoning Process")
                        reasoning_steps = response.get("reasoning_steps", [])
                        
                        # Display summary with step count
                        st.info(f"📊 **{len(reasoning_steps)} reasoning step(s) completed**")
                        
                        # Display each reasoning step in full without truncation
                        for i, step in enumerate(reasoning_steps, 1):
                            # Create a preview for the expander title (first 80 chars)
                            preview = step[:80].replace('\n', ' ') + "..." if len(step) > 80 else step.replace('\n', ' ')
                            
                            with st.expander(f"**Step {i}:** {preview}", expanded=(i == 1)):
                                # Use markdown for better formatting - this ensures full text is displayed
                                st.markdown(step)
                                
                                # Show character count for very long steps
                                if len(step) > 1000:
                                    st.caption(f"📏 Step length: {len(step):,} characters")
                                
                                # For extremely long reasoning steps, also show in a scrollable text area
                                if len(step) > 2000:
                                    st.markdown("**Full Text (scrollable):**")
                                    st.text_area(
                                        f"Step {i} - Full Text",
                                        value=step,
                                        height=300,
                                        key=f"reasoning_step_{i}_full",
                                        label_visibility="collapsed"
                                    )
                        
                        # Show complete reasoning in a single view for easy reading
                        if len(reasoning_steps) > 1:
                            with st.expander("📋 Complete Reasoning Process (All Steps Combined)", expanded=False):
                                complete_reasoning = "\n\n" + "="*80 + "\n\n".join([
                                    f"## Step {i}\n\n{step}\n" 
                                    for i, step in enumerate(reasoning_steps, 1)
                                ])
                                
                                # Display as markdown
                                st.markdown(complete_reasoning)
                                
                                # Also provide as raw text for copying
                                st.markdown("**📋 Copy Full Reasoning (Raw Text):**")
                                st.code(complete_reasoning, language=None)
                                
                                # Show total length
                                total_length = sum(len(step) for step in reasoning_steps)
                                st.caption(f"📏 Total reasoning length: {total_length:,} characters across {len(reasoning_steps)} steps")
                    
                    # Display sources
                    st.subheader("Sources")
                    sources = response.get("sources", [])
                    if sources:
                        for i, source in enumerate(sources, 1):
                            with st.expander(f"Source {i}: {source.get('chunk_id', 'Unknown')}"):
                                st.write(f"**File:** {source.get('file_name', 'Unknown')}")
                                st.write(f"**Modality:** {source.get('modality', 'unknown')}")
                                st.write(f"**Score:** {source.get('score', 0):.3f}")
                                st.write(f"**Preview:** {source.get('content_preview', '')}")
                    else:
                        st.info("No sources found")
                    
                    # Display metadata
                    with st.expander("Query Metadata"):
                        metadata = response.get("metadata", {})
                        st.json(metadata)
                        st.write(f"**Total Time:** {elapsed_time:.2f}s")
                        st.write(f"**Confidence:** {response.get('confidence', 0):.2f}")
                        
                        if use_crewai:
                            st.write(f"**Iterations:** {metadata.get('iterations', 1)}")
                            st.write(f"**Methods Used:** {', '.join(metadata.get('methods_used', []))}")
                            if metadata.get('hallucination_score') is not None:
                                st.write(f"**Hallucination Score:** {metadata.get('hallucination_score', 0):.3f}")
                
                except Exception as e:
                    st.error(f"Query failed: {e}")
                    logger.error(f"Query error: {e}")


# Telemetry Dashboard Page
if page == "Telemetry" or st.session_state.get("show_telemetry", False):
    st.session_state.show_telemetry = False  # Reset flag
    
    st.header("📊 Agent Telemetry Dashboard")
    
    telemetry = get_telemetry_collector()
    stats = telemetry.get_agent_stats()
    recent_metrics = telemetry.get_recent_metrics(limit=50)
    
    if not stats:
        st.info("No telemetry data available yet. Run some queries to see metrics.")
    else:
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        total_ops = sum(s["total_operations"] for s in stats.values())
        total_success = sum(s["success_count"] for s in stats.values())
        total_errors = sum(s["error_count"] for s in stats.values())
        avg_duration = sum(s.get("avg_duration_ms", 0) for s in stats.values()) / len(stats) if stats else 0
        
        with col1:
            st.metric("Total Operations", total_ops)
        with col2:
            st.metric("Success Rate", f"{(total_success/total_ops*100):.1f}%" if total_ops > 0 else "0%")
        with col3:
            st.metric("Errors", total_errors)
        with col4:
            st.metric("Avg Duration", f"{avg_duration:.1f}ms")
        
        # Per-agent statistics
        st.subheader("Agent Performance")
        for agent_name, agent_data in stats.items():
            with st.expander(f"🤖 {agent_name}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Operations:** {agent_data['total_operations']}")
                    st.write(f"**Success:** {agent_data['success_count']}")
                    st.write(f"**Errors:** {agent_data['error_count']}")
                with col2:
                    st.write(f"**Avg Duration:** {agent_data.get('avg_duration_ms', 0):.2f}ms")
                    st.write(f"**Total Duration:** {agent_data['total_duration_ms']:.2f}ms")
                with col3:
                    success_rate = (agent_data['success_count'] / agent_data['total_operations'] * 100) if agent_data['total_operations'] > 0 else 0
                    st.write(f"**Success Rate:** {success_rate:.1f}%")
        
        # Recent metrics
        st.subheader("Recent Operations")
        if recent_metrics:
            for metric in recent_metrics[-20:]:
                status = "✅" if not metric.get("error") else "❌"
                with st.expander(f"{status} {metric['agent_name']} - {metric['operation']} ({metric.get('duration_ms', 0):.1f}ms)"):
                    st.write(f"**Timestamp:** {metric.get('timestamp', 'N/A')}")
                    st.write(f"**Duration:** {metric.get('duration_ms', 0):.2f}ms")
                    if metric.get("error"):
                        st.error(f"**Error:** {metric['error']}")
                    if metric.get("metadata"):
                        st.json(metric["metadata"])
        
        # Export button
        if st.button("📥 Export Telemetry"):
            export_path = telemetry.export_metrics()
            st.success(f"Telemetry exported to: {export_path}")
            st.code(str(export_path))


# Graph Explorer Page
elif page == "Graph Explorer":
    st.header("Knowledge Graph Explorer")
    st.write("Explore the knowledge graph structure interactively")
    
    # Graph visualization options
    col1, col2 = st.columns(2)
    with col1:
        view_mode = st.radio(
            "View Mode",
            ["Entity Search", "Full Graph", "Subgraph by Entity"],
            help="Entity Search: Find entities and their connections\nFull Graph: View entire graph (may be large)\nSubgraph: View neighborhood around an entity"
        )
    
    with col2:
        max_nodes = st.slider("Max Nodes", min_value=10, max_value=200, value=50, 
                             help="Maximum number of nodes to display (for performance)")
    
    # Entity search input
    entity_name = st.text_input("Search for entity (optional)", 
                                placeholder="e.g., Project Apollo, Aparavi, Normandy")
    
    # Graph options
    with st.expander("Graph Options"):
        col1, col2, col3 = st.columns(3)
        with col1:
            show_labels = st.checkbox("Show Node Labels", value=True)
            show_edges = st.checkbox("Show Edge Labels", value=True)
        with col2:
            physics_enabled = st.checkbox("Enable Physics", value=True, 
                                          help="Interactive physics simulation for layout")
            hierarchical = st.checkbox("Hierarchical Layout", value=False,
                                      help="Tree-like layout instead of force-directed")
        with col3:
            depth = st.slider("Traversal Depth", min_value=1, max_value=3, value=2,
                             help="How many relationship hops to traverse")
    
    if st.button("🔍 Explore Graph", type="primary"):
        try:
            neo4j_client = system["neo4j_client"]
            
            # Query graph data based on view mode
            # Note: Neo4j doesn't allow parameters in variable-length path patterns [*1..$depth]
            # So we need to interpolate depth directly into the query string
            if view_mode == "Entity Search" and entity_name:
                # Find entity and its connections with proper labels
                query = f"""
                MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower($entity_name)
                WITH n, labels(n) as n_labels LIMIT 1
                MATCH path = (n)-[*1..{depth}]-(connected)
                WITH n, n_labels, connected, labels(connected) as connected_labels, 
                     relationships(path) as rels, nodes(path) as path_nodes
                UNWIND path_nodes as node
                WITH DISTINCT n, n_labels, connected, connected_labels, rels, 
                     collect(DISTINCT {{node: node, labels: labels(node)}}) as all_nodes
                LIMIT $limit
                RETURN n as start, n_labels as start_labels, 
                       connected, connected_labels, rels, all_nodes
                """
                results = neo4j_client.execute_query(query, {
                    "entity_name": entity_name,
                    "limit": max_nodes
                })
                
                if not results:
                    # Fallback: find entity and direct relationships
                    query = """
                    MATCH (n)
                    WHERE toLower(n.name) CONTAINS toLower($entity_name)
                    WITH n, labels(n) as n_labels LIMIT 1
                    MATCH (n)-[r]-(m)
                    WITH n, n_labels, m, labels(m) as m_labels, r, type(r) as rel_type
                    LIMIT $limit
                    RETURN n as start, n_labels as start_labels,
                           m as connected, m_labels as connected_labels,
                           [{type: rel_type}] as rels
                    """
                    results = neo4j_client.execute_query(query, {
                        "entity_name": entity_name,
                        "limit": max_nodes
                    })
            
            elif view_mode == "Full Graph":
                # Get sample of all nodes and relationships with labels
                query = """
                MATCH (n)-[r]-(m)
                RETURN n, labels(n) as n_labels, 
                       r, type(r) as rel_type,
                       m, labels(m) as m_labels
                LIMIT $limit
                """
                results = neo4j_client.execute_query(query, {"limit": max_nodes})
            
            elif view_mode == "Subgraph by Entity" and entity_name:
                # Get subgraph around entity with all relationships
                query = f"""
                MATCH (center)
                WHERE toLower(center.name) CONTAINS toLower($entity_name)
                WITH center, labels(center) as center_labels LIMIT 1
                MATCH path = (center)-[*1..{depth}]-(neighbor)
                WITH center, center_labels, neighbor, labels(neighbor) as neighbor_labels,
                     relationships(path) as rels, nodes(path) as path_nodes
                UNWIND path_nodes as node
                WITH DISTINCT center, center_labels, neighbor, neighbor_labels, rels,
                     collect(DISTINCT {{node: node, labels: labels(node)}}) as all_nodes
                LIMIT $limit
                RETURN center as start, center_labels as start_labels,
                       neighbor as connected, neighbor_labels as connected_labels,
                       rels, all_nodes
                """
                results = neo4j_client.execute_query(query, {
                    "entity_name": entity_name,
                    "limit": max_nodes
                })
            else:
                st.warning("Please enter an entity name for this view mode")
                results = []
            
            if results:
                # Build graph visualization
                try:
                    from pyvis.network import Network
                    import networkx as nx
                    
                    # Create NetworkX graph
                    G = nx.Graph()
                    nodes_data = {}
                    edges_data = []
                    
                    # Process results and build graph
                    all_nodes_seen = set()
                    
                    for record in results:
                        # Handle different query result formats
                        start = record.get("start") or record.get("n")
                        connected = record.get("connected") or record.get("m")
                        rels = record.get("rels") or ([record.get("r")] if record.get("r") else [])
                        all_nodes = record.get("all_nodes", [])
                        start_labels = record.get("start_labels") or record.get("n_labels", [])
                        connected_labels = record.get("connected_labels") or record.get("m_labels", [])
                        rel_type = record.get("rel_type")
                        
                        # Helper function to add node
                        def add_node(node, node_labels_list=None):
                            if not node or not isinstance(node, dict):
                                return None
                                
                            node_id = node.get("id") or node.get("name") or str(hash(str(node)))
                            
                            if node_id in all_nodes_seen:
                                return node_id
                            all_nodes_seen.add(node_id)
                            
                            node_label = node.get("name") or node.get("file_name") or node_id
                            
                            # Determine node type from labels
                            if node_labels_list:
                                node_labels = node_labels_list
                            elif isinstance(node, dict) and "labels" in node:
                                node_labels = node["labels"]
                            elif "file_name" in node:
                                node_labels = ["Document"]
                            elif "name" in node and "file_name" not in node:
                                node_labels = ["Entity"]
                            else:
                                node_labels = ["Node"]
                            
                            node_type = node_labels[0] if node_labels else "Node"
                            
                            G.add_node(node_id, label=node_label, type=node_type)
                            nodes_data[node_id] = {
                                "label": node_label,
                                "type": node_type,
                                "properties": node
                            }
                            return node_id
                        
                        # Process all nodes from path if available
                        if all_nodes:
                            for node_entry in all_nodes:
                                if isinstance(node_entry, dict):
                                    node = node_entry.get("node")
                                    node_labels = node_entry.get("labels", [])
                                    add_node(node, node_labels)
                        
                        # Add start and connected nodes
                        start_id = None
                        connected_id = None
                        
                        if start:
                            start_id = add_node(start, start_labels)
                        
                        if connected:
                            connected_id = add_node(connected, connected_labels)
                        
                        # Add edges
                        if start_id and connected_id and start_id != connected_id:
                            # Get relationship type
                            edge_type = rel_type or "RELATED_TO"
                            if rels and len(rels) > 0:
                                for rel in rels:
                                    if rel and isinstance(rel, dict):
                                        edge_type = rel.get("type") or rel.get("rel_type") or edge_type
                                        break
                            
                            # Check if edge already exists
                            if not G.has_edge(start_id, connected_id):
                                G.add_edge(start_id, connected_id, label=edge_type)
                                edges_data.append({
                                    "source": start_id,
                                    "target": connected_id,
                                    "type": edge_type
                                })
                    
                    if len(G.nodes()) == 0:
                        st.warning("No graph data found. Try a different search term or view mode.")
                    else:
                        # Create pyvis network
                        net = Network(
                            height="600px",
                            width="100%",
                            bgcolor="#222222",
                            font_color="white",
                            directed=True
                        )
                        
                        # Configure physics
                        if physics_enabled:
                            net.set_options("""
                            {
                              "physics": {
                                "enabled": true,
                                "barnesHut": {
                                  "gravitationalConstant": -2000,
                                  "centralGravity": 0.1,
                                  "springLength": 200,
                                  "springConstant": 0.04,
                                  "damping": 0.09
                                }
                              }
                            }
                            """)
                        else:
                            net.set_options("""
                            {
                              "physics": {
                                "enabled": false
                              }
                            }
                            """)
                        
                        # Add nodes with colors based on type
                        node_colors = {
                            "Person": "#FF6B6B",
                            "Organization": "#4ECDC4",
                            "Location": "#95E1D3",
                            "Concept": "#F38181",
                            "Document": "#AA96DA",
                            "Image": "#FCBAD3",
                            "Audio": "#FFD93D",
                            "Entity": "#6BCB77",
                            "Node": "#C7CEEA"
                        }
                        
                        for node_id, data in nodes_data.items():
                            node_type = data["type"]
                            color = node_colors.get(node_type, "#C7CEEA")
                            label = data["label"] if show_labels else ""
                            
                            # Add tooltip with properties
                            title = f"{node_type}: {data['label']}"
                            if data.get("properties"):
                                props = data["properties"]
                                if isinstance(props, dict):
                                    props_str = "\\n".join([f"{k}: {v}" for k, v in list(props.items())[:5]])
                                    title += f"\\n\\nProperties:\\n{props_str}"
                            
                            net.add_node(
                                node_id,
                                label=label,
                                title=title,
                                color=color,
                                size=20 if node_type in ["Document", "Image", "Audio"] else 15
                            )
                        
                        # Add edges
                        for edge in edges_data:
                            edge_label = edge["type"] if show_edges else ""
                            net.add_edge(
                                edge["source"],
                                edge["target"],
                                label=edge_label,
                                title=edge["type"],
                                color="#848484",
                                width=2
                            )
                        
                        # Generate HTML
                        net.save_graph("temp_graph.html")
                        
                        # Display graph
                        st.subheader(f"Graph Visualization ({len(G.nodes())} nodes, {len(G.edges())} edges)")
                        
                        with open("temp_graph.html", "r", encoding="utf-8") as f:
                            graph_html = f.read()
                        
                        st.components.v1.html(graph_html, height=650, scrolling=True)
                        
                        # Show graph statistics
                        with st.expander("Graph Statistics"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Nodes", len(G.nodes()))
                            with col2:
                                st.metric("Edges", len(G.edges()))
                            with col3:
                                if len(G.nodes()) > 0:
                                    density = nx.density(G)
                                    st.metric("Density", f"{density:.3f}")
                            
                            # Node type distribution
                            node_types = {}
                            for node_id, data in nodes_data.items():
                                node_type = data["type"]
                                node_types[node_type] = node_types.get(node_type, 0) + 1
                            
                            if node_types:
                                st.write("**Node Types:**")
                                for node_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
                                    st.write(f"  • {node_type}: {count}")
                            
                            # Relationship types
                            rel_types = {}
                            for edge in edges_data:
                                rel_type = edge["type"]
                                rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
                            
                            if rel_types:
                                st.write("**Relationship Types:**")
                                for rel_type, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True):
                                    st.write(f"  • {rel_type}: {count}")
                        
                        # Clean up temp file
                        import os
                        if os.path.exists("temp_graph.html"):
                            os.remove("temp_graph.html")
                
                except ImportError:
                    st.error("Graph visualization requires 'pyvis' package. Install with: pip install pyvis")
                    st.info("Falling back to text view...")
                    
                    # Fallback to text view
                    st.subheader("Graph Data (Text View)")
                    for record in results[:20]:  # Limit to 20 for text view
                        with st.expander(f"Node: {record.get('start', {}).get('name', 'Unknown')}"):
                            st.json(record)
            
            else:
                st.info("No graph data found. Try a different search term or ensure data is ingested.")
        
        except Exception as e:
            st.error(f"Graph exploration failed: {e}")
            logger.error(f"Graph exploration error: {e}")
            import traceback
            st.code(traceback.format_exc())


# Evaluation Page
elif page == "Evaluation":
    st.header("System Evaluation")
    st.write("Run evaluation tests on the system")
    
    # Evaluation configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        test_cases = st.number_input("Test Cases", min_value=1, max_value=100, value=10, 
                                     help="Number of SQuAD v2 test cases to run")
    with col2:
        parallel_workers = st.number_input("Parallel Workers", min_value=1, max_value=10, value=3,
                                          help="Number of parallel workers for evaluation")
    with col3:
        skip_ingestion = st.checkbox("Skip Ingestion", value=False,
                                    help="Skip data ingestion (assumes data is already ingested)")
        use_automatic_upload = st.checkbox("Auto Upload to Confident AI", value=True,
                                           help="Use DeepEval's automatic Confident AI upload")
    
    st.info(f"💡 This will run: `run_evaluation.py --test-cases {test_cases} --parallel {parallel_workers} {'--skip-ingestion' if skip_ingestion else ''} {'--use-automatic-upload' if use_automatic_upload else ''}`")
    
    if st.button("🚀 Run Evaluation", type="primary"):
        with st.spinner("Running evaluation..."):
            try:
                # Import and call run_evaluation function with specified parameters
                from evals.run_evaluation import run_evaluation
                
                # Call run_evaluation with the specified parameters
                results = run_evaluation(
                    ingest_data=not skip_ingestion,
                    test_cases=test_cases,
                    max_workers=parallel_workers,
                    use_automatic_upload=use_automatic_upload
                )
                
                if not results:
                    st.error("Evaluation failed to complete. Check logs for details.")
                    st.stop()
                
                evaluation_results = results.get("evaluation_results", {})
                
                st.subheader("✅ Evaluation Results")
                
                # Summary metrics
                st.markdown("### Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Tests", evaluation_results.get('total_tests', 0))
                with col2:
                    st.metric("Test Cases", results.get('test_cases_count', 0))
                with col3:
                    if evaluation_results.get('confident_ai_report_url'):
                        st.markdown(f"[📊 View Confident AI Report]({evaluation_results.get('confident_ai_report_url')})")
                
                # DeepEval Metrics
                st.markdown("### DeepEval Metrics - Generator (Answer Quality)")
                deepeval_col1, deepeval_col2, deepeval_col3 = st.columns(3)
                
                hallucination_score = evaluation_results.get('avg_hallucination_score')
                if hallucination_score is not None:
                    hallucination_rate = evaluation_results.get('hallucination_rate', 0)
                    with deepeval_col1:
                        st.metric("Hallucination Score", f"{hallucination_score:.3f}")
                        st.metric("Hallucination Rate", f"{hallucination_rate:.1%}")
                        st.caption("Lower is better (0 = no hallucinations)")
                else:
                    with deepeval_col1:
                        st.info("Hallucination: N/A (no context provided)")
                
                relevancy = evaluation_results.get('avg_answer_relevancy')
                if relevancy is not None:
                    with deepeval_col2:
                        st.metric("Answer Relevancy", f"{relevancy:.3f}")
                        st.caption("Higher is better (0-1 scale)")
                else:
                    with deepeval_col2:
                        st.info("Answer Relevancy: N/A (no expected output)")
                
                faithfulness = evaluation_results.get('avg_faithfulness')
                if faithfulness is not None:
                    with deepeval_col3:
                        st.metric("Faithfulness", f"{faithfulness:.3f}")
                        st.caption("Higher is better (0-1 scale)")
                else:
                    with deepeval_col3:
                        st.info("Faithfulness: N/A (no context provided)")
                
                # Performance metrics
                latency = evaluation_results.get('latency', {})
                st.markdown("### Performance Metrics")
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                with perf_col1:
                    st.metric("Mean Latency", f"{latency.get('mean', 0):.3f}s")
                with perf_col2:
                    st.metric("P50 Latency", f"{latency.get('p50', 0):.3f}s")
                with perf_col3:
                    st.metric("P95 Latency", f"{latency.get('p95', 0):.3f}s")
                with perf_col4:
                    st.metric("P99 Latency", f"{latency.get('p99', 0):.3f}s")
                
                # Full results in expander
                with st.expander("📋 Full Evaluation Results (JSON)"):
                    st.json(results)
                
                # Show ingestion results if available
                ingestion_results = results.get("ingestion_results")
                if ingestion_results:
                    with st.expander("📥 Ingestion Results"):
                        st.json(ingestion_results)
            
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                logger.error(f"Evaluation error: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

