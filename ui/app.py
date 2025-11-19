"""Streamlit UI for the multimodal RAG system."""

import streamlit as st
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import validate_config
from utils.logging import logger
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.query_pipeline import QueryPipeline
from agents.retrieval_agent import RetrievalAgent
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
        
        # Agent
        retrieval_agent = RetrievalAgent(hybrid_search)
        
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
        
        logger.success("System initialized successfully")
        return {
            "ingestion_pipeline": ingestion_pipeline,
            "query_pipeline": query_pipeline,
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


# Main UI
st.title("🔍 Multimodal Enterprise RAG System")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select Page", ["File Upload", "Query", "Graph Explorer", "Evaluation"])
    
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
    
    query_type = st.selectbox(
        "Query Type",
        ["factual_lookup", "visual_qa", "audio_qa", "reasoning", "summarization", "semantic_linkage"]
    )
    
    query = st.text_area("Enter your query", height=100)
    
    if st.button("Search"):
        if not query:
            st.warning("Please enter a query")
        else:
            with st.spinner("Processing query..."):
                try:
                    start_time = time.time()
                    response = query_pipeline.process(
                        query=query,
                        query_type=query_type,
                        limit=10
                    )
                    elapsed_time = time.time() - start_time
                    
                    # Display answer
                    st.subheader("Answer")
                    st.write(response.get("answer", "No answer generated"))
                    
                    # Display sources
                    st.subheader("Sources")
                    sources = response.get("sources", [])
                    for i, source in enumerate(sources, 1):
                        with st.expander(f"Source {i}: {source.get('chunk_id', 'Unknown')}"):
                            st.write(f"**Modality:** {source.get('modality', 'unknown')}")
                            st.write(f"**Score:** {source.get('score', 0):.3f}")
                            st.write(f"**Preview:** {source.get('content_preview', '')}")
                    
                    # Display metadata
                    with st.expander("Query Metadata"):
                        metadata = response.get("metadata", {})
                        st.json(metadata)
                        st.write(f"**Total Time:** {elapsed_time:.2f}s")
                        st.write(f"**Confidence:** {response.get('confidence', 0):.2f}")
                
                except Exception as e:
                    st.error(f"Query failed: {e}")
                    logger.error(f"Query error: {e}")


# Graph Explorer Page
elif page == "Graph Explorer":
    st.header("Knowledge Graph Explorer")
    st.write("Explore the knowledge graph structure")
    
    entity_name = st.text_input("Search for entity")
    
    if st.button("Search Graph"):
        if entity_name:
            try:
                graph_search = GraphSearch(system["neo4j_client"])
                results = graph_search.search_by_entity(entity_name, limit=10)
                
                if results:
                    st.subheader("Related Content")
                    for result in results:
                        st.write(f"**File:** {result.get('file_name')}")
                        st.write(f"**Modality:** {result.get('modality')}")
                        st.write("---")
                else:
                    st.info("No results found")
            except Exception as e:
                st.error(f"Graph search failed: {e}")


# Evaluation Page
elif page == "Evaluation":
    st.header("System Evaluation")
    st.write("Run evaluation tests on the system")
    
    if st.button("Run Evaluation"):
        with st.spinner("Running evaluation..."):
            try:
                from evals.test_suite import TestSuite
                test_suite = TestSuite()
                test_suite.build_test_suite()
                
                results = test_suite.evaluate(query_pipeline)
                
                st.subheader("Evaluation Results")
                st.json(results)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precision@5", f"{results.get('avg_precision_at_5', 0):.3f}")
                with col2:
                    st.metric("Recall@5", f"{results.get('avg_recall_at_5', 0):.3f}")
                with col3:
                    st.metric("F1 Score", f"{results.get('avg_f1', 0):.3f}")
                
                st.metric("Exact Match Rate", f"{results.get('exact_match_rate', 0):.3f}")
                st.metric("Avg Semantic Similarity", f"{results.get('avg_semantic_similarity', 0):.3f}")
                
                # DeepEval Metrics
                st.subheader("DeepEval Metrics")
                deepeval_col1, deepeval_col2, deepeval_col3 = st.columns(3)
                
                hallucination_score = results.get('avg_hallucination_score')
                if hallucination_score is not None:
                    hallucination_rate = results.get('hallucination_rate', 0)
                    with deepeval_col1:
                        st.metric("Hallucination Score", f"{hallucination_score:.3f}")
                        st.metric("Hallucination Rate", f"{hallucination_rate:.1%}")
                        st.caption("Lower is better (0 = no hallucinations)")
                else:
                    with deepeval_col1:
                        st.info("Hallucination: N/A (no context provided)")
                
                relevancy = results.get('avg_answer_relevancy')
                if relevancy is not None:
                    with deepeval_col2:
                        st.metric("Answer Relevancy", f"{relevancy:.3f}")
                        st.caption("Higher is better (0-1 scale)")
                else:
                    with deepeval_col2:
                        st.info("Answer Relevancy: N/A (no expected output)")
                
                faithfulness = results.get('avg_faithfulness')
                if faithfulness is not None:
                    with deepeval_col3:
                        st.metric("Faithfulness", f"{faithfulness:.3f}")
                        st.caption("Higher is better (0-1 scale)")
                else:
                    with deepeval_col3:
                        st.info("Faithfulness: N/A (no context provided)")
                
                latency = results.get('latency', {})
                st.subheader("Latency Metrics")
                st.write(f"**Mean:** {latency.get('mean', 0):.3f}s")
                st.write(f"**P50:** {latency.get('p50', 0):.3f}s")
                st.write(f"**P95:** {latency.get('p95', 0):.3f}s")
                st.write(f"**P99:** {latency.get('p99', 0):.3f}s")
            
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                logger.error(f"Evaluation error: {e}")

