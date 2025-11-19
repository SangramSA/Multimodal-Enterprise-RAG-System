# Multimodal Enterprise RAG System - Text Architecture Diagram

## System Architecture (Text-Based)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE LAYER                              │
│                         Streamlit UI (File Upload, Query, Graph Explorer)    │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │ Text         │  │ Image        │  │ Audio        │                       │
│  │ Processor    │  │ Processor    │  │ Processor    │                       │
│  │              │  │              │  │              │                       │
│  │ PDF, TXT     │  │ JPG, PNG     │  │ MP3, WAV     │                       │
│  │ LangChain    │  │ OCR + GPT-4V │  │ Whisper API  │                       │
│  │ Chunking     │  │ Captioning   │  │ Transcription│                       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                       │
│         │                  │                  │                               │
│         └──────────────────┴──────────────────┘                               │
│                              │                                                │
│                              ▼                                                │
│                    ┌─────────────────────┐                                    │
│                    │  Ingestion Pipeline │                                    │
│                    └──────────┬──────────┘                                    │
└───────────────────────────────┼─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROCESSING & EXTRACTION LAYER                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │ Entity          │  │ Relationship     │  │ Domain           │          │
│  │ Extractor       │  │ Extractor        │  │ Classifier       │          │
│  │                 │  │                  │  │                  │          │
│  │ GPT-4           │  │ Extract          │  │ GPT-4             │          │
│  │ Structured     │  │ Relationships    │  │ Domain Tags      │          │
│  │ Output          │  │                  │  │                  │          │
│  └────────┬────────┘  └────────┬─────────┘  └────────┬─────────┘          │
│           │                    │                     │                      │
│           └────────────────────┴─────────────────────┘                      │
│                                  │                                            │
│                                  ▼                                            │
│                        ┌──────────────────┐                                  │
│                        │ Schema Generator  │                                  │
│                        │ Dynamic Schema    │                                  │
│                        └─────────┬─────────┘                                  │
└──────────────────────────────────┼─────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│         STORAGE LAYER             │  │         STORAGE LAYER             │
│                                  │  │                                  │
│  ┌────────────────────────────┐ │  │  ┌────────────────────────────┐ │
│  │ Neo4j                      │ │  │  │ Qdrant                    │ │
│  │                            │ │  │  │                            │ │
│  │ Knowledge Graph:           │ │  │  │ Vector Database:           │ │
│  │ • Entities                 │ │  │  │ • Chunk Embeddings        │ │
│  │ • Relationships            │ │  │  │ • Metadata                │ │
│  │ • Content Nodes            │ │  │  │ • Domain Tags             │ │
│  │ • Domain Tags             │ │  │  │ • Modality Info            │ │
│  └────────────────────────────┘ │  │  └────────────────────────────┘ │
└──────────────────────────────────┘  └──────────────────────────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SEARCH LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │ Keyword      │  │ Vector       │  │ Graph        │                      │
│  │ Search       │  │ Search       │  │ Search       │                      │
│  │              │  │              │  │              │                      │
│  │ BM25         │  │ Semantic     │  │ Cypher       │                      │
│  │ Matching     │  │ Similarity   │  │ Queries      │                      │
│  │              │  │              │  │              │                      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                      │
│         │                  │                  │                              │
│         └──────────────────┴──────────────────┘                              │
│                              │                                                │
│                              ▼                                                │
│                    ┌─────────────────────┐                                  │
│                    │  Hybrid Search       │                                  │
│                    │  RRF Reranking      │                                  │
│                    └──────────┬───────────┘                                  │
└───────────────────────────────┼─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AGENT & QUERY PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │ Query            │  │ Retrieval        │  │ Query            │          │
│  │ Rewriter         │  │ Agent            │  │ Pipeline         │          │
│  │                  │  │                  │  │                  │          │
│  │ Triage &         │  │ LangChain       │  │ End-to-End        │          │
│  │ Expansion        │  │ Agent            │  │ Processing        │          │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘          │
│           │                      │                      │                     │
│           └──────────────────────┴──────────────────────┘                     │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLM SERVICES                                        │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ OpenAI API                                                           │  │
│  │                                                                      │  │
│  │ • GPT-4o (Text Generation, Entity Extraction, Domain Classification)  │  │
│  │ • GPT-4 Vision (Image Captioning, OCR)                             │  │
│  │ • Whisper (Audio Transcription)                                     │  │
│  │ • text-embedding-3-small (Vector Embeddings)                        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EVALUATION FRAMEWORK                                    │
│                                                                              │
│  ┌──────────────────┐              ┌──────────────────┐                    │
│  │ Test Suite       │              │ Metrics          │                    │
│  │                  │              │                  │                    │
│  │ • SQuAD v2       │              │ • Precision@K    │                    │
│  │ • DocVQA         │              │ • Recall@K      │                    │
│  │ • FLEURS         │              │ • F1 Score      │                    │
│  │ • DeepEval       │              │ • Hallucination │                    │
│  │                  │              │ • Relevancy     │                    │
│  └──────────────────┘              │ • Faithfulness  │                    │
│                                     └──────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘

## Data Flow Summary

### Ingestion Flow
1. User uploads file → Ingestion Pipeline
2. Route to appropriate processor (Text/Image/Audio)
3. Extract content and create chunks
4. Extract entities and relationships
5. Classify domain tags
6. Store in Neo4j (graph) and Qdrant (vectors)

### Query Flow
1. User submits query → Query Pipeline
2. Validate and rewrite query
3. Retrieval Agent orchestrates search
4. Hybrid Search combines results
5. Generate answer with GPT-4
6. Return response with citations

## Component Interactions

- **Solid lines (─)**: Direct data flow
- **Dashed lines (┈)**: API calls to external services
- **Arrows (→)**: Direction of data flow

