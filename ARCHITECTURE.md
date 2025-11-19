# Multimodal Enterprise RAG System - Architecture Diagram

## System Architecture Overview

```mermaid
graph TB
    subgraph "User Interface Layer"
        UI[Streamlit UI<br/>File Upload, Query, Graph Explorer]
    end

    subgraph "Data Ingestion Pipeline"
        IP[Ingestion Pipeline]
        TP[Text Processor<br/>PDF, TXT]
        ImP[Image Processor<br/>JPG, PNG<br/>OCR + GPT-4V]
        AP[Audio Processor<br/>MP3, WAV<br/>Whisper API]
    end

    subgraph "Processing & Extraction"
        EE[Entity Extractor<br/>GPT-4 Structured Output]
        RE[Relationship Extractor]
        DC[Domain Classifier<br/>GPT-4]
        SG[Schema Generator<br/>Dynamic Schema]
    end

    subgraph "Storage Layer"
        NEO4J[(Neo4j<br/>Knowledge Graph<br/>Entities & Relationships)]
        QDRANT[(Qdrant<br/>Vector Database<br/>Semantic Search)]
    end

    subgraph "Search Layer"
        KS[Keyword Search<br/>BM25]
        VS[Vector Search<br/>Semantic Similarity]
        GS[Graph Search<br/>Cypher Queries]
        HS[Hybrid Search<br/>RRF Reranking]
    end

    subgraph "Agent & Query Pipeline"
        QR[Query Rewriter<br/>Triage & Expansion]
        RA[Retrieval Agent<br/>LangChain Agent]
        QP[Query Pipeline<br/>End-to-End Processing]
    end

    subgraph "LLM Services"
        OPENAI[OpenAI API<br/>GPT-4o, GPT-4V, Whisper<br/>Embeddings]
    end

    subgraph "Evaluation Framework"
        TS[Test Suite<br/>DeepEval]
        METRICS[Custom Metrics<br/>Precision, Recall, F1]
    end

    %% Data Flow - Ingestion
    UI -->|Upload Files| IP
    IP --> TP
    IP --> ImP
    IP --> AP
    
    TP -->|Chunks| EE
    ImP -->|Chunks| EE
    AP -->|Chunks| EE
    
    EE -->|Entities| RE
    EE -->|Text| DC
    RE -->|Relationships| SG
    DC -->|Domain Tags| SG
    
    SG -->|Graph Data| NEO4J
    EE -->|Chunks + Metadata| QDRANT
    
    %% Query Flow
    UI -->|Query| QP
    QP -->|Validate| QR
    QR -->|Rewritten Query| RA
    RA -->|Orchestrate| HS
    
    HS --> KS
    HS --> VS
    HS --> GS
    
    KS --> QDRANT
    VS --> QDRANT
    GS --> NEO4J
    
    HS -->|Retrieved Context| QP
    QP -->|Generate Answer| OPENAI
    OPENAI -->|Answer| QP
    QP -->|Response| UI
    
    %% LLM Services
    TP -.->|OCR/Caption| OPENAI
    ImP -.->|Vision API| OPENAI
    AP -.->|Transcription| OPENAI
    EE -.->|Structured Output| OPENAI
    DC -.->|Classification| OPENAI
    QP -.->|Answer Generation| OPENAI
    VS -.->|Embeddings| OPENAI
    
    %% Evaluation
    TS -->|Test Cases| QP
    QP -->|Results| METRICS
    METRICS -->|Evaluation Report| TS
    
    %% Styling
    classDef storage fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef search fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef llm fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef ui fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    
    class NEO4J,QDRANT storage
    class EE,RE,DC,SG,TP,ImP,AP processing
    class KS,VS,GS,HS search
    class OPENAI llm
    class UI ui
```

## Component Details

### 1. Data Ingestion Pipeline
- **Text Processor**: Handles PDF and TXT files, uses LangChain's RecursiveCharacterTextSplitter for intelligent chunking
- **Image Processor**: OCR (Tesseract) + GPT-4 Vision for captioning and text extraction
- **Audio Processor**: OpenAI Whisper API for transcription

### 2. Processing & Extraction
- **Entity Extractor**: GPT-4 with structured output to extract entities (Person, Organization, Location, Concept, Date)
- **Relationship Extractor**: Identifies relationships between entities
- **Domain Classifier**: GPT-4 classifies content into domains (finance, legal, technical, medical, etc.)
- **Schema Generator**: Dynamically generates graph schema based on extracted entities

### 3. Storage Layer
- **Neo4j**: Knowledge graph storing entities, relationships, and content nodes with domain tags
- **Qdrant**: Vector database storing chunk embeddings with rich metadata (modality, domain tags, file info)

### 4. Search Layer
- **Keyword Search**: BM25-based keyword matching with metadata filtering
- **Vector Search**: Semantic similarity search using OpenAI embeddings
- **Graph Search**: Cypher queries for entity traversal and relationship discovery
- **Hybrid Search**: Combines all three methods using Reciprocal Rank Fusion (RRF)

### 5. Agent & Query Pipeline
- **Query Rewriter**: Classifies query type and expands queries with synonyms
- **Retrieval Agent**: LangChain agent that orchestrates search tools dynamically
- **Query Pipeline**: End-to-end flow from validation → retrieval → generation → post-processing

### 6. LLM Services
- **OpenAI API**: 
  - GPT-4o for text generation and structured extraction
  - GPT-4 Vision for image understanding
  - Whisper for audio transcription
  - text-embedding-3-small for vector embeddings

### 7. Evaluation Framework
- **Test Suite**: DeepEval-based evaluation with test cases from SQuAD v2, DocVQA, and FLEURS
- **Metrics**: Custom metrics (Precision@K, Recall@K, F1) + DeepEval metrics (Hallucination, Relevancy, Faithfulness)

## Data Flow

### Ingestion Flow
1. User uploads file (PDF, TXT, JPG, PNG, MP3) via Streamlit UI
2. Ingestion Pipeline routes to appropriate processor
3. Processor extracts content and creates chunks
4. Entity Extractor extracts entities and relationships
5. Domain Classifier assigns domain tags
6. Graph Builder creates nodes and relationships in Neo4j
7. Vector Store indexes chunks with embeddings in Qdrant

### Query Flow
1. User submits query via Streamlit UI
2. Query Pipeline validates input
3. Query Rewriter classifies and expands query
4. Retrieval Agent orchestrates search tools
5. Hybrid Search combines results from Keyword, Vector, and Graph search
6. Retrieved context is passed to GPT-4 for answer generation
7. Response with citations is returned to UI

## Technology Stack

- **UI**: Streamlit
- **LLM**: OpenAI (GPT-4o, GPT-4 Vision, Whisper)
- **Graph DB**: Neo4j 5.15.0
- **Vector DB**: Qdrant v1.7.0
- **Orchestration**: LangChain 1.0
- **Evaluation**: DeepEval
- **Deployment**: Docker Compose (local development)

## Key Features

- ✅ Multimodal ingestion (text, image, audio)
- ✅ Intelligent chunking with LangChain
- ✅ LLM-based entity and relationship extraction
- ✅ Automatic domain classification
- ✅ Hybrid search (keyword + vector + graph)
- ✅ Agent-based retrieval orchestration
- ✅ Comprehensive error handling
- ✅ Evaluation-first approach with DeepEval

