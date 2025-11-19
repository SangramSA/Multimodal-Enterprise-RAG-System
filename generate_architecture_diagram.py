#!/usr/bin/env python3
"""
Generate architecture diagram for Multimodal Enterprise RAG System.
Creates a visual diagram using matplotlib and networkx.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(20, 16))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Color scheme
colors = {
    'ui': '#fce4ec',
    'ingestion': '#e1bee7',
    'processing': '#ce93d8',
    'storage': '#ba68c8',
    'search': '#ab47bc',
    'agent': '#9c27b0',
    'llm': '#8e24aa',
    'eval': '#7b1fa2'
}

# Define components with positions and sizes
components = {
    # UI Layer
    'UI': {'pos': (5, 11.5), 'size': (3, 0.6), 'color': colors['ui'], 'label': 'Streamlit UI\n(File Upload, Query)'},
    
    # Ingestion Layer
    'TextProc': {'pos': (1.5, 10), 'size': (1.2, 0.6), 'color': colors['ingestion'], 'label': 'Text Processor\nPDF, TXT'},
    'ImageProc': {'pos': (4.5, 10), 'size': (1.2, 0.6), 'color': colors['ingestion'], 'label': 'Image Processor\nJPG, PNG'},
    'AudioProc': {'pos': (7.5, 10), 'size': (1.2, 0.6), 'color': colors['ingestion'], 'label': 'Audio Processor\nMP3, WAV'},
    'IngestionPipeline': {'pos': (4.5, 9), 'size': (1.5, 0.5), 'color': colors['ingestion'], 'label': 'Ingestion Pipeline'},
    
    # Processing Layer
    'EntityExtractor': {'pos': (1.5, 7.5), 'size': (1.2, 0.6), 'color': colors['processing'], 'label': 'Entity Extractor\nGPT-4'},
    'RelExtractor': {'pos': (4.5, 7.5), 'size': (1.2, 0.6), 'color': colors['processing'], 'label': 'Relationship\nExtractor'},
    'DomainClassifier': {'pos': (7.5, 7.5), 'size': (1.2, 0.6), 'color': colors['processing'], 'label': 'Domain Classifier\nGPT-4'},
    'SchemaGen': {'pos': (4.5, 6.5), 'size': (1.5, 0.5), 'color': colors['processing'], 'label': 'Schema Generator'},
    
    # Storage Layer
    'Neo4j': {'pos': (1.5, 5), 'size': (1.5, 0.8), 'color': colors['storage'], 'label': 'Neo4j\nKnowledge Graph'},
    'Qdrant': {'pos': (7.5, 5), 'size': (1.5, 0.8), 'color': colors['storage'], 'label': 'Qdrant\nVector DB'},
    
    # Search Layer
    'KeywordSearch': {'pos': (1.5, 3.5), 'size': (1.2, 0.6), 'color': colors['search'], 'label': 'Keyword Search\nBM25'},
    'VectorSearch': {'pos': (4.5, 3.5), 'size': (1.2, 0.6), 'color': colors['search'], 'label': 'Vector Search\nSemantic'},
    'GraphSearch': {'pos': (7.5, 3.5), 'size': (1.2, 0.6), 'color': colors['search'], 'label': 'Graph Search\nCypher'},
    'HybridSearch': {'pos': (4.5, 2.5), 'size': (1.5, 0.5), 'color': colors['search'], 'label': 'Hybrid Search\nRRF'},
    
    # Agent Layer
    'QueryRewriter': {'pos': (1.5, 1.5), 'size': (1.2, 0.6), 'color': colors['agent'], 'label': 'Query Rewriter'},
    'RetrievalAgent': {'pos': (4.5, 1.5), 'size': (1.2, 0.6), 'color': colors['agent'], 'label': 'Retrieval Agent\nLangChain'},
    'QueryPipeline': {'pos': (7.5, 1.5), 'size': (1.2, 0.6), 'color': colors['agent'], 'label': 'Query Pipeline'},
    
    # LLM Services
    'OpenAI': {'pos': (9.5, 6), 'size': (0.4, 2), 'color': colors['llm'], 'label': 'OpenAI\nAPI'},
    
    # Evaluation
    'TestSuite': {'pos': (0.5, 1.5), 'size': (0.8, 0.6), 'color': colors['eval'], 'label': 'Test Suite\nDeepEval'},
}

# Draw components
for name, comp in components.items():
    x, y = comp['pos']
    w, h = comp['size']
    
    # Create rounded rectangle
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=comp['color'],
        edgecolor='black',
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(box)
    
    # Add label
    ax.text(x, y, comp['label'], 
            ha='center', va='center',
            fontsize=8, weight='bold',
            wrap=True)

# Define connections (from, to, style)
connections = [
    # UI to Ingestion
    ('UI', 'IngestionPipeline', 'solid'),
    
    # Ingestion to Processors
    ('IngestionPipeline', 'TextProc', 'solid'),
    ('IngestionPipeline', 'ImageProc', 'solid'),
    ('IngestionPipeline', 'AudioProc', 'solid'),
    
    # Processors to Entity Extractor
    ('TextProc', 'EntityExtractor', 'solid'),
    ('ImageProc', 'EntityExtractor', 'solid'),
    ('AudioProc', 'EntityExtractor', 'solid'),
    
    # Processing flow
    ('EntityExtractor', 'RelExtractor', 'solid'),
    ('EntityExtractor', 'DomainClassifier', 'solid'),
    ('RelExtractor', 'SchemaGen', 'solid'),
    ('DomainClassifier', 'SchemaGen', 'solid'),
    
    # To Storage
    ('SchemaGen', 'Neo4j', 'solid'),
    ('EntityExtractor', 'Qdrant', 'solid'),
    
    # Storage to Search
    ('Neo4j', 'GraphSearch', 'solid'),
    ('Qdrant', 'KeywordSearch', 'solid'),
    ('Qdrant', 'VectorSearch', 'solid'),
    
    # Search to Hybrid
    ('KeywordSearch', 'HybridSearch', 'solid'),
    ('VectorSearch', 'HybridSearch', 'solid'),
    ('GraphSearch', 'HybridSearch', 'solid'),
    
    # Agent flow
    ('HybridSearch', 'RetrievalAgent', 'solid'),
    ('RetrievalAgent', 'QueryRewriter', 'solid'),
    ('QueryRewriter', 'QueryPipeline', 'solid'),
    ('QueryPipeline', 'UI', 'solid'),
    
    # LLM connections (dashed)
    ('TextProc', 'OpenAI', 'dashed'),
    ('ImageProc', 'OpenAI', 'dashed'),
    ('AudioProc', 'OpenAI', 'dashed'),
    ('EntityExtractor', 'OpenAI', 'dashed'),
    ('DomainClassifier', 'OpenAI', 'dashed'),
    ('QueryPipeline', 'OpenAI', 'dashed'),
    ('VectorSearch', 'OpenAI', 'dashed'),
    
    # Evaluation
    ('TestSuite', 'QueryPipeline', 'dotted'),
]

def draw_arrow(from_comp, to_comp, style='solid'):
    """Draw arrow between components."""
    from_pos = components[from_comp]['pos']
    to_pos = components[to_comp]['pos']
    
    # Calculate arrow start and end points
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    dist = np.sqrt(dx**2 + dy**2)
    
    # Normalize and scale to component edges
    if dist > 0:
        scale = 0.3
        start_x = from_pos[0] + (dx / dist) * scale
        start_y = from_pos[1] + (dy / dist) * scale
        end_x = to_pos[0] - (dx / dist) * scale
        end_y = to_pos[1] - (dy / dist) * scale
        
        if style == 'solid':
            arrow = FancyArrowPatch(
                (start_x, start_y), (end_x, end_y),
                arrowstyle='->', mutation_scale=20,
                color='black', linewidth=1.5, zorder=1
            )
        elif style == 'dashed':
            arrow = FancyArrowPatch(
                (start_x, start_y), (end_x, end_y),
                arrowstyle='->', mutation_scale=20,
                color='gray', linewidth=1, linestyle='--', zorder=1
            )
        else:  # dotted
            arrow = FancyArrowPatch(
                (start_x, start_y), (end_x, end_y),
                arrowstyle='->', mutation_scale=20,
                color='blue', linewidth=1, linestyle=':', zorder=1
            )
        ax.add_patch(arrow)

# Draw all connections
for from_comp, to_comp, style in connections:
    draw_arrow(from_comp, to_comp, style)

# Add title
ax.text(5, 11.8, 'Multimodal Enterprise RAG System - Architecture Diagram',
        ha='center', va='top', fontsize=16, weight='bold')

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=colors['ui'], label='UI Layer'),
    mpatches.Patch(facecolor=colors['ingestion'], label='Ingestion'),
    mpatches.Patch(facecolor=colors['processing'], label='Processing'),
    mpatches.Patch(facecolor=colors['storage'], label='Storage'),
    mpatches.Patch(facecolor=colors['search'], label='Search'),
    mpatches.Patch(facecolor=colors['agent'], label='Agent'),
    mpatches.Patch(facecolor=colors['llm'], label='LLM Services'),
    mpatches.Patch(facecolor=colors['eval'], label='Evaluation'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

# Add connection style legend
ax.text(9.8, 0.2, 'Solid: Data Flow\nDashed: API Calls\nDotted: Evaluation',
        ha='right', va='bottom', fontsize=7, style='italic')

plt.tight_layout()
plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight')
print("✅ Architecture diagram saved as 'architecture_diagram.png'")
plt.close()

