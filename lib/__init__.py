"""
Scalable Hybrid RAG+KG Library

This library provides components for building hybrid RAG + Knowledge Graph
systems that scale to large documents (350+ pages).

Key Components:
- ScalableGraphStore: Persistent graph with subgraph extraction
- ScalableHybridRAG: Hybrid query system combining RAG + KG
- EntityExtractor: LLM-enhanced entity extraction (Phase 2)
- IncrementalKGBuilder: Chunk-by-chunk KG building (Phase 3)
- DocumentClassifier: Auto-detect document type (narrative, technical, etc.)
- ChunkTypeDetector: Pattern-based per-chunk type detection (English, fast)
- LLMChunkClassifier: LLM-based per-chunk type detection (multi-language)

Usage:
    from lib import ScalableGraphStore, ScalableHybridRAG

    # Create graph store
    graph_store = ScalableGraphStore("./kg_store", "my_graph")
    graph_store.add_triple("Paris", "IS_CAPITAL_OF", "France")

    # Create hybrid system
    hybrid = ScalableHybridRAG(llm, vector_store, graph_store)
    result = hybrid.query("What is the capital of France?")
    print(result['response'])

    # Phase 2: Enhanced entity extraction
    from lib import EntityExtractor
    extractor = EntityExtractor(llm, graph_store.get_entities())
    entities = extractor.extract("Tell me about Paris")

    # Phase 3: Incremental KG building for large documents
    from lib import IncrementalKGBuilder, DocumentProcessor
    builder = IncrementalKGBuilder(llm, graph_store)
    stats = builder.build_from_file("large_document.txt")
    print(f"Detected type: {stats['doc_type']}")  # Auto-detected!

    # Document type classification
    from lib import DocumentClassifier, DocumentType
    classifier = DocumentClassifier(llm)
    doc_type = classifier.classify("Once upon a time...")
    print(doc_type)  # DocumentType.NARRATIVE

    # Hybrid mode with multi-language support (French, English, etc.)
    from lib import LLMChunkClassifier
    classifier = LLMChunkClassifier(llm)
    chunk_type, confidence = classifier.classify("Il était une fois...")
    print(chunk_type)  # DocumentType.NARRATIVE

    # Use IncrementalKGBuilder with LLM classifier (default, multi-language)
    builder = IncrementalKGBuilder(llm, graph_store,
                                   hybrid_mode=True,
                                   use_llm_classifier=True)
    stats = builder.build_from_file("document_francais.txt")
    print(stats['chunk_type_stats'])  # {'narrative': 45, 'technical': 8, ...}

    # Fast mode (English only, pattern-based)
    builder_fast = IncrementalKGBuilder(llm, graph_store,
                                        use_llm_classifier=False)
"""

from .graph_store import ScalableGraphStore
from .scalable_hybrid import ScalableHybridRAG
from .entity_extractor import EntityExtractor, EntityResolver
from .kg_builder import IncrementalKGBuilder, StreamingKGBuilder, DocumentProcessor
from .doc_classifier import DocumentClassifier, DocumentType, classify_document
from .chunk_detector import (
    ChunkTypeDetector, detect_chunk_type,
    LLMChunkClassifier, classify_chunk_llm,
    MultilingualChunkDetector, detect_chunk_multilingual
)
from .extraction_prompts import (
    get_extraction_prompt,
    get_inverse_relation,
    get_relationship_examples
)
from .hybrid_processor import HybridDocumentProcessor
from .viz3d import visualize_graph_3d, MultiPageGraph3D, Node3D, Edge3D

__all__ = [
    # Phase 1 & 4: Core scalable components
    'ScalableGraphStore',
    'ScalableHybridRAG',
    # Phase 2: Entity extraction
    'EntityExtractor',
    'EntityResolver',
    # Phase 3: Incremental KG building
    'IncrementalKGBuilder',
    'StreamingKGBuilder',
    'DocumentProcessor',
    # Document type classification
    'DocumentClassifier',
    'DocumentType',
    'classify_document',
    # Chunk type detection - English only (fast)
    'ChunkTypeDetector',
    'detect_chunk_type',
    # Chunk type detection - Multi-language pattern-based (French, English)
    'MultilingualChunkDetector',
    'detect_chunk_multilingual',
    # Chunk type detection - LLM-based (any language)
    'LLMChunkClassifier',
    'classify_chunk_llm',
    # Extraction utilities
    'get_extraction_prompt',
    'get_inverse_relation',
    'get_relationship_examples',
    # Multi-page hybrid processor
    'HybridDocumentProcessor',
    # 3D Visualization
    'visualize_graph_3d',
    'MultiPageGraph3D',
    'Node3D',
    'Edge3D',
]
__version__ = '3.5.0'
