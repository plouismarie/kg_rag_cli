"""
Hybrid Document Processor

Process multi-page documents for combined RAG + KG systems.
Supports directories of page files (e.g., page0001.txt, page0002.txt).

This provides a unified interface for building both:
- Vector store (RAG): For semantic similarity search
- Knowledge graph (KG): For structured relationship queries

Usage:
    from lib import HybridDocumentProcessor

    processor = HybridDocumentProcessor(
        llm=llm,
        embeddings=embeddings,
        persist_path="./hybrid_store"
    )

    # Process multi-page document
    hybrid = processor.process_directory(
        directory="bookUlysseRag2/",
        collection_name="ulysses"
    )

    # Query the hybrid system
    result = hybrid.query("Who is Paris?")

    # Later: Load existing without reprocessing
    hybrid = processor.load_existing("ulysses")
"""
from pathlib import Path
from glob import glob
from typing import Dict, List, Optional, Callable
import time

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

from .graph_store import ScalableGraphStore
from .kg_builder import IncrementalKGBuilder
from .scalable_hybrid import ScalableHybridRAG


class HybridDocumentProcessor:
    """
    Process multi-page documents for hybrid RAG+KG.

    Builds both:
    - Vector store (RAG): For semantic similarity search
    - Knowledge graph (KG): For structured relationship queries

    This class provides a single entry point for processing directories
    of page files and creating a complete hybrid query system.

    Attributes:
        llm: LangChain LLM instance for KG extraction and queries
        embeddings: Embedding model for RAG vector store
        persist_path: Directory for persistent storage of both RAG and KG

    Usage:
        processor = HybridDocumentProcessor(
            llm=llm,
            embeddings=embeddings,
            persist_path="./hybrid_store"
        )

        # Process multi-page document
        hybrid = processor.process_directory(
            directory="bookUlysseRag2/",
            collection_name="ulysses"
        )

        # Query the hybrid system
        result = hybrid.query("Who is Paris?")
        print(result['response'])
        print(result['kg_context'])  # Knowledge graph relationships
        print(result['sources'])     # RAG source documents

        # Later: Load existing without reprocessing
        hybrid = processor.load_existing("ulysses")
    """

    def __init__(self, llm, embeddings,
                 persist_path: str = "./hybrid_store",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        Initialize the hybrid document processor.

        Args:
            llm: LangChain LLM instance for KG extraction and hybrid queries
            embeddings: Embedding model (HuggingFace or OpenAI) for vector store
            persist_path: Directory for persistent storage (default "./hybrid_store")
            chunk_size: Characters per RAG chunk (default 500)
            chunk_overlap: Overlap between chunks (default 50)
        """
        self.llm = llm
        self.embeddings = embeddings
        self.persist_path = Path(persist_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Ensure persist path exists
        self.persist_path.mkdir(parents=True, exist_ok=True)

    def process_directory(self,
                          directory: str,
                          collection_name: str,
                          file_pattern: str = "*.txt",
                          sort_files: bool = True,
                          progress_callback: Optional[Callable] = None,
                          force_rebuild: bool = False,
                          use_llm_classifier: bool = False,
                          verbose: bool = False,
                          batch_chunks: int = 4,
                          llm_timeout: int = 60,
                          pattern_only: bool = False,
                          verbose_mode: bool = False,
                          enable_entity_resolution: bool = True,
                          dedup_mode: str = 'standard',
                          debug: bool = False) -> ScalableHybridRAG:
        """
        Process a directory of page files and create hybrid RAG+KG system.

        This method:
        1. Reads all page files from the directory
        2. Creates RAG chunks with page metadata
        3. Builds vector store for semantic search
        4. Extracts knowledge graph triples from each page
        5. Returns a ScalableHybridRAG instance for querying

        Args:
            directory: Path to directory containing page files
            collection_name: Name for vector store collection and graph
            file_pattern: Glob pattern for files (default "*.txt")
            sort_files: Whether to sort files by name (default True)
            progress_callback: Optional callback(stage, current, total, stats)
                               stage: 'rag_indexing' or 'kg_building'
            force_rebuild: If True, rebuild even if exists (default False)
            use_llm_classifier: If True, use LLM for chunk classification.
                               If False (default), use pattern-based (French/English)
            verbose: If True, print per-chunk progress during KG building
            batch_chunks: Number of chunks to batch together per LLM call (default 4)
                         Higher values = fewer LLM calls, more stable
            llm_timeout: Timeout in seconds for LLM calls (default 60).
                        Falls back to pattern extraction if exceeded.
            pattern_only: If True, skip LLM calls entirely for KG building.
                         Uses only pattern-based extraction (fast, stable).
            verbose_mode: If True, use verbose extraction prompts that extract
                         more entities (descriptive phrases, concepts, etc.)
                         and bypass strict entity validation. (default False)
            enable_entity_resolution: If True, deduplicate entities during insertion.
                         Set False for verbose graphs with no merging. (default True)
            dedup_mode: Deduplication mode when entity_resolution is enabled:
                - 'standard': Full dedup (case + title + fuzzy matching)
                - 'light': Light dedup (case-insensitive + substring only)
                - 'none': No dedup (pass-through)
            debug: If True, print detailed timing info for each operation.

        Returns:
            ScalableHybridRAG instance ready for queries

        Raises:
            FileNotFoundError: If directory doesn't exist or no matching files

        Example:
            hybrid = processor.process_directory(
                directory="bookUlysseRag2/",
                collection_name="ulysses"
            )
            result = hybrid.query("Qui est Pâris?")
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Find all matching files
        pattern = str(directory / file_pattern)
        files = glob(pattern)

        if not files:
            raise FileNotFoundError(f"No files matching {file_pattern} in {directory}")

        if sort_files:
            files = sorted(files)

        print(f"[HybridProcessor] Found {len(files)} pages in {directory}")

        # =====================================================================
        # CLEANUP: Remove old index if force_rebuild is enabled
        # =====================================================================
        if force_rebuild and self.persist_path.exists():
            import shutil
            print(f"[HybridProcessor] Force rebuild enabled - removing old index at {self.persist_path}")
            try:
                shutil.rmtree(self.persist_path)
                print(f"[HybridProcessor] Old index removed successfully")
            except Exception as e:
                print(f"[HybridProcessor] Warning: Could not remove old index: {e}")

        start_time = time.time()
        stats = {
            'pages': len(files),
            'rag_chunks': 0,
            'kg_triples': 0,
            'rag_time': 0.0,
            'kg_time': 0.0,
            'total_time': 0.0
        }

        # =====================================================================
        # STEP 1: Build RAG Vector Store
        # =====================================================================
        print("[HybridProcessor] Building RAG vector store...")
        rag_start = time.time()

        # Read all pages and create documents with metadata
        all_docs = []
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )

        for page_num, filepath in enumerate(files):
            page_name = Path(filepath).name
            content = Path(filepath).read_text(encoding='utf-8')

            # Split page into chunks
            chunks = text_splitter.split_text(content)

            # Create documents with page metadata
            for chunk_num, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'source': str(filepath),
                        'page': page_name,
                        'page_num': page_num + 1,
                        'chunk_num': chunk_num
                    }
                )
                all_docs.append(doc)

            if progress_callback:
                progress_callback('rag_indexing', page_num + 1, len(files), stats)

        print(f"[HybridProcessor] Created {len(all_docs)} RAG chunks from {len(files)} pages")
        stats['rag_chunks'] = len(all_docs)

        # Create vector store
        chroma_path = str(self.persist_path / "chroma_db")
        vector_store = Chroma.from_documents(
            documents=all_docs,
            collection_name=collection_name,
            embedding=self.embeddings,
            persist_directory=chroma_path
        )

        stats['rag_time'] = time.time() - rag_start
        print(f"[HybridProcessor] RAG indexing complete in {stats['rag_time']:.1f}s")

        # =====================================================================
        # STEP 2: Build Knowledge Graph
        # =====================================================================
        print("[HybridProcessor] Building knowledge graph...")
        kg_start = time.time()

        # Create graph store
        graph_store = ScalableGraphStore(
            persist_path=str(self.persist_path / "kg_store"),
            graph_name=collection_name,
            enable_entity_resolution=enable_entity_resolution,
            dedup_mode=dedup_mode
        )

        # Create KG builder with pattern-based classification (French/English)
        kg_builder = IncrementalKGBuilder(
            llm=self.llm,
            graph_store=graph_store,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            hybrid_mode=True,
            use_llm_classifier=use_llm_classifier,
            llm_timeout=llm_timeout,
            pattern_only=pattern_only,
            verbose_mode=verbose_mode,
            debug=debug
        )

        # Process directory
        def kg_progress(current, total, kg_stats):
            if progress_callback:
                progress_callback('kg_building', current, total, kg_stats)

        kg_stats = kg_builder.build_from_directory(
            directory=str(directory),
            file_pattern=file_pattern,
            sort_files=sort_files,
            progress_callback=kg_progress,
            verbose=verbose,
            batch_chunks=batch_chunks
        )

        stats['kg_triples'] = kg_stats.get('total_triples', 0)
        stats['kg_entities'] = kg_stats.get('unique_entities', 0)
        stats['kg_time'] = time.time() - kg_start
        print(f"[HybridProcessor] KG building complete in {stats['kg_time']:.1f}s")

        # =====================================================================
        # STEP 3: Create Hybrid System
        # =====================================================================
        print("[HybridProcessor] Creating hybrid RAG+KG system...")

        hybrid = ScalableHybridRAG(
            llm=self.llm,
            vector_store=vector_store,
            graph_store=graph_store,
            collection_name=collection_name,  # Sanitized name for Chroma
            source_directory=str(directory)   # Original path for file loading
        )

        stats['total_time'] = time.time() - start_time
        print(f"[HybridProcessor] Complete! {stats['rag_chunks']} RAG chunks, "
              f"{stats['kg_triples']} KG triples in {stats['total_time']:.1f}s")

        # Store stats for later access
        self._last_stats = stats

        return hybrid

    def load_existing(self, collection_name: str, source_directory: str = None) -> ScalableHybridRAG:
        """
        Load existing hybrid system from persistent storage.

        Use this to reload a previously processed document without
        reprocessing. Both the vector store and knowledge graph
        are loaded from disk.

        Args:
            collection_name: Name used when processing the document
            source_directory: Original source directory (for page loading)

        Returns:
            ScalableHybridRAG instance ready for queries

        Raises:
            FileNotFoundError: If persistent storage doesn't exist

        Example:
            # First time: process document
            hybrid = processor.process_directory("book/", "mybook")

            # Later: load without reprocessing
            hybrid = processor.load_existing("mybook", source_directory="book/")
        """
        # Load vector store
        chroma_path = str(self.persist_path / "chroma_db")
        if not Path(chroma_path).exists():
            raise FileNotFoundError(f"No vector store found at {chroma_path}")

        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=chroma_path
        )

        # Load graph store
        graph_store = ScalableGraphStore(
            persist_path=str(self.persist_path / "kg_store"),
            graph_name=collection_name
        )

        print(f"[HybridProcessor] Loaded existing hybrid system: {collection_name}")
        print(f"[HybridProcessor] KG entities: {len(graph_store.get_entities())}")

        # Create hybrid system
        return ScalableHybridRAG(
            llm=self.llm,
            vector_store=vector_store,
            graph_store=graph_store,
            collection_name=collection_name,
            source_directory=source_directory  # Pass original path for file loading
        )

    def process_single_file(self,
                            filepath: str,
                            collection_name: str,
                            progress_callback: Optional[Callable] = None,
                            use_llm_classifier: bool = False) -> ScalableHybridRAG:
        """
        Process a single file and create hybrid RAG+KG system.

        Convenience method for single-file documents.

        Args:
            filepath: Path to the text file
            collection_name: Name for vector store collection and graph
            progress_callback: Optional progress callback
            use_llm_classifier: Use LLM for chunk classification

        Returns:
            ScalableHybridRAG instance ready for queries
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Create a temporary directory structure
        directory = filepath.parent
        file_pattern = filepath.name

        return self.process_directory(
            directory=str(directory),
            collection_name=collection_name,
            file_pattern=file_pattern,
            progress_callback=progress_callback,
            use_llm_classifier=use_llm_classifier
        )

    @property
    def last_stats(self) -> Optional[Dict]:
        """Return statistics from last processing run."""
        return getattr(self, '_last_stats', None)

    def get_storage_info(self) -> Dict:
        """
        Get information about persistent storage.

        Returns:
            Dict with storage paths and sizes
        """
        info = {
            'persist_path': str(self.persist_path),
            'chroma_path': str(self.persist_path / "chroma_db"),
            'kg_path': str(self.persist_path / "kg_store"),
            'chroma_exists': (self.persist_path / "chroma_db").exists(),
            'kg_exists': (self.persist_path / "kg_store").exists(),
        }

        # Calculate sizes if exist
        if info['chroma_exists']:
            chroma_size = sum(
                f.stat().st_size for f in (self.persist_path / "chroma_db").rglob('*') if f.is_file()
            )
            info['chroma_size_mb'] = chroma_size / (1024 * 1024)

        if info['kg_exists']:
            kg_size = sum(
                f.stat().st_size for f in (self.persist_path / "kg_store").rglob('*') if f.is_file()
            )
            info['kg_size_mb'] = kg_size / (1024 * 1024)

        return info
