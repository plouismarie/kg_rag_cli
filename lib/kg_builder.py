"""
Phase 3: Incremental Knowledge Graph Builder

Processes large documents chunk-by-chunk instead of all at once.
Key features:
- Streaming triple extraction (doesn't load entire document)
- Progress tracking for long documents
- Batch persistence to avoid data loss
- Configurable chunk sizes for memory management
- Document-type-aware extraction (narrative, technical, conversational, scientific)
"""
from typing import List, Dict, Set, Tuple, Optional, Generator, Callable
from pathlib import Path
from glob import glob
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import threading
import time
import re

from .graph_store import ScalableGraphStore
from .doc_classifier import DocumentClassifier, DocumentType
from .chunk_detector import ChunkTypeDetector, LLMChunkClassifier, MultilingualChunkDetector
from .extraction_prompts import (
    get_extraction_prompt,
    get_inverse_relation,
    get_inverse_mapping,
    EXTRACTION_PROMPTS
)


class IncrementalKGBuilder:
    """
    Builds Knowledge Graphs incrementally from large documents.

    Unlike the original approach that processes entire documents at once,
    this builder:
    - Processes documents chunk-by-chunk (streaming)
    - Extracts triples from each chunk using LLM
    - Adds to graph store incrementally with progress tracking
    - Saves periodically to prevent data loss

    Usage:
        builder = IncrementalKGBuilder(llm, graph_store)
        stats = builder.build_from_file("large_document.txt")
        print(f"Extracted {stats['total_triples']} triples")
    """

    def __init__(self, llm, graph_store: ScalableGraphStore,
                 chunk_size: int = 800,
                 chunk_overlap: int = 100,
                 save_interval: int = 10,
                 auto_detect_type: bool = True,
                 doc_type: Optional[DocumentType] = None,
                 hybrid_mode: bool = True,
                 use_llm_classifier: bool = False,
                 llm_timeout: int = 120,
                 batch_size: int = 4,
                 pattern_only: bool = False,
                 verbose_mode: bool = False,
                 debug: bool = False):
        """
        Initialize the incremental KG builder.

        Args:
            llm: LangChain LLM instance for triple extraction
            graph_store: ScalableGraphStore to add triples to
            chunk_size: Characters per chunk (default 800, optimized for small models)
                       Reduced from 2000 to prevent timeouts with small LLMs
            chunk_overlap: Overlap between chunks to avoid boundary issues
            save_interval: Save to disk every N chunks
            auto_detect_type: Whether to auto-detect document type (default True)
            doc_type: Force a specific document type (overrides auto-detect)
            hybrid_mode: Enable per-chunk type detection for mixed documents (default True)
            use_llm_classifier: If True, use LLM-based classification (multi-language).
                               If False (default), use pattern-based (faster, safer).
            llm_timeout: Timeout in seconds for LLM calls (default 120). If exceeded,
                        falls back to pattern-based extraction.
                        Increased from 60s to handle complex prompts with small models.
            batch_size: Number of chunks to process per LLM call (default 4).
                       Higher values = fewer LLM calls = faster but more memory.
                       Use 1 for no batching, 4-8 recommended for GPU.
            pattern_only: If True, skip LLM entirely and use only pattern extraction.
                         Fast and stable, but less accurate.
            verbose_mode: If True, use verbose extraction prompts and bypass strict
                         entity validation. Extracts more entities including descriptive
                         phrases, concepts, and longer entities. (default False)
            debug: If True, print detailed timing info for each operation.
        """
        self.llm = llm
        self.graph_store = graph_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.save_interval = save_interval
        self.llm_timeout = llm_timeout
        self.batch_size = batch_size
        self.pattern_only = pattern_only
        self.verbose_mode = verbose_mode
        self.debug = debug

        # Validate batch_size
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        # Document type detection
        self.auto_detect_type = auto_detect_type
        self.classifier = DocumentClassifier(llm, timeout=llm_timeout, debug=debug) if auto_detect_type else None
        self.doc_type = doc_type  # Will be set during build if auto_detect

        # Hybrid mode: per-chunk type detection
        self.hybrid_mode = hybrid_mode
        self.use_llm_classifier = use_llm_classifier
        if hybrid_mode:
            if use_llm_classifier:
                # LLM-based: any language, most accurate
                self.chunk_classifier = LLMChunkClassifier(llm, use_cache=True, timeout=llm_timeout, debug=debug)
            else:
                # Pattern-based with auto language detection (French, English)
                self.chunk_classifier = MultilingualChunkDetector()
        else:
            self.chunk_classifier = None
        self.chunk_type_stats = {}  # Track chunk types for stats

        # Statistics
        self.stats = {
            'chunks_processed': 0,
            'total_triples': 0,
            'unique_entities': 0,
            'errors': 0,
            'processing_time': 0.0,
            'doc_type': None
        }

    def build_from_file(self, filepath: str,
                        progress_callback: Optional[Callable] = None,
                        add_inverse: bool = True) -> Dict:
        """
        Build KG from a text file incrementally.

        Args:
            filepath: Path to the text file
            progress_callback: Optional callback(chunk_num, total_chunks, stats)
            add_inverse: Whether to add inverse relationships

        Returns:
            Dict with extraction statistics
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        content = filepath.read_text(encoding='utf-8')
        return self.build_from_text(content, progress_callback, add_inverse)

    def build_from_directory(self, directory: str,
                             file_pattern: str = "*.txt",
                             sort_files: bool = True,
                             progress_callback: Optional[Callable] = None,
                             add_inverse: bool = True,
                             verbose: bool = False,
                             batch_chunks: int = 1) -> Dict:
        """
        Build KG from a directory of page files incrementally.

        Processes each file in sequence, maintaining context across pages.
        Useful for multi-page documents like books where files are numbered
        (e.g., page0001.txt, page0002.txt, etc.)

        Args:
            directory: Path to directory containing page files
            file_pattern: Glob pattern for files (default "*.txt")
            sort_files: Whether to sort files by name (default True)
            progress_callback: Optional callback(page_num, total_pages, stats)
            add_inverse: Whether to add inverse relationships
            verbose: Whether to print per-chunk progress (default False)
            batch_chunks: Number of chunks to batch together for LLM call (default 1)
                         Higher values = fewer LLM calls but more text per call

        Returns:
            Dict with extraction statistics including per-page breakdown
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

        print(f"[KGBuilder] Found {len(files)} pages in {directory}")

        start_time = time.time()
        self._reset_stats()

        # Track entity deduplication stats (if enabled)
        dedup_stats_start = None
        if self.graph_store.enable_entity_resolution:
            dedup_stats_start = self.graph_store.get_deduplication_stats()

        # Track per-page statistics
        page_stats = []
        total_triples_before = 0

        # Cache for inverse relations (shared across all pages)
        inverse_cache: Dict[str, str] = {}

        # Auto-detect document type from first page (sample)
        if self.doc_type is None:
            if self.auto_detect_type and self.classifier:
                first_content = Path(files[0]).read_text(encoding='utf-8')[:2000]
                self.doc_type = self.classifier.classify(first_content)
                print(f"[KGBuilder] Detected document type: {self.doc_type.value}")
            else:
                self.doc_type = DocumentType.NARRATIVE  # Default for multi-page docs
                print(f"[KGBuilder] Using default document type: {self.doc_type.value}")

        self.stats['doc_type'] = self.doc_type.value

        for page_num, filepath in enumerate(files):
            page_start = time.time()
            page_name = Path(filepath).name
            total_triples_before = self.stats['total_triples']

            print(f"[KGBuilder] Processing page {page_num + 1}/{len(files)}: {page_name}")

            try:
                # Read page content
                content = Path(filepath).read_text(encoding='utf-8')

                # Split into chunks
                chunks = list(self._chunk_text(content))

                # Batch chunks together for fewer LLM calls
                num_batches = (len(chunks) + batch_chunks - 1) // batch_chunks

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_chunks
                    end_idx = min(start_idx + batch_chunks, len(chunks))
                    batch = chunks[start_idx:end_idx]

                    # Combine chunks in batch
                    combined_text = "\n\n".join(batch)

                    try:
                        if verbose:
                            if batch_chunks > 1:
                                print(f"  [KGBuilder] Batch {batch_idx + 1}/{num_batches} (chunks {start_idx + 1}-{end_idx}/{len(chunks)}) - extracting...", end="", flush=True)
                            else:
                                print(f"  [KGBuilder] Chunk {start_idx + 1}/{len(chunks)} - extracting triples...", end="", flush=True)

                        # Extract triples from combined batch
                        triples = self._extract_triples(combined_text)

                        # Batch resolve inverses
                        if add_inverse and triples:
                            self._update_inverse_cache_batch([p for s, p, o in triples], inverse_cache)

                        if verbose:
                            print(f" found {len(triples)} triples")

                        # Add triples to graph with page metadata for 3D visualization
                        page_metadata = {
                            'source_page': page_name,
                            'page_num': page_num + 1
                        }
                        for subj, pred, obj in triples:
                            # Validate both entities before adding
                            if self._is_valid_entity(subj) and self._is_valid_entity(obj):
                                self.graph_store.add_triple(subj, pred, obj, metadata=page_metadata, auto_save=False)
                                self.stats['total_triples'] += 1

                                # Add inverse relationship if enabled
                                if add_inverse:
                                    inverse_pred = self._get_inverse(pred, inverse_cache)
                                    self.graph_store.add_triple(obj, inverse_pred, subj, metadata=page_metadata, auto_save=False)
                                    self.stats['total_triples'] += 1
                            else:
                                # Skip invalid entities (optional: track in stats)
                                if verbose and (not self._is_valid_entity(subj) or not self._is_valid_entity(obj)):
                                    invalid_ent = subj if not self._is_valid_entity(subj) else obj
                                    # Only print if pronoun or generic (not long phrases to avoid spam)
                                    if len(invalid_ent.split()) <= 2:
                                        print(f"    [Skipped invalid entity: {invalid_ent}]")

                        self.stats['chunks_processed'] += len(batch)

                    except Exception as e:
                        if verbose:
                            print(f" ERROR: {e}")
                        else:
                            print(f"[KGBuilder] Error processing batch in {page_name}: {e}")
                        self.stats['errors'] += 1

                # Save after each page
                self.graph_store.save()

                page_triples = self.stats['total_triples'] - total_triples_before
                page_time = time.time() - page_start

                page_stats.append({
                    'page': page_name,
                    'page_num': page_num + 1,
                    'triples': page_triples,
                    'chunks': len(chunks),
                    'time': page_time
                })

                print(f"[KGBuilder] Page {page_num + 1}/{len(files)}: {page_name} - "
                      f"{page_triples} triples in {page_time:.1f}s")

            except Exception as e:
                print(f"[KGBuilder] Error processing page {page_name}: {e}")
                self.stats['errors'] += 1

            # Progress callback
            if progress_callback:
                progress_callback(page_num + 1, len(files), self.stats.copy())

        # Final stats
        self.stats['processing_time'] = time.time() - start_time
        self.stats['unique_entities'] = len(self.graph_store.get_entities())
        self.stats['pages_processed'] = len(files)
        self.stats['page_stats'] = page_stats

        # Add chunk type stats if hybrid mode is enabled
        if self.hybrid_mode and self.chunk_type_stats:
            self.stats['chunk_type_stats'] = self.chunk_type_stats.copy()
            print(f"[KGBuilder] Chunk types: {self.chunk_type_stats}")

        # Add entity deduplication stats (if enabled)
        if self.graph_store.enable_entity_resolution:
            dedup_stats_end = self.graph_store.get_deduplication_stats()
            self.stats['deduplication'] = dedup_stats_end

            # Calculate improvements
            if dedup_stats_start and dedup_stats_end:
                entities_added = dedup_stats_end['total_entities'] - dedup_stats_start.get('total_entities', 0)
                aliases_added = dedup_stats_end['total_aliases'] - dedup_stats_start.get('total_aliases', 0)
                dedup_ratio = dedup_stats_end.get('deduplication_ratio', 0)

                print(f"[KGBuilder] Entity Deduplication: {aliases_added} aliases → {entities_added} canonical entities")
                print(f"[KGBuilder] Dedup ratio: {dedup_ratio:.2f}")

                if dedup_stats_end.get('top_merged'):
                    print(f"[KGBuilder] Most merged entities:")
                    for entity, count in dedup_stats_end['top_merged'][:5]:
                        print(f"  - {entity}: {count} mentions")

        print(f"[KGBuilder] Complete! {self.stats['total_triples']} triples, "
              f"{self.stats['unique_entities']} entities from {len(files)} pages "
              f"in {self.stats['processing_time']:.1f}s")

        return self.stats.copy()

    def build_from_text(self, text: str,
                        progress_callback: Optional[Callable] = None,
                        add_inverse: bool = True) -> Dict:
        """
        Build KG from text content incrementally with batch processing.

        Args:
            text: Document text content
            progress_callback: Optional callback(chunk_num, total_chunks, stats)
            add_inverse: Whether to add inverse relationships

        Returns:
            Dict with extraction statistics
        """
        start_time = time.time()
        self._reset_stats()

        # Track entity deduplication stats (if enabled)
        dedup_stats_start = None
        if self.graph_store.enable_entity_resolution:
            dedup_stats_start = self.graph_store.get_deduplication_stats()

        # Auto-detect document type from first portion of text
        if self.doc_type is None:
            if self.auto_detect_type and self.classifier:
                self.doc_type = self.classifier.classify(text[:2000])
                print(f"[KGBuilder] Detected document type: {self.doc_type.value}")
            else:
                self.doc_type = DocumentType.TECHNICAL  # Default fallback
                print(f"[KGBuilder] Using default document type: {self.doc_type.value}")
        else:
            print(f"[KGBuilder] Using specified document type: {self.doc_type.value}")

        self.stats['doc_type'] = self.doc_type.value

        # Split into chunks
        chunks = list(self._chunk_text(text))
        total_chunks = len(chunks)

        # Calculate batches
        num_batches = (total_chunks + self.batch_size - 1) // self.batch_size

        print(f"[KGBuilder] Processing {total_chunks} chunks in {num_batches} batches (batch_size={self.batch_size})...")

        # Cache for inverse relations
        inverse_cache: Dict[str, str] = {}

        # Process chunks in batches
        for batch_idx in range(num_batches):
            batch_start = time.time()

            # Get chunk indices for this batch
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_chunks)
            batch = chunks[start_idx:end_idx]

            print(f"[KGBuilder] Batch {batch_idx + 1}/{num_batches}: Processing chunks {start_idx}-{end_idx-1}")

            try:
                # Combine multiple chunks into single text
                combined_text = "\n\n".join(batch)

                # Extract triples from combined batch (ONE LLM call for entire batch)
                triples = self._extract_triples(combined_text)

                # Batch resolve inverses
                if add_inverse and triples:
                    self._update_inverse_cache_batch([p for s, p, o in triples], inverse_cache)

                # Add triples to graph
                for subj, pred, obj in triples:
                    self.graph_store.add_triple(subj, pred, obj, auto_save=False)
                    self.stats['total_triples'] += 1

                    # Add inverse relationship if enabled
                    if add_inverse:
                        inverse_pred = self._get_inverse(pred, inverse_cache)
                        self.graph_store.add_triple(obj, inverse_pred, subj, auto_save=False)
                        self.stats['total_triples'] += 1

                self.stats['chunks_processed'] += len(batch)

                batch_elapsed = time.time() - batch_start
                print(f"[KGBuilder] Batch {batch_idx + 1}/{num_batches} completed in {batch_elapsed:.2f}s, extracted {len(triples)} triples")

            except Exception as e:
                print(f"[KGBuilder] Error processing batch {batch_idx}: {e}")
                self.stats['errors'] += 1

            # Periodic save
            if (batch_idx + 1) % self.save_interval == 0:
                self.graph_store.save()
                print(f"[KGBuilder] Progress: Batch {batch_idx + 1}/{num_batches}, "
                      f"{self.stats['total_triples']} triples")

            # Progress callback
            if progress_callback:
                progress_callback(end_idx, total_chunks, self.stats.copy())

        # Final save
        self.graph_store.save()

        self.stats['processing_time'] = time.time() - start_time
        self.stats['unique_entities'] = len(self.graph_store.get_entities())

        # Add chunk type stats if hybrid mode is enabled
        if self.hybrid_mode and self.chunk_type_stats:
            self.stats['chunk_type_stats'] = self.chunk_type_stats.copy()
            print(f"[KGBuilder] Chunk types: {self.chunk_type_stats}")

        # Add entity deduplication stats (if enabled)
        if self.graph_store.enable_entity_resolution:
            dedup_stats_end = self.graph_store.get_deduplication_stats()
            self.stats['deduplication'] = dedup_stats_end

            # Calculate improvements
            if dedup_stats_start and dedup_stats_end:
                entities_added = dedup_stats_end['total_entities'] - dedup_stats_start.get('total_entities', 0)
                aliases_added = dedup_stats_end['total_aliases'] - dedup_stats_start.get('total_aliases', 0)
                dedup_ratio = dedup_stats_end.get('deduplication_ratio', 0)

                print(f"[KGBuilder] Entity Deduplication: {aliases_added} aliases → {entities_added} canonical entities")
                print(f"[KGBuilder] Dedup ratio: {dedup_ratio:.2f}")

        print(f"[KGBuilder] Complete! {self.stats['total_triples']} triples, "
              f"{self.stats['unique_entities']} entities in {self.stats['processing_time']:.1f}s")

        return self.stats.copy()

    def _chunk_text(self, text: str) -> Generator[str, None, None]:
        """
        Split text into overlapping chunks.

        Uses sentence boundaries when possible to avoid cutting mid-sentence.
        """
        # Clean text
        text = text.strip()
        if not text:
            return

        # Try to split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    yield current_chunk.strip()

                # Start new chunk with overlap
                # Take last few words from previous chunk
                if self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + sentence + " "
                else:
                    current_chunk = sentence + " "

        # Yield remaining text
        if current_chunk.strip():
            yield current_chunk.strip()

    def _extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract knowledge triples from a text chunk.

        If pattern_only is True, uses only regex patterns (fast, stable).
        Otherwise uses LLM with timeout protection.

        Returns list of (subject, predicate, object) tuples.
        """
        if self.debug:
            print(f"      [DEBUG] _extract_triples: starting ({len(text)} chars)")
            t_start = time.time()

        # Pattern-only mode: skip LLM entirely
        if self.pattern_only:
            triples = self._extract_triples_pattern_fallback(text)
            if self.debug:
                print(f"      [DEBUG] _extract_triples: pattern-only took {time.time()-t_start:.2f}s -> {len(triples)} triples")
            return triples

        # Determine chunk type for prompt selection
        if self.hybrid_mode and self.chunk_classifier:
            if self.debug:
                t_classify = time.time()
            # Per-chunk type detection
            # Uses LLM-based (multi-language) or pattern-based depending on config
            chunk_type, confidence = self.chunk_classifier.classify(text) if self.use_llm_classifier else self.chunk_classifier.detect(text)

            if self.debug:
                method = "LLM" if self.use_llm_classifier else "pattern"
                print(f"      [DEBUG] Chunk classification ({method}): {time.time()-t_classify:.2f}s -> {chunk_type.value}")

            # Track stats
            type_name = chunk_type.value
            self.chunk_type_stats[type_name] = self.chunk_type_stats.get(type_name, 0) + 1

            # Use detected type (UNKNOWN falls back to hybrid multi-type prompt)
            prompt_template = get_extraction_prompt(chunk_type, verbose=self.verbose_mode)
        else:
            # Use document-level type
            prompt_template = get_extraction_prompt(self.doc_type or DocumentType.TECHNICAL, verbose=self.verbose_mode)

        prompt = prompt_template.format(text=text)

        # Use manual daemon thread for timeout handling
        result_container = {}

        def worker():
            try:
                result_container['response'] = self.llm.invoke(prompt)
            except Exception as e:
                result_container['error'] = e

        if self.debug:
            t_llm = time.time()
            print(f"      [DEBUG] LLM extraction: starting (prompt: {len(prompt)} chars, text: {len(text)} chars)...")
        else:
            print(".", end="", flush=True)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=self.llm_timeout)

        if t.is_alive():
            # TIMEOUT
            if self.debug:
                print(f"      [DEBUG] LLM extraction: TIMEOUT after {self.llm_timeout}s")
                print(f"      [DEBUG]   -> Using pattern fallback instead")
            else:
                print(f" TIMEOUT ({self.llm_timeout}s) - using pattern fallback", end="")
            
            # Fallback: extract basic patterns from text
            triples = self._extract_triples_pattern_fallback(text)
            return triples

        if 'error' in result_container:
            print(f"[KGBuilder] Triple extraction error: {result_container['error']}")
            return []

        response = result_container.get('response', '')

        if self.debug:
            print(f"      [DEBUG] LLM extraction: {time.time()-t_llm:.2f}s")

        # Handle different response types
        if isinstance(response, dict):
            response_text = response.get('result', '') or response.get('text', str(response))
        else:
            # Extract .content from AIMessage objects to avoid metadata pollution
            response_text = response.content if hasattr(response, 'content') else str(response)

        # Parse triples from response
        triples = self._parse_triples(response_text)

        if self.debug:
            print(f"      [DEBUG] _extract_triples total: {time.time()-t_start:.2f}s -> {len(triples)} triples")

        return triples

    def _is_valid_entity(self, entity: str) -> bool:
        """
        Filter out invalid entities (pronouns, generic terms, overly long phrases).

        Validates entities to ensure only high-quality noun entities are extracted:
        - Minimum 2 characters
        - Maximum 4 words (disabled in verbose mode)
        - No pronouns (he, she, it, they, etc.)
        - No generic terms (speaker, person, someone, thing, etc.)
        - Not mostly lowercase words (likely phrases, not proper nouns) - disabled in verbose mode

        In verbose mode, only rejects:
        - Empty or very short entities (< 2 chars)
        - Pure pronouns

        Args:
            entity: Entity string to validate

        Returns:
            True if entity is valid, False otherwise

        Examples:
            >>> _is_valid_entity("George Hadley")  # True (proper noun)
            >>> _is_valid_entity("He")  # False (pronoun)
            >>> _is_valid_entity("the concern of the father about the situation")  # False in standard, True in verbose
        """
        entity = entity.strip()

        # Reject empty or very short
        if len(entity) < 2:
            return False

        # In verbose mode, only reject pronouns - allow everything else
        if self.verbose_mode:
            pronouns = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'me', 'him', 'her', 'them', 'us'}
            if entity.lower() in pronouns:
                return False
            return True

        # Standard mode: strict validation
        # Reject overly long entities (likely full phrases)
        word_count = len(entity.split())
        if word_count > 4:  # Max 4 words per entity
            return False

        # Reject pronouns
        pronouns = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'me', 'him', 'her', 'them', 'us'}
        if entity.lower() in pronouns:
            return False

        # Reject generic terms (too vague)
        generic = {'speaker', 'person', 'someone', 'something', 'thing', 'place', 'time', 'way'}
        if entity.lower() in generic:
            return False

        # Reject if entity is mostly lowercase conjunctions/prepositions
        # (likely a phrase, not a proper noun)
        lowercase_words = [w for w in entity.split() if w.islower()]
        if len(lowercase_words) > len(entity.split()) // 2:
            # More than half the words are lowercase (not proper nouns)
            return False

        return True

    def _extract_triples_pattern_fallback(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Simple pattern-based triple extraction as fallback when LLM times out.
        Extracts basic entity relationships using regex patterns.
        """
        triples = []

        # Pattern 1: "X is Y" or "X est Y" (French)
        is_patterns = [
            r'([A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý]?[a-zà-ÿ]+)*)\s+(?:is|est|was|était)\s+(?:a|an|un|une|the|le|la)?\s*([a-zà-ÿ]+(?:\s+[a-zà-ÿ]+)*)',
        ]

        for pattern in is_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for subj, obj in matches[:3]:  # Limit to 3 per pattern
                if len(subj) > 2 and len(obj) > 2:
                    triples.append((subj.strip(), 'IS_A', obj.strip()))

        # Pattern 2: "X, Y of Z" (titles)
        title_pattern = r'([A-ZÀ-Ý][a-zà-ÿ]+),?\s+(?:king|queen|prince|goddess|god|roi|reine|prince|déesse|dieu)\s+(?:of|de|du|des)\s+([A-ZÀ-Ý][a-zà-ÿ]+)'
        matches = re.findall(title_pattern, text, re.IGNORECASE)
        for name, place in matches[:3]:
            triples.append((name.strip(), 'RULES', place.strip()))

        # Pattern 3: "X and Y" (relationships)
        and_pattern = r'([A-ZÀ-Ý][a-zà-ÿ]+)\s+(?:and|et)\s+([A-ZÀ-Ý][a-zà-ÿ]+)'
        matches = re.findall(and_pattern, text)
        for name1, name2 in matches[:2]:
            if len(name1) > 2 and len(name2) > 2:
                triples.append((name1.strip(), 'RELATED_TO', name2.strip()))

        return triples

    def _parse_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Parse triples from LLM response.

        Handles various formats:
        - (Subject, PREDICATE, Object)
        - Subject, PREDICATE, Object
        - Subject -> PREDICATE -> Object
        - Natural language (qwen-compatible): "Subject is/was ROLE of Object"
        """
        triples = []

        # Pattern 1: (Subject, Predicate, Object)
        pattern1 = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
        matches = re.findall(pattern1, text)
        for match in matches:
            subj, pred, obj = [m.strip() for m in match]
            if subj and pred and obj:
                triples.append((subj, pred.upper(), obj))

        # Pattern 2: Subject -> Predicate -> Object (if pattern 1 didn't match)
        if not triples:
            pattern2 = r'([^->\n]+)\s*->\s*([^->\n]+)\s*->\s*([^->\n]+)'
            matches = re.findall(pattern2, text)
            for match in matches:
                subj, pred, obj = [m.strip() for m in match]
                if subj and pred and obj:
                    triples.append((subj, pred.upper(), obj))

        # Pattern 3: Simple comma-separated (Subject, Predicate, Object per line)
        if not triples:
            for line in text.split('\n'):
                line = line.strip()
                if line and ',' in line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        subj, pred, obj = parts[0], parts[1], parts[2]
                        if subj and pred and obj:
                            triples.append((subj, pred.upper(), obj))

        # Pattern 4: Natural Language (qwen-compatible)
        # Extracts from sentences like "X is/was ROLE of Y" or "X VERB Y"
        if not triples:
            for line in text.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Pattern 4a: "X is/was (a/the) ROLE of Y"
                # Example: "Paris was a prince of Troy" → (Paris, PRINCE_OF, Troy)
                pattern4a = r'([A-Z][a-zA-Z\s]+?)\s+(?:is|was)\s+(?:a|the|an)?\s*([a-z][a-zA-Z\s]+?)\s+of\s+([A-Z][a-zA-Z\s]+?)(?:\.|$)'
                matches = re.findall(pattern4a, line, re.IGNORECASE)
                for match in matches:
                    subj, role, obj = [m.strip() for m in match]
                    if subj and role and obj:
                        predicate = role.upper().replace(' ', '_') + '_OF'
                        triples.append((subj, predicate, obj))

                # Pattern 4b: "X is/was Y's ROLE"
                # Example: "Paris was King Priam's son" → (Paris, SON_OF, King Priam)
                pattern4b = r"([A-Z][a-zA-Z\s]+?)\s+(?:is|was)\s+([A-Z][a-zA-Z\s]+?)'s\s+([a-z][a-zA-Z]+)"
                matches = re.findall(pattern4b, line)
                for match in matches:
                    subj, obj, role = [m.strip() for m in match]
                    if subj and role and obj:
                        predicate = role.upper() + '_OF'
                        triples.append((subj, predicate, obj))

                # Pattern 4c: "X VERB Y Z" → (X, VERB, Z)
                # Example: "Hera offered him power" → (Hera, OFFERED, power)
                pattern4c = r'([A-Z][a-zA-Z\s]+?)\s+(offered|promised|chose|gave|took|received|sought|desired)\s+(?:him|her|them)?\s*([a-z][a-zA-Z\s]+?)(?:\.|,|$)'
                matches = re.findall(pattern4c, line, re.IGNORECASE)
                for match in matches:
                    subj, verb, obj = [m.strip() for m in match]
                    if subj and verb and obj:
                        predicate = verb.upper()
                        # Convert past tense to present if needed
                        if predicate.endswith('ED'):
                            if predicate == 'OFFERED':
                                predicate = 'OFFERS'
                            elif predicate == 'PROMISED':
                                predicate = 'PROMISES'
                            elif predicate == 'CHOSE' or predicate == 'CHOOSED':
                                predicate = 'CHOOSES'
                        triples.append((subj, predicate, obj))

        return triples

    def _update_inverse_cache_batch(self, predicates: List[str], cache: Dict[str, str]):
        """
        Batch process inverse relations to minimize LLM calls.
        """
        # Identify predicates not in cache
        unknowns = sorted(list(set(p for p in predicates if p not in cache)))
        if not unknowns:
            return

        # First pass: Check static rules
        really_unknowns = []
        for p in unknowns:
            pred_upper = p.upper()
            inv = get_inverse_relation(pred_upper, self.doc_type or DocumentType.TECHNICAL)
            if not inv.startswith('INVERSE_OF_'):
                cache[p] = inv
            else:
                really_unknowns.append(p)
        
        if not really_unknowns:
            return

        # Second pass: Batch LLM call for true unknowns
        if self.debug:
            print(f"      [DEBUG] Batch resolving inverses for: {really_unknowns}")
            
        prompt = f"""Given a list of relationship predicates from a knowledge graph, provide the INVERSE relationship for each.
Provide the output as a list of "Original -> Inverse" pairs.
Use uppercase for inverses.
Context: {self.doc_type.value if self.doc_type else 'General'}

Predicates:
""" + "\n".join(f"- {p}" for p in really_unknowns) + """

Format:
Original -> Inverse

Response:"""

        try:
            # Use manual daemon thread
            result_container = {}

            def worker():
                try:
                    result_container['response'] = self.llm.invoke(prompt)
                except Exception as e:
                    result_container['error'] = e

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            t.join(timeout=self.llm_timeout * 2) # Double timeout for batch

            if t.is_alive():
                # TIMEOUT
                if self.debug:
                    print(f"      [DEBUG] Batch inverse TIMEOUT")
                return

            if 'error' in result_container:
                if self.debug:
                    print(f"[KGBuilder] Batch error: {result_container['error']}")
                return

            response = result_container.get('response', '')

            if isinstance(response, dict):
                text = response.get('result', '') or response.get('text', '')
            else:
                text = str(response)

            # Parse "Original -> Inverse" lines
            for line in text.split('\n'):
                if '->' in line:
                    parts = line.split('->')
                    if len(parts) == 2:
                        orig = parts[0].strip().strip('- ').strip()
                        inv = parts[1].strip().upper()
                        # Map back to original if found (handling case sensitivity loose)
                        for unknown in really_unknowns:
                            if unknown.lower() == orig.lower():
                                cache[unknown] = inv
                                break
                                
        except Exception as e:
            print(f"[KGBuilder] Batch inverse error: {e}")

    def _get_inverse(self, predicate: str, cache: Dict[str, str]) -> str:
        """
        Get or compute inverse relationship.

        Uses type-specific inverse mappings first, then falls back to LLM.
        Uses caching to avoid repeated LLM calls for same predicates.
        """
        if predicate in cache:
            return cache[predicate]

        pred_upper = predicate.upper()

        # Try type-specific inverse mapping first
        inverse = get_inverse_relation(pred_upper, self.doc_type or DocumentType.TECHNICAL)

        # If we got a real inverse (not a fallback), use it
        if not inverse.startswith('INVERSE_OF_'):
            cache[predicate] = inverse
            return inverse

        # Use LLM for unknown predicates (with timeout)
        prompt = f"""Given the relationship '{predicate}' between entities in a knowledge graph,
provide ONLY the inverse relationship label in uppercase.
Examples:
- 'HAS CAPITAL' -> 'IS CAPITAL OF'
- 'SHARES BORDER WITH' -> 'SHARES BORDER WITH' (symmetric)
- 'IS LOCATED IN' -> 'CONTAINS'

Relationship: '{predicate}'
Inverse (uppercase only):"""

        try:
            if self.debug:
                print(f"      [DEBUG] _get_inverse: LLM call for '{predicate}'...")
                t0 = time.time()

            # Use manual daemon thread
            result_container = {}

            def worker():
                try:
                    result_container['response'] = self.llm.invoke(prompt)
                except Exception as e:
                    result_container['error'] = e

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            t.join(timeout=self.llm_timeout)

            if t.is_alive():
                # TIMEOUT
                if self.debug:
                    print(f"      [DEBUG] _get_inverse: TIMEOUT, using fallback")
                cache[predicate] = f"INVERSE_{pred_upper}"
                return cache[predicate]
            
            if 'error' in result_container:
                raise result_container['error']

            response = result_container.get('response', '')

            if self.debug:
                print(f"      [DEBUG] _get_inverse: {time.time()-t0:.2f}s")

            if isinstance(response, dict):
                inverse = response.get('result', '') or response.get('text', '')
            else:
                # Extract .content from AIMessage objects to avoid metadata pollution
                inverse = response.content if hasattr(response, 'content') else str(response)

            inverse = inverse.strip().upper()
            if inverse:
                cache[predicate] = inverse
                return inverse
        except Exception as e:
            print(f"[KGBuilder] Failed to compute inverse for {predicate}: {e}")

        # Fallback: reverse the predicate
        cache[predicate] = f"INVERSE_{pred_upper}"
        return cache[predicate]

    def _reset_stats(self):
        """Reset statistics for new build."""
        self.stats = {
            'chunks_processed': 0,
            'total_triples': 0,
            'unique_entities': 0,
            'errors': 0,
            'processing_time': 0.0
        }
        self.chunk_type_stats = {}  # Reset chunk type tracking


class StreamingKGBuilder:
    """
    Memory-efficient KG builder for very large files.

    Reads file line-by-line instead of loading entirely into memory.
    Useful for documents larger than available RAM.
    """

    def __init__(self, llm, graph_store: ScalableGraphStore,
                 lines_per_chunk: int = 50):
        """
        Initialize streaming builder.

        Args:
            llm: LangChain LLM for extraction
            graph_store: Target graph store
            lines_per_chunk: Lines to process per LLM call
        """
        self.llm = llm
        self.graph_store = graph_store
        self.lines_per_chunk = lines_per_chunk
        self.builder = IncrementalKGBuilder(llm, graph_store)

    def build_from_large_file(self, filepath: str,
                              progress_callback: Optional[Callable] = None) -> Dict:
        """
        Process very large file without loading into memory.

        Args:
            filepath: Path to large text file
            progress_callback: Optional progress callback

        Returns:
            Extraction statistics
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        start_time = time.time()
        stats = {
            'lines_processed': 0,
            'chunks_processed': 0,
            'total_triples': 0,
            'unique_entities': 0,
            'errors': 0,
            'processing_time': 0.0
        }

        current_chunk = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                current_chunk.append(line)
                stats['lines_processed'] += 1

                if len(current_chunk) >= self.lines_per_chunk:
                    chunk_text = ''.join(current_chunk)
                    try:
                        triples = self.builder._extract_triples(chunk_text)
                        for subj, pred, obj in triples:
                            self.graph_store.add_triple(subj, pred, obj, auto_save=False)
                            stats['total_triples'] += 1
                        stats['chunks_processed'] += 1
                    except Exception as e:
                        stats['errors'] += 1

                    current_chunk = []

                    # Periodic save
                    if stats['chunks_processed'] % 10 == 0:
                        self.graph_store.save()
                        if progress_callback:
                            progress_callback(stats['chunks_processed'], -1, stats)

        # Process remaining lines
        if current_chunk:
            chunk_text = ''.join(current_chunk)
            try:
                triples = self.builder._extract_triples(chunk_text)
                for subj, pred, obj in triples:
                    self.graph_store.add_triple(subj, pred, obj, auto_save=False)
                    stats['total_triples'] += 1
                stats['chunks_processed'] += 1
            except:
                stats['errors'] += 1

        self.graph_store.save()

        stats['processing_time'] = time.time() - start_time
        stats['unique_entities'] = len(self.graph_store.get_entities())

        print(f"[StreamingKGBuilder] Complete! {stats['total_triples']} triples from "
              f"{stats['lines_processed']} lines in {stats['processing_time']:.1f}s")

        return stats


class DocumentProcessor:
    """
    High-level document processor that combines chunking, extraction, and graph building.

    Provides a simple interface for processing various document types.
    """

    def __init__(self, llm, graph_store: ScalableGraphStore):
        self.llm = llm
        self.graph_store = graph_store
        self.builder = IncrementalKGBuilder(llm, graph_store)

    def process_text_file(self, filepath: str, **kwargs) -> Dict:
        """Process a plain text file."""
        return self.builder.build_from_file(filepath, **kwargs)

    def process_text(self, text: str, **kwargs) -> Dict:
        """Process raw text content."""
        return self.builder.build_from_text(text, **kwargs)

    def estimate_processing_time(self, filepath: str) -> Dict:
        """
        Estimate processing time for a file.

        Returns:
            Dict with file stats and time estimate
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        content = filepath.read_text(encoding='utf-8')
        num_chars = len(content)
        num_words = len(content.split())

        # Rough estimates based on chunk processing
        num_chunks = max(1, num_chars // self.builder.chunk_size)

        # Assume ~2-5 seconds per chunk (depends on LLM speed)
        est_time_min = num_chunks * 2
        est_time_max = num_chunks * 5

        return {
            'file_path': str(filepath),
            'characters': num_chars,
            'words': num_words,
            'estimated_chunks': num_chunks,
            'estimated_time_seconds': {
                'min': est_time_min,
                'max': est_time_max
            }
        }
