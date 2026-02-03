"""
Scalable Hybrid RAG+KG Query System
Combines vector retrieval with subgraph extraction

This module provides the ScalableHybridRAG class that:
- Extracts SUBGRAPH (20-50 nodes) instead of full graph (10,000+ nodes)
- Uses entity extraction from query to guide KG retrieval (Phase 2 enhanced)
- Enriches RAG chunks with graph context
"""
from typing import List, Dict, Any, Optional, Set
import time

from .graph_store import ScalableGraphStore
from .entity_extractor import EntityExtractor


class ScalableHybridRAG:
    """
    Hybrid RAG + Knowledge Graph system that scales to large documents.

    Key difference from original hybrid_rag():
    - Extracts SUBGRAPH (20-50 nodes) instead of full graph (10,000+ nodes)
    - Uses entity extraction from query to guide KG retrieval
    - Enriches RAG chunks with graph context

    Usage:
        hybrid = ScalableHybridRAG(llm, vector_store, graph_store)
        result = hybrid.query("What can you tell me about Telemachus?")
        print(result['response'])
    """

    def __init__(self, llm, vector_store, graph_store: ScalableGraphStore,
                 embeddings=None, collection_name: str = None,
                 source_directory: str = None,
                 use_enhanced_extraction: bool = True):
        """
        Initialize scalable hybrid RAG system.

        Args:
            llm: LangChain LLM instance (local Ollama or remote)
            vector_store: Chroma vector store instance
            graph_store: ScalableGraphStore instance
            embeddings: Optional embeddings model (for future use)
            collection_name: Collection name for storage (sanitized)
            source_directory: Original source directory path (for file loading)
            use_enhanced_extraction: Use Phase 2 EntityExtractor (default True)
        """
        self.llm = llm
        self.vector_store = vector_store  # Chroma instance
        self.graph_store = graph_store    # ScalableGraphStore
        self.embeddings = embeddings
        self.collection_name = collection_name  # Sanitized name for storage
        self.source_directory = source_directory  # Original path for file loading
        self._known_entities = graph_store.get_entities()

        # Phase 2: Enhanced entity extraction
        self.use_enhanced_extraction = use_enhanced_extraction
        if use_enhanced_extraction:
            self.entity_extractor = EntityExtractor(
                llm=llm,
                known_entities=self._known_entities,
                use_cache=True
            )
        else:
            self.entity_extractor = None

    def refresh_entity_index(self):
        """Refresh the known entities index from the graph store."""
        self._known_entities = self.graph_store.get_entities()
        # Also refresh EntityExtractor if using enhanced extraction
        if self.entity_extractor:
            self.entity_extractor.update_known_entities(self._known_entities)

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses rough heuristic: 1 token ≈ 4 characters (conservative for GPT models).
        For more accuracy, could use tiktoken library.

        Args:
            text: Input text string

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def _calculate_context_budget(self, query: str, model_context_limit: int = 128000,
                                   response_reserve: int = 4000,
                                   system_reserve: int = 1000) -> int:
        """
        Calculate available token budget for RAG chunks.

        Args:
            query: User query string
            model_context_limit: Total context window (default: 128k for gpt-4o-mini)
            response_reserve: Tokens reserved for LLM response (default: 4000)
            system_reserve: Tokens reserved for system prompts and formatting (default: 1000)

        Returns:
            Available tokens for RAG+KG context
        """
        query_tokens = self._estimate_tokens(query)
        available = model_context_limit - query_tokens - response_reserve - system_reserve
        return max(available, 10000)  # Minimum 10k tokens for context

    def extract_query_entities(self, query: str,
                               use_llm_fallback: bool = True) -> List[str]:
        """
        Extract entities from user query to guide KG retrieval.

        Phase 2 Enhanced: Uses EntityExtractor for better extraction with:
        - Pattern-based extraction (proper nouns, capitalized words)
        - Known entity matching (substring, fuzzy)
        - LLM-based extraction (fallback for complex queries)

        Args:
            query: User's question
            use_llm_fallback: If True, use LLM when no matches found

        Returns:
            List of matched entity names from the graph
        """
        # Phase 2: Use enhanced EntityExtractor if available
        if self.entity_extractor:
            return self.entity_extractor.extract(query, use_llm=use_llm_fallback)

        # Fallback to original simple matching
        matched = []
        query_lower = query.lower()
        query_words = [w for w in query_lower.split() if len(w) > 3]

        # Method 1: Direct substring matching against known entities
        for entity in self._known_entities:
            entity_lower = entity.lower()

            # Check if entity appears in query
            if entity_lower in query_lower:
                matched.append(entity)
                continue

            # Check if any query word appears in entity
            for word in query_words:
                if word in entity_lower:
                    matched.append(entity)
                    break

        # Method 2: LLM-based extraction (fallback)
        if not matched and use_llm_fallback and self.llm:
            prompt = f"""Extract the key entities (names, places, concepts) from this query.
Return ONLY a comma-separated list, nothing else.

Query: {query}

Entities:"""
            try:
                response = self.llm.invoke(prompt)
                if isinstance(response, str):
                    llm_entities = [e.strip() for e in response.strip().split(",")]
                else:
                    # Handle dict response from some LLMs
                    text = response.get('result', '') if isinstance(response, dict) else str(response)
                    llm_entities = [e.strip() for e in text.strip().split(",")]

                # Match LLM entities against known entities
                for ent in llm_entities:
                    if not ent:
                        continue
                    ent_lower = ent.lower()
                    for known in self._known_entities:
                        known_lower = known.lower()
                        if (ent_lower in known_lower or known_lower in ent_lower):
                            matched.append(known)

            except Exception as e:
                print(f"[Hybrid] LLM entity extraction failed: {e}")

        # Deduplicate and limit
        return list(set(matched))[:10]

    def query(self, query: str,
              kg_hops: int = 2,
              max_kg_nodes: int = 50,
              score_threshold: float = 0.75,
              model_context_limit: int = 128000,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Execute hybrid RAG + KG query with dynamic context sizing.

        Args:
            query: User question
            kg_hops: Number of hops for subgraph extraction
            max_kg_nodes: Maximum nodes in subgraph
            score_threshold: Minimum similarity threshold (0.0-1.0, default: 0.75)
                             Lower = more permissive, Higher = more selective
                             Uses adaptive retry: 0.75 → 0.50 → 0.25 → 0.10
            model_context_limit: LLM context window size in tokens (default: 128k)
            verbose: Print debug info

        Returns:
            Dict with:
                - response: LLM generated answer
                - query_entities: Entities found in query
                - rag_chunks: Number of chunks retrieved (dynamic)
                - rag_tokens_used: Estimated tokens used by RAG context
                - rag_candidates: Total candidates above threshold
                - rag_duplicates: Number of duplicates removed
                - rag_unique: Unique chunks after deduplication
                - context_budget: Available token budget
                - subgraph_nodes: Number of KG nodes used
                - subgraph_edges: Number of KG edges used
                - timing: Dict of timing metrics
        """
        timing = {}

        # 1. Extract entities from query
        start = time.time()
        query_entities = self.extract_query_entities(query)
        timing['entity_extraction'] = time.time() - start

        if verbose:
            print(f"[Hybrid] Extracted entities: {query_entities}")

        # 2. Retrieve RAG chunks DYNAMICALLY with ADAPTIVE RETRY and DEDUPLICATION
        start = time.time()
        try:
            # Calculate context budget
            context_budget_tokens = self._calculate_context_budget(query, model_context_limit)

            # Adaptive retry strategy: Try up to 3 times with progressively lower thresholds
            # Start high (0.75) for precision, jump down by 0.25 if no results
            # Minimum threshold is 0.10 (never goes below)
            max_retries = 3
            threshold_step = 0.25  # Large jumps: 0.75 → 0.50 → 0.25 → 0.10
            current_threshold = score_threshold
            all_chunks = []
            retry_count = 0

            while retry_count < max_retries and len(all_chunks) == 0:
                previous_threshold = current_threshold  # Track before retry

                # Retrieve ALL chunks above threshold (no k limit)
                retriever = self.vector_store.as_retriever(
                    search_type='similarity_score_threshold',
                    search_kwargs={
                        "k": 100000,  # Very high k to get ALL above threshold
                        "score_threshold": current_threshold
                    }
                )
                all_chunks = retriever.invoke(query)

                # If no chunks found, lower threshold and retry
                if len(all_chunks) == 0:
                    retry_count += 1
                    current_threshold = max(0.1, current_threshold - threshold_step)
                    # Always show threshold changes (not just in verbose mode)
                    if retry_count < max_retries:
                        print(f"[Hybrid] No chunks at threshold {previous_threshold:.2f}, retrying at {current_threshold:.2f}...")
                else:
                    # Always show when we succeed after a retry
                    if retry_count > 0:
                        print(f"[Hybrid] ✓ Found {len(all_chunks)} candidates at threshold {current_threshold:.2f}")
                    elif verbose:
                        print(f"[Hybrid] Retrieved {len(all_chunks)} candidates above threshold {current_threshold:.2f}")

            # If still no chunks after retries, warn and continue with empty
            if len(all_chunks) == 0:
                print(f"[Hybrid] Warning: No chunks found even after {max_retries} retries (final threshold: {current_threshold:.2f})")

            # DEDUPLICATION: Remove duplicate chunks before token selection
            # Use (page_num, chunk_num) as unique key, same as source excerpt deduplication
            seen = set()
            deduplicated_chunks = []
            duplicate_count = 0
            duplicate_details = []  # Track details of duplicates for diagnostics

            for chunk in all_chunks:
                metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
                page_num = metadata.get('page_num', 0)
                chunk_num = metadata.get('chunk_num', 0)

                # Skip duplicates
                key = (page_num, chunk_num)
                if key in seen:
                    duplicate_count += 1
                    # Track duplicate info for diagnostics
                    if verbose and duplicate_count <= 3:  # Only track first 3
                        content_preview = chunk.page_content[:50] if hasattr(chunk, 'page_content') else 'N/A'
                        duplicate_details.append({
                            'key': key,
                            'preview': content_preview,
                            'metadata': metadata
                        })
                    continue
                seen.add(key)
                deduplicated_chunks.append(chunk)

            if verbose and duplicate_count > 0:
                print(f"[Hybrid] Removed {duplicate_count} duplicate chunks ({len(deduplicated_chunks)} unique remaining)")
                # Print diagnostic info about duplicates
                if duplicate_details:
                    print(f"[Hybrid] Sample duplicate details (first {len(duplicate_details)}):")
                    for i, dup in enumerate(duplicate_details):
                        print(f"  Duplicate {i+1}: page={dup['key'][0]}, chunk={dup['key'][1]}")
                        print(f"    Preview: {dup['preview']}...")
                        print(f"    Metadata: {dup['metadata']}")

            # Dynamically select chunks that fit in context budget
            rag_chunks = []
            total_tokens = 0
            avg_kg_tokens = 1000  # Reserve ~1000 tokens for KG context
            available_for_rag = context_budget_tokens - avg_kg_tokens

            for chunk in deduplicated_chunks:
                content = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
                chunk_tokens = self._estimate_tokens(content)

                # Add chunk if it fits in budget
                if total_tokens + chunk_tokens <= available_for_rag:
                    rag_chunks.append(chunk)
                    total_tokens += chunk_tokens
                else:
                    break  # Stop when budget exhausted

            if verbose:
                print(f"[Hybrid] Selected {len(rag_chunks)} chunks fitting in {available_for_rag} token budget")
                print(f"[Hybrid] Using ~{total_tokens} tokens for RAG context")

        except Exception as e:
            print(f"[Hybrid] RAG retrieval error: {e}")
            rag_chunks = []
            all_chunks = []
            deduplicated_chunks = []
            duplicate_count = 0
            total_tokens = 0
            available_for_rag = 0
        timing['rag_retrieval'] = time.time() - start

        # 3. Extract subgraph (THE KEY SCALABILITY FEATURE)
        start = time.time()
        if query_entities:
            subgraph = self.graph_store.get_subgraph(
                entities=query_entities,
                hops=kg_hops,
                max_nodes=max_kg_nodes
            )
        else:
            # No entities found - use empty subgraph
            import networkx as nx
            subgraph = nx.DiGraph()
            if verbose:
                print("[Hybrid] No entities matched, using RAG only")
        timing['subgraph_extraction'] = time.time() - start

        # 4. Extract source information for display
        start = time.time()
        source_excerpts, source_pages, page_provenance = self._extract_source_information(
            rag_chunks=rag_chunks,
            subgraph=subgraph,
            collection_name=self.collection_name  # Use stored collection name
        )
        timing['source_extraction'] = time.time() - start

        # 5. Build enriched context
        start = time.time()
        context = self._build_context(rag_chunks, subgraph, query_entities)
        timing['context_building'] = time.time() - start

        # 5. Generate response
        start = time.time()
        prompt = f"""Based on the following context from the document, answer the question.
Use the Knowledge Graph Context to understand relationships between entities.

{context}

Question: {query}

Answer:"""

        try:
            response = self.llm.invoke(prompt)
            # Handle different response types
            if isinstance(response, dict):
                response = response.get('result', str(response))
            response = str(response)
        except Exception as e:
            response = f"Error generating response: {e}"
        timing['llm_generation'] = time.time() - start

        timing['total'] = sum(timing.values())

        if verbose:
            print(f"[Hybrid] Total time: {timing['total']:.2f}s")
            self._print_timing_summary(timing)

        return {
            'response': response,
            'query_entities': query_entities,
            'rag_chunks': len(rag_chunks),
            'rag_tokens_used': total_tokens,  # NEW: Track token usage
            'rag_candidates': len(all_chunks),  # NEW: Total candidates above threshold
            'rag_duplicates': duplicate_count,  # NEW: Number of duplicates removed
            'rag_unique': len(deduplicated_chunks),  # NEW: Unique chunks after deduplication
            'context_budget': available_for_rag,  # NEW: Available token budget
            'subgraph_nodes': subgraph.number_of_nodes(),
            'subgraph_edges': subgraph.number_of_edges(),
            'timing': timing,
            # Source extraction
            'source_excerpts': source_excerpts,
            'source_pages': source_pages,
            'page_provenance': page_provenance,  # NEW: Track RAG vs KG provenance
            'subgraph': subgraph  # Needed for visualization
        }

    def _build_context(self, rag_chunks, subgraph, query_entities) -> str:
        """
        Build enriched context from RAG chunks + graph.

        Args:
            rag_chunks: Retrieved document chunks
            subgraph: Extracted KG subgraph
            query_entities: Entities from query

        Returns:
            Formatted context string for LLM
        """
        parts = []

        # Add RAG chunks
        if rag_chunks:
            for i, chunk in enumerate(rag_chunks):
                content = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
                parts.append(f"[Document Excerpt {i+1}]\n{content}")
        else:
            parts.append("[No relevant document excerpts found]")

        # Add graph context
        graph_context = self.graph_store.format_subgraph_context(subgraph)
        if graph_context:
            parts.append(graph_context)

        return "\n\n".join(parts)

    def _extract_rag_excerpts(self, rag_chunks, max_excerpt_length: int = 500):
        """Extract excerpts from RAG chunks and collect page numbers."""
        excerpts = []
        page_numbers = set()
        seen = set()  # Track (page_num, chunk_num) to avoid duplicates

        for chunk in rag_chunks:
            metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
            content = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)

            page_num = metadata.get('page_num', 0)
            chunk_num = metadata.get('chunk_num', 0)

            # Skip duplicates
            key = (page_num, chunk_num)
            if key in seen:
                continue
            seen.add(key)

            excerpt = {
                'page': metadata.get('page', 'unknown'),
                'page_num': page_num,
                'chunk_num': chunk_num,
                'excerpt': content[:max_excerpt_length],
                'source': 'rag',
                'full_length': len(content)
            }
            excerpts.append(excerpt)

            if page_num:
                page_numbers.add(page_num)

        return excerpts, page_numbers

    def _extract_kg_pages(self, subgraph):
        """Extract unique page numbers from KG subgraph edges."""
        page_numbers = set()

        for u, v, data in subgraph.edges(data=True):
            if 'page_num' in data:
                page_numbers.add(data['page_num'])

        return page_numbers

    def _resolve_page_path(self, page_num, metadata_source=None, collection_name=None):
        """
        Resolve file path for a given page number.

        Priority order:
        1. metadata_source (from RAG chunks) - most accurate
        2. source_directory (original input directory) - reliable
        3. Return None (no dangerous cross-collection glob fallback)

        Args:
            page_num: Page number to load
            metadata_source: File path from RAG chunk metadata (if available)
            collection_name: Unused (kept for backward compatibility)

        Returns:
            Path object if file found, None otherwise
        """
        from pathlib import Path

        # Try metadata source first (from RAG chunks)
        if metadata_source:
            path = Path(metadata_source)
            if path.exists():
                return path

        # Try source_directory (most reliable for KG-only pages)
        if self.source_directory:
            patterns = [
                f"{self.source_directory}/page{page_num:03d}.txt",
                f"{self.source_directory}/page{page_num:04d}.txt",
                f"{self.source_directory}/page_{page_num:03d}.txt",
            ]
            for pattern in patterns:
                path = Path(pattern)
                if path.exists():
                    return path

        # Return None instead of glob fallback to prevent cross-collection contamination
        # If we can't find the file with known paths, don't guess
        return None

    def _load_page_texts(self, page_numbers, rag_chunks):
        """Load full text for specified pages from original files."""
        from pathlib import Path

        # Build mapping: page_num -> source path from RAG chunks
        page_to_source = {}
        for chunk in rag_chunks:
            if hasattr(chunk, 'metadata'):
                metadata = chunk.metadata
                page_num = metadata.get('page_num')
                source = metadata.get('source')
                if page_num and source:
                    page_to_source[page_num] = source

        # Load each page
        source_pages = {}
        MAX_PAGE_SIZE = 10000

        for page_num in page_numbers:
            metadata_source = page_to_source.get(page_num)
            page_path = self._resolve_page_path(page_num, metadata_source)

            if page_path:
                try:
                    page_text = page_path.read_text(encoding='utf-8')

                    # Truncate if too large
                    if len(page_text) > MAX_PAGE_SIZE:
                        page_text = page_text[:MAX_PAGE_SIZE] + f"\n\n[TRUNCATED: Full page is {len(page_text)} chars]"

                    source_pages[page_num] = page_text
                except UnicodeDecodeError:
                    try:
                        page_text = page_path.read_text(encoding='latin-1')
                        source_pages[page_num] = page_text
                    except Exception as e:
                        source_pages[page_num] = f"[ENCODING ERROR: {e}]"
                except Exception as e:
                    source_pages[page_num] = f"[READ ERROR: {e}]"
            else:
                source_pages[page_num] = "[FILE NOT FOUND: Original page file has been deleted or moved]"

        return source_pages

    def _extract_source_information(self, rag_chunks, subgraph, collection_name=None):
        """Extract source excerpts and full page text from RAG chunks and KG subgraph."""
        # Extract RAG excerpts and page numbers
        rag_excerpts, rag_pages = self._extract_rag_excerpts(rag_chunks)

        # Extract KG page numbers
        kg_pages = self._extract_kg_pages(subgraph)

        # Track provenance: which pages came from RAG vs KG, and KG relationships per page
        page_provenance = {}

        # Mark RAG pages
        for page in rag_pages:
            page_provenance[page] = {'source': 'rag', 'kg_edges': []}

        # Count KG edges per page and mark KG-only pages
        for u, v, data in subgraph.edges(data=True):
            page_num = data.get('page_num')
            if page_num:
                if page_num not in page_provenance:
                    # This is a KG-only page (no RAG chunks)
                    page_provenance[page_num] = {'source': 'kg', 'kg_edges': []}

                # Add edge to this page's list (for verbose display)
                relation = data.get('relation', 'RELATED_TO')
                page_provenance[page_num]['kg_edges'].append((u, relation, v))

        # Merge page numbers (set union automatically deduplicates)
        all_pages = rag_pages | kg_pages

        # Load full page texts
        source_pages = self._load_page_texts(all_pages, rag_chunks) if all_pages else {}

        return rag_excerpts, source_pages, page_provenance

    def _print_timing_summary(self, timing: Dict[str, float]):
        """Print formatted timing summary."""
        print("\n[Hybrid] Timing Breakdown:")
        for operation, duration in timing.items():
            if operation != 'total':
                print(f"  {operation:25s} {duration:>6.3f}s")
        print(f"  {'─' * 32}")
        print(f"  {'Total':25s} {timing['total']:>6.3f}s")

    def find_connections(self, entity1: str, entity2: str,
                        max_depth: int = 5) -> Optional[Dict]:
        """
        Find connection path between two entities (multi-hop reasoning).

        Args:
            entity1: First entity
            entity2: Second entity
            max_depth: Maximum path length

        Returns:
            Dict with path info, or None if no connection
        """
        path = self.graph_store.find_path(entity1, entity2, max_depth)

        if not path:
            return None

        # Get relationships along the path
        relationships = []
        for i in range(len(path) - 1):
            edge_data = self.graph_store.graph.get_edge_data(path[i], path[i+1])
            if edge_data:
                rel = edge_data.get('relation', 'RELATED_TO')
                relationships.append(f"{path[i]} --[{rel}]--> {path[i+1]}")

        return {
            'path': path,
            'length': len(path) - 1,
            'relationships': relationships
        }

    def explain_entity(self, entity: str) -> str:
        """
        Get all known information about an entity from the graph.

        Args:
            entity: Entity name to look up

        Returns:
            Formatted string with entity information
        """
        neighbors = self.graph_store.get_entity_neighbors(entity)

        if not neighbors['outgoing'] and not neighbors['incoming']:
            return f"No information found for entity: {entity}"

        lines = [f"[Information about: {entity}]"]

        if neighbors['outgoing']:
            lines.append("  Outgoing relationships:")
            for rel in neighbors['outgoing'][:10]:
                lines.append(f"    → {rel}")

        if neighbors['incoming']:
            lines.append("  Incoming relationships:")
            for rel in neighbors['incoming'][:10]:
                lines.append(f"    ← {rel}")

        return "\n".join(lines)

    def compare_entities(self, entity1: str, entity2: str) -> Dict[str, Any]:
        """
        Compare two entities based on their graph relationships.

        Args:
            entity1: First entity
            entity2: Second entity

        Returns:
            Dict with comparison information
        """
        neighbors1 = self.graph_store.get_entity_neighbors(entity1)
        neighbors2 = self.graph_store.get_entity_neighbors(entity2)

        # Find shared relationships
        all_rels1 = set(neighbors1['outgoing'] + neighbors1['incoming'])
        all_rels2 = set(neighbors2['outgoing'] + neighbors2['incoming'])

        # Extract just the entity names from relationships
        entities1 = set()
        entities2 = set()

        for n in self.graph_store.graph.successors(entity1):
            entities1.add(n)
        for n in self.graph_store.graph.predecessors(entity1):
            entities1.add(n)
        for n in self.graph_store.graph.successors(entity2):
            entities2.add(n)
        for n in self.graph_store.graph.predecessors(entity2):
            entities2.add(n)

        shared = entities1 & entities2
        unique_to_1 = entities1 - entities2
        unique_to_2 = entities2 - entities1

        return {
            'entity1': entity1,
            'entity2': entity2,
            'shared_connections': list(shared),
            'unique_to_entity1': list(unique_to_1),
            'unique_to_entity2': list(unique_to_2),
            'entity1_total_connections': len(entities1),
            'entity2_total_connections': len(entities2)
        }

    def stats(self) -> Dict[str, Any]:
        """Get hybrid system statistics."""
        graph_stats = self.graph_store.stats()
        return {
            'graph': graph_stats,
            'known_entities': len(self._known_entities),
            'vector_store_type': type(self.vector_store).__name__,
            'llm_type': type(self.llm).__name__
        }
