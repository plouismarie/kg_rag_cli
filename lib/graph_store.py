"""
Persistent Graph Store with Subgraph Extraction
Uses NetworkX with GML persistence - no external dependencies

This module provides the core scalability feature:
- Extract SUBGRAPH (20-50 nodes) instead of full graph (10,000+ nodes)
- BFS-based N-hop neighbor discovery
- Entity indexing for fast lookup
"""
import networkx as nx
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
import time


class ScalableGraphStore:
    """
    Persistent knowledge graph with query-driven subgraph extraction.

    Key features:
    - Persists to GML file (compatible with existing graphs)
    - Extracts subgraph around query entities (20-50 nodes vs 10,000)
    - BFS-based N-hop neighbor discovery
    - Entity indexing for fast lookup

    Usage:
        store = ScalableGraphStore("./kg_store", "my_graph")
        store.add_triple("Paris", "IS_CAPITAL_OF", "France")
        subgraph = store.get_subgraph(["Paris"], hops=2, max_nodes=30)
    """

    def __init__(self, persist_path: str, graph_name: str = "knowledge_graph",
                 enable_entity_resolution: bool = True, dedup_mode: str = 'standard'):
        """
        Initialize graph store.

        Args:
            persist_path: Directory to store graph files
            graph_name: Name for the graph file (without extension)
            enable_entity_resolution: If True, deduplicate entities during insertion.
                                     Set False for backward compatibility.
            dedup_mode: Deduplication mode when entity_resolution is enabled:
                - 'standard': Full dedup (case + title + fuzzy matching)
                - 'light': Light dedup (case-insensitive + substring only)
                - 'none': No dedup (pass-through)
        """
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.graph_file = self.persist_path / f"{graph_name}.gml"
        self.graph_name = graph_name
        self.graph = nx.DiGraph()
        self._entity_index: Set[str] = set()

        # Entity resolution setup
        self.enable_entity_resolution = enable_entity_resolution
        self.dedup_mode = dedup_mode
        if enable_entity_resolution:
            # Import here to avoid circular imports
            from .entity_extractor import EntityResolver
            self.entity_resolver = EntityResolver(known_entities=set(), dedup_mode=dedup_mode)
        else:
            self.entity_resolver = None

        self._load_if_exists()

    def _load_if_exists(self):
        """Load existing graph AND entity resolver state from disk."""
        if self.graph_file.exists():
            print(f"[GraphStore] Loading from {self.graph_file}")
            try:
                self.graph = nx.read_gml(self.graph_file)
                self._entity_index = set(self.graph.nodes())

                # Deserialize node pages strings back to sets
                for node in self.graph.nodes():
                    if 'pages' in self.graph.nodes[node]:
                        pages_str = self.graph.nodes[node]['pages']
                        if isinstance(pages_str, str):
                            # Handle both empty and non-empty strings
                            if pages_str:
                                self.graph.nodes[node]['pages'] = {int(p) for p in pages_str.split(',') if p}
                            else:
                                # Empty string -> empty set
                                self.graph.nodes[node]['pages'] = set()
                        elif isinstance(pages_str, (int, float)):
                            # Single page stored as number
                            self.graph.nodes[node]['pages'] = {int(pages_str)}

                print(f"[GraphStore] Loaded {len(self._entity_index)} entities, "
                      f"{self.graph.number_of_edges()} edges")

                # Load entity resolver state if enabled
                if self.enable_entity_resolution and self.entity_resolver:
                    resolver_file = self.persist_path / f"{self.graph_name}_resolver.json"
                    if resolver_file.exists():
                        import json
                        try:
                            with open(resolver_file, 'r') as f:
                                resolver_state = json.load(f)
                            self.entity_resolver.canonical_map = resolver_state.get('canonical_map', {})
                            self.entity_resolver.aliases = resolver_state.get('aliases', {})
                            self.entity_resolver.entity_frequency = resolver_state.get('entity_frequency', {})
                            print(f"[GraphStore] Loaded resolver state: "
                                  f"{len(self.entity_resolver.canonical_map)} canonical entities")
                        except Exception as e:
                            print(f"[GraphStore] Warning: Could not load resolver state: {e}")
            except Exception as e:
                print(f"[GraphStore] Error loading graph: {e}")
                self.graph = nx.DiGraph()
                self._entity_index = set()

    def save(self):
        """Persist graph AND entity resolver state to disk."""
        # Serialize node pages sets to strings for GML compatibility
        # GML doesn't support Python sets, so convert to comma-separated strings
        for node in self.graph.nodes():
            if 'pages' in self.graph.nodes[node]:
                pages_set = self.graph.nodes[node]['pages']
                if isinstance(pages_set, set):
                    self.graph.nodes[node]['pages'] = ','.join(str(p) for p in sorted(pages_set))

        nx.write_gml(self.graph, self.graph_file)
        print(f"[GraphStore] Saved to {self.graph_file}")

        # Save entity resolver state if enabled
        if self.enable_entity_resolution and self.entity_resolver:
            resolver_file = self.persist_path / f"{self.graph_name}_resolver.json"
            import json
            try:
                resolver_state = {
                    'canonical_map': self.entity_resolver.canonical_map,
                    'aliases': self.entity_resolver.aliases,
                    'entity_frequency': self.entity_resolver.entity_frequency
                }
                with open(resolver_file, 'w') as f:
                    json.dump(resolver_state, f, indent=2)
                print(f"[GraphStore] Saved resolver state to {resolver_file}")
            except Exception as e:
                print(f"[GraphStore] Warning: Could not save resolver state: {e}")

    def save_to_gml(self, filename: str):
        """Save graph to GML file for compatibility with existing workflow."""
        nx.write_gml(self.graph, filename)
        print(f"[GraphStore] Saved to GML: {filename}")

    def load_from_gml(self, filename: str):
        """Load graph from GML file."""
        self.graph = nx.read_gml(filename)
        self._entity_index = set(self.graph.nodes())
        print(f"[GraphStore] Loaded from GML: {filename} ({self.graph.number_of_nodes()} nodes)")

    def add_triple(self, subject: str, predicate: str, obj: str,
                   metadata: Optional[Dict] = None, auto_save: bool = False):
        """
        Add a single triple to the graph with optional entity deduplication.

        If entity resolution is enabled:
        - "Paris", "paris", "Prince Paris" all resolve to "Paris"
        - Graph stays compact even with multi-page documents

        Args:
            subject: Source entity
            predicate: Relationship type
            obj: Target entity
            metadata: Optional additional attributes
            auto_save: If True, persist after each add (slower)
        """
        # Sanitize strings for GML compatibility
        subject = str(subject).strip()
        obj = str(obj).strip()
        predicate = str(predicate).strip()

        # Entity resolution: deduplicate entities before adding to graph
        if self.enable_entity_resolution and self.entity_resolver:
            subject_resolved = self.entity_resolver.resolve(subject, auto_register=True)
            obj_resolved = self.entity_resolver.resolve(obj, auto_register=True)

            # Track original mentions as metadata if they differ
            if subject_resolved != subject or obj_resolved != obj:
                if metadata is None:
                    metadata = {}
                else:
                    metadata = metadata.copy()  # Don't modify original

                if subject_resolved != subject:
                    metadata['subject_original'] = subject
                if obj_resolved != obj:
                    metadata['object_original'] = obj

            subject = subject_resolved
            obj = obj_resolved

        # Add to graph (now with deduplicated entities)
        self.graph.add_node(subject)
        self.graph.add_node(obj)
        self.graph.add_edge(subject, obj, relation=predicate, **(metadata or {}))
        self._entity_index.add(subject)
        self._entity_index.add(obj)

        # Track which pages each node appears in (for 3D visualization)
        page_num = metadata.get('page_num') if metadata else None
        if page_num is not None:
            # Initialize pages set if not present or wrong type
            if 'pages' not in self.graph.nodes[subject] or not isinstance(self.graph.nodes[subject].get('pages'), set):
                self.graph.nodes[subject]['pages'] = set()
            if 'pages' not in self.graph.nodes[obj] or not isinstance(self.graph.nodes[obj].get('pages'), set):
                self.graph.nodes[obj]['pages'] = set()
            # Add page to node's pages set
            self.graph.nodes[subject]['pages'].add(page_num)
            self.graph.nodes[obj]['pages'].add(page_num)

        if auto_save:
            self.save()

    def add_triples_batch(self, triples: List[Tuple[str, str, str]],
                          metadata: Optional[Dict] = None):
        """
        Add multiple triples efficiently.

        Args:
            triples: List of (subject, predicate, object) tuples
            metadata: Optional metadata to add to all triples
        """
        for subj, pred, obj in triples:
            self.add_triple(subj, pred, obj, metadata, auto_save=False)
        self.save()
        print(f"[GraphStore] Added {len(triples)} triples")

    def get_entities(self) -> Set[str]:
        """Get all entity names for matching."""
        return self._entity_index.copy()

    def get_deduplication_stats(self) -> Dict:
        """
        Get entity deduplication statistics.

        Returns:
            Dict with:
            - enabled: Whether entity resolution is enabled
            - total_entities: Number of canonical entities
            - total_aliases: Number of alias mappings
            - total_mentions: Total entity mentions resolved
            - deduplication_ratio: aliases / entities (higher = more merging)
            - top_merged: Most frequently merged entities with their usage counts
        """
        if not self.enable_entity_resolution or not self.entity_resolver:
            return {'enabled': False}

        total_mentions = sum(self.entity_resolver.entity_frequency.values())

        return {
            'enabled': True,
            'total_entities': len(self.entity_resolver.canonical_map),
            'total_aliases': len(self.entity_resolver.aliases),
            'total_mentions': total_mentions,
            'deduplication_ratio': (
                len(self.entity_resolver.aliases) / len(self.entity_resolver.canonical_map)
                if self.entity_resolver.canonical_map else 0
            ),
            'top_merged': sorted(
                self.entity_resolver.entity_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    def get_all_triples(self) -> List[Tuple[str, str, str]]:
        """Get all triples in the graph."""
        triples = []
        for u, v, data in self.graph.edges(data=True):
            relation = data.get('relation', 'RELATED_TO')
            triples.append((u, relation, v))
        return triples

    def get_subgraph(self, entities: List[str], hops: int = 2,
                     max_nodes: int = 50) -> nx.DiGraph:
        """
        Extract subgraph around given entities within N hops.

        This is the KEY SCALABILITY FEATURE:
        - Instead of loading 10,000 nodes, load only 20-50 relevant ones
        - Uses BFS to find neighbors within N hops
        - Limits total nodes to prevent context overflow

        Args:
            entities: Starting entities (from user query)
            hops: Number of hops to expand (1-2 recommended)
            max_nodes: Maximum nodes to return

        Returns:
            NetworkX DiGraph containing only the relevant subgraph
        """
        start_time = time.time()
        relevant_nodes = set()

        for entity in entities:
            # Try exact match first
            if entity not in self.graph:
                # Try case-insensitive match
                matched = False
                for node in self._entity_index:
                    if entity.lower() == node.lower():
                        entity = node
                        matched = True
                        break
                if not matched:
                    # Try partial match
                    for node in self._entity_index:
                        if entity.lower() in node.lower() or node.lower() in entity.lower():
                            entity = node
                            break

            if entity in self.graph:
                # BFS to find nodes within N hops
                visited = {entity}
                frontier = [entity]

                for hop in range(hops):
                    next_frontier = []
                    for node in frontier:
                        # Get both predecessors and successors (bidirectional)
                        neighbors = (set(self.graph.successors(node)) |
                                   set(self.graph.predecessors(node)))
                        for n in neighbors:
                            if n not in visited:
                                visited.add(n)
                                next_frontier.append(n)
                    frontier = next_frontier

                relevant_nodes.update(visited)

        # Limit nodes to prevent context overflow
        if len(relevant_nodes) > max_nodes:
            # Prioritize: keep query entities + closest neighbors
            priority_nodes = [e for e in entities if e in relevant_nodes]
            remaining = list(relevant_nodes - set(priority_nodes))[:max_nodes - len(priority_nodes)]
            relevant_nodes = set(priority_nodes + remaining)

        # Extract subgraph
        subgraph = self.graph.subgraph(relevant_nodes).copy()

        elapsed = time.time() - start_time
        print(f"[GraphStore] Extracted subgraph: {subgraph.number_of_nodes()} nodes, "
              f"{subgraph.number_of_edges()} edges in {elapsed:.3f}s")

        return subgraph

    def find_path(self, source: str, target: str, max_depth: int = 5) -> Optional[List[str]]:
        """
        Find shortest path between two entities (multi-hop reasoning).

        Args:
            source: Starting entity
            target: Target entity
            max_depth: Maximum path length

        Returns:
            List of entities in path, or None if no path exists
        """
        try:
            path = nx.shortest_path(self.graph, source, target)
            return path if len(path) <= max_depth else None
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_triples_from_subgraph(self, subgraph: nx.DiGraph) -> List[Tuple[str, str, str]]:
        """Convert subgraph to list of triples for context."""
        triples = []
        for u, v, data in subgraph.edges(data=True):
            relation = data.get('relation', 'RELATED_TO')
            triples.append((u, relation, v))
        return triples

    def format_subgraph_context(self, subgraph: nx.DiGraph, max_triples: int = 20) -> str:
        """
        Format subgraph as text context for LLM.

        Args:
            subgraph: The extracted subgraph
            max_triples: Maximum number of relationships to include

        Returns:
            Formatted string for LLM context
        """
        triples = self.get_triples_from_subgraph(subgraph)[:max_triples]

        if not triples:
            return ""

        lines = ["[Knowledge Graph Context]"]
        for subj, pred, obj in triples:
            lines.append(f"  {subj} --[{pred}]--> {obj}")

        lines.append(f"[{subgraph.number_of_nodes()} entities, "
                    f"{subgraph.number_of_edges()} relationships]")
        return "\n".join(lines)

    def get_entity_neighbors(self, entity: str) -> Dict[str, List[str]]:
        """
        Get all neighbors of an entity with their relationships.

        Args:
            entity: Entity to look up

        Returns:
            Dict with 'outgoing' and 'incoming' relationship lists
        """
        result = {'outgoing': [], 'incoming': []}

        if entity not in self.graph:
            return result

        # Outgoing edges
        for neighbor in self.graph.successors(entity):
            edge_data = self.graph.get_edge_data(entity, neighbor)
            relation = edge_data.get('relation', 'RELATED_TO')
            result['outgoing'].append(f"{relation} -> {neighbor}")

        # Incoming edges
        for neighbor in self.graph.predecessors(entity):
            edge_data = self.graph.get_edge_data(neighbor, entity)
            relation = edge_data.get('relation', 'RELATED_TO')
            result['incoming'].append(f"{neighbor} -> {relation}")

        return result

    def stats(self) -> Dict:
        """Get graph statistics."""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'persist_path': str(self.graph_file),
            'is_connected': nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        }

    def get_entity_pages(self) -> Dict[str, Set[int]]:
        """
        Get mapping of entity -> set of page numbers.

        Used by 3D visualization to identify which pages each entity appears on.

        Returns:
            Dict mapping entity names to sets of page numbers
        """
        entity_pages = {}
        for node in self.graph.nodes():
            pages = self.graph.nodes[node].get('pages', set())
            if isinstance(pages, str):
                # Handle case where pages is still serialized string
                pages = {int(p) for p in pages.split(',') if p}
            elif isinstance(pages, (int, float)):
                pages = {int(pages)}
            elif not isinstance(pages, set):
                pages = set()
            if pages:
                entity_pages[node] = pages
        return entity_pages

    @classmethod
    def from_existing_gml(cls, gml_path: str) -> "ScalableGraphStore":
        """
        Load from an existing GML file (migrate from old system).

        Args:
            gml_path: Path to existing GML file

        Returns:
            New ScalableGraphStore instance with loaded graph
        """
        path = Path(gml_path)
        store = cls(persist_path=str(path.parent), graph_name=path.stem)
        return store

    @classmethod
    def from_langchain_graph(cls, langchain_graph, persist_path: str,
                             graph_name: str) -> "ScalableGraphStore":
        """
        Migrate from LangChain NetworkxEntityGraph.

        Args:
            langchain_graph: LangChain NetworkxEntityGraph instance
            persist_path: Where to store the new graph
            graph_name: Name for the new graph

        Returns:
            New ScalableGraphStore with migrated data
        """
        store = cls(persist_path=persist_path, graph_name=graph_name)

        # Get triples from LangChain graph
        triples = langchain_graph.get_triples()
        for subj, obj, pred in triples:  # Note: LangChain uses (subj, obj, pred) order
            store.add_triple(subj, pred, obj, auto_save=False)

        store.save()
        print(f"[GraphStore] Migrated {len(triples)} triples from LangChain graph")
        return store
