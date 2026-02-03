"""
Knowledge Graph RAG CLI - Version 3 (Verbose Extraction)

A command-line tool for building and querying knowledge graphs from text documents.
Uses LLM-based triple extraction with automatic inverse relation learning.

VERSION 3 FEATURES:
- Verbose extraction mode (default): Extracts more entities including descriptive
  phrases, concepts, emotional states, and longer entities
- Entity deduplication disabled by default for maximum graph verbosity
- Uses enhanced prompts that encourage comprehensive extraction

Usage:
    # Index a document with verbose extraction (default)
    python kg_rag_cli_v3.py --mode index --input book.txt

    # Index a multi-page document
    python kg_rag_cli_v3.py --mode index --input-dir Input/MultiPage/TheVeldt/

    # Query an existing index
    python kg_rag_cli_v3.py --mode query --input book.txt --question "Who is the main character?"

Author: Plm and Claude
License: MIT
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

# =============================================================================
# LangChain imports
# =============================================================================

from langchain_community.graphs import NetworkxEntityGraph
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader

# =============================================================================
# Local lib imports (for multi-page and entity resolution support)
# =============================================================================

from lib import (
    ScalableGraphStore,
    HybridDocumentProcessor,
    ScalableHybridRAG,
    IncrementalKGBuilder
)

# =============================================================================
# Constants
# =============================================================================

# Directory where graphs are stored
STORE_DIR = Path("kg_store")

# Default model configuration
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 1


# =============================================================================
# Argument Parser
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for the CLI.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Knowledge Graph RAG: Build and query knowledge graphs from text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single-file mode:
    Index a document:
      python kg_rag_cli.py --mode index --input book.txt

    Query an index:
      python kg_rag_cli.py --mode query --input book.txt --question "Who is the main character?"

    Index without visualization:
      python kg_rag_cli.py --mode index --input book.txt --no-visualize

  Multi-page mode:
    Index a directory (collection auto-derived from path):
      python kg_rag_cli.py --mode index --input-dir Input/MultiPage/Ulysses/

    Index with explicit collection name:
      python kg_rag_cli.py --mode index --input-dir Input/MultiPage/Ulysses/ --collection ulysses_multi

    Force rebuild (delete old index before re-indexing):
      python kg_rag_cli.py --mode index --input-dir Input/MultiPage/Ulysses/ --force-rebuild

    Query a multi-page index (same collection/path as indexing):
      python kg_rag_cli.py --mode query --input-dir Input/MultiPage/Ulysses/ --question "Who is Paris?"
      python kg_rag_cli.py --mode query --collection ulysses_multi --question "Who is Paris?"

    Disable entity deduplication:
      python kg_rag_cli.py --mode index --input-dir book_pages/ --no-entity-resolution
        """
    )

    # Required arguments
    parser.add_argument(
        "--mode", "-m",
        required=True,
        choices=["index", "query"],
        help="Mode: 'index' to build graph, 'query' to ask questions"
    )

    # Input source arguments (one of --input or --input-dir is required)
    parser.add_argument(
        "--input", "-i",
        help="Single-file mode: path to text file (index) or filename of the index (query)"
    )
    parser.add_argument(
        "--input-dir",
        help="Multi-page mode: directory containing page files"
    )
    parser.add_argument(
        "--file-pattern",
        default="*.txt",
        help="Glob pattern for page files in multi-page mode (default: *.txt)"
    )
    parser.add_argument(
        "--collection", "-c",
        help="Collection name for multi-page indexes (optional: auto-derived from directory path if not specified)"
    )

    # Query mode arguments
    parser.add_argument(
        "--question", "-q",
        help="Question to ask (required for query mode)"
    )

    # Optional arguments
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild: delete existing index before re-indexing (index mode only)"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip HTML visualization generation (index mode only)"
    )
    parser.add_argument(
        "--no-entity-resolution",
        action="store_true",
        help="Disable entity deduplication completely (all entity variations stay separate)"
    )
    parser.add_argument(
        "--enable-entity-resolution-light",
        action="store_true",
        default=True,  # V3: Light dedup is now the default
        help="Enable light entity deduplication (case-insensitive + substring matching) [DEFAULT]"
    )
    parser.add_argument(
        "--enable-entity-resolution",
        action="store_true",
        help="Enable full entity deduplication (case + title + fuzzy matching)"
    )
    parser.add_argument(
        "--verbose-extraction",
        action="store_true",
        default=True,  # V3: Enabled by default
        help="Use verbose extraction prompts for more comprehensive graphs (default in v3: enabled)"
    )
    parser.add_argument(
        "--compact-extraction",
        action="store_true",
        help="Use compact extraction (disable verbose mode) - extracts only proper nouns"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--viz3d",
        action="store_true",
        help="Generate 3D multi-page visualization (multi-page mode only, requires plotly)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Minimum similarity score for RAG chunks (0.0-1.0, default: 0.75). "
             "Lower values retrieve more chunks, higher values are more selective. "
             "Uses adaptive retry: 0.75 → 0.50 → 0.25 → 0.10"
    )

    return parser


# =============================================================================
# LLM Initialization
# =============================================================================

def get_llm() -> ChatOpenAI:
    """
    Initialize and return the LLM instance.

    Returns:
        ChatOpenAI: Configured LLM instance

    Raises:
        SystemExit: If OPENAI_API_KEY is not set
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("\nPlease set it with:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)

    return ChatOpenAI(
        temperature=DEFAULT_TEMPERATURE,
        api_key=api_key,
        model=DEFAULT_MODEL
    )


# =============================================================================
# Helper Functions
# =============================================================================

import re


def extract_json_from_response(text: str) -> dict:
    """
    Extract JSON from LLM response, handling various formats.

    The LLM may return:
    - Pure JSON: {"key": "value"}
    - Markdown code block: ```json\n{"key": "value"}\n```
    - Text with embedded JSON: "Here is the result: {"key": "value"}"

    Args:
        text: Raw text response from LLM

    Returns:
        dict: Parsed JSON object

    Raises:
        ValueError: If no valid JSON found
    """
    if not text or not text.strip():
        raise ValueError("Empty response from LLM")

    text = text.strip()

    # Try 1: Direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try 2: Extract from markdown code block (```json ... ``` or ``` ... ```)
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try 3: Find JSON object pattern in text
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not extract valid JSON from response: {text[:200]}...")


def compute_all_inverse_relations(llm, predicates: list) -> dict:
    """
    Compute inverse relations for multiple predicates in a single LLM call.

    This batches all predicates into one API call instead of making
    separate calls for each predicate.

    Args:
        llm: The language model instance
        predicates: List of predicate strings (e.g., ["HAS_SON", "KNOWS"])

    Returns:
        dict: Mapping from each predicate to its inverse
              (e.g., {"HAS_SON": "HAS_FATHER", "KNOWS": "KNOWN_BY"})
    """
    if not predicates:
        return {}

    # Format predicates as a bulleted list
    predicates_list = "\n".join([f"- {p}" for p in predicates])

    prompt = f"""For each relationship label below, output the inverse relationship label in uppercase that expresses the same connection from the target to the source.

Input predicates:
{predicates_list}

Output as JSON object where keys are the input predicates and values are their inverses.
Example: {{"HAS_SON": "HAS_FATHER", "KNOWS": "KNOWN_BY"}}

Output only valid JSON, no markdown formatting or additional text:"""

    response = llm.invoke(prompt)

    try:
        result = extract_json_from_response(response.content)
    except ValueError as e:
        print(f"Warning: Failed to parse LLM response: {e}")
        print("Falling back to default inverse naming...")
        # Fallback: generate default inverse names
        result = {p: f"INVERSE_{p}" for p in predicates}

    return result


def ensure_store_directory():
    """Create the storage directory if it doesn't exist."""
    STORE_DIR.mkdir(exist_ok=True)


def sanitize_path_for_index(path: str) -> str:
    """
    Convert file path to safe index name for storage.

    Removes leading ./ or .\ and replaces path separators with underscores.

    Args:
        path: File path (e.g., "Input/SinglePage/Ulysses.txt")

    Returns:
        str: Sanitized name (e.g., "Input_SinglePage_Ulysses.txt")

    Examples:
        "Input/SinglePage/Ulysses.txt" → "Input_SinglePage_Ulysses.txt"
        "./book.txt" → "book.txt"
        "book.txt" → "book.txt"
    """
    # Remove leading ./ or .\
    path = path.lstrip('./\\')
    # Replace path separators with underscores
    sanitized = path.replace('/', '_').replace('\\', '_')
    return sanitized


def get_index_paths(index_name: str) -> tuple:
    """
    Get the file paths for a given index name.

    Sanitizes the input path to create safe filenames.

    Args:
        index_name: Name/path of the index (can be full path like "Input/SinglePage/Ulysses.txt")

    Returns:
        tuple: (gml_path, html_path)
    """
    # Sanitize the index name for filesystem safety
    safe_name = sanitize_path_for_index(index_name)
    gml_path = STORE_DIR / f"{safe_name}.gml"
    html_path = STORE_DIR / f"{safe_name}.html"
    return gml_path, html_path


def _sanitize_graph_for_pyvis(graph):
    """Convert non-JSON-serializable node/edge attributes to JSON-safe types."""
    import networkx as nx

    # Create deep copy to avoid modifying original
    sanitized = graph.copy()

    # Convert sets to sorted lists in node attributes
    for node in sanitized.nodes():
        for attr_name, attr_value in list(sanitized.nodes[node].items()):
            if isinstance(attr_value, set):
                sanitized.nodes[node][attr_name] = sorted(list(attr_value))

    # Convert sets to lists in edge attributes (if any)
    for u, v, data in sanitized.edges(data=True):
        for attr_name, attr_value in list(data.items()):
            if isinstance(attr_value, set):
                sanitized[u][v][attr_name] = sorted(list(attr_value))

    return sanitized


def visualize_graph_smart(graph, output_path: str, max_nodes: int = None,
                          verbose: bool = False) -> None:
    """
    Smart visualization that adapts to graph size.

    Automatically selects the best visualization strategy based on number of nodes:
    - < 50 nodes: Standard PyVis layout
    - 50-150 nodes: Community detection with color coding
    - 150-300 nodes: Community detection + top 100 nodes by centrality
    - 300+ nodes: Statistical dashboard (no graph visualization)

    Args:
        graph: NetworkxEntityGraph or NetworkX graph object
        output_path: Path to save HTML file
        max_nodes: Optional override for node limit (not currently used)
        verbose: Print detailed visualization info

    Example:
        visualize_graph_smart(graph, "output.html", verbose=True)
    """
    from pyvis.network import Network
    import networkx as nx

    # Extract NetworkX graph if wrapped
    if hasattr(graph, '_graph'):
        G = graph._graph
    else:
        G = graph

    # Sanitize node attributes for JSON serialization (PyVis compatibility)
    G = _sanitize_graph_for_pyvis(G)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Strategy 1: SMALL graphs (< 50 nodes) - Standard visualization
    if num_nodes < 50:
        if verbose:
            print(f"  [Visualization] Standard layout ({num_nodes} nodes)")

        net = Network(notebook=True, cdn_resources='remote',
                     height='750px', width='100%')
        net.from_nx(G)
        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "barnesHut": {
                    "gravitationalConstant": -8000,
                    "springLength": 250
                }
            }
        }
        """)
        net.show(str(output_path))

    # Strategy 2: MEDIUM graphs (50-150 nodes) - Community detection
    elif num_nodes < 150:
        if verbose:
            print(f"  [Visualization] Community detection ({num_nodes} nodes)")

        # Detect communities using NetworkX built-in greedy modularity
        from networkx.algorithms import community as nx_comm
        communities = list(nx_comm.greedy_modularity_communities(G))

        # Color palette for communities
        COLORS = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                  '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']

        # Map nodes to communities
        node_to_community = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                node_to_community[node] = idx

        # Create visualization with community colors
        net = Network(notebook=True, cdn_resources='remote',
                     height='750px', width='100%')
        net.from_nx(G)

        # Apply community colors to nodes
        for node in net.nodes:
            comm_id = node_to_community.get(node['id'], 0)
            node['color'] = COLORS[comm_id % len(COLORS)]
            node['title'] = f"{node['id']} (Community {comm_id + 1})"

        net.set_options("""
        {
            "physics": {"enabled": true, "barnesHut": {"gravitationalConstant": -10000}}
        }
        """)
        net.show(str(output_path))

        if verbose:
            print(f"    Communities detected: {len(communities)}")

    # Strategy 3: LARGE graphs (150-300 nodes) - Top nodes + communities
    elif num_nodes < 300:
        if verbose:
            print(f"  [Visualization] Top nodes + communities ({num_nodes} → 100 nodes)")

        # Compute centrality and filter to top 100 nodes
        centrality = nx.degree_centrality(G)
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:100]
        top_node_ids = [node for node, _ in top_nodes]
        subgraph = G.subgraph(top_node_ids)

        # Detect communities on the subgraph
        from networkx.algorithms import community as nx_comm
        communities = list(nx_comm.greedy_modularity_communities(subgraph))

        COLORS = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231']
        node_to_community = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                node_to_community[node] = idx

        # Visualize subgraph
        net = Network(notebook=True, cdn_resources='remote',
                     height='750px', width='100%')
        net.from_nx(subgraph)

        # Size by centrality, color by community
        for node in net.nodes:
            node_id = node['id']
            node['size'] = centrality[node_id] * 50 + 10
            comm_id = node_to_community.get(node_id, 0)
            node['color'] = COLORS[comm_id % len(COLORS)]
            node['title'] = f"{node_id}<br>Centrality: {centrality[node_id]:.3f}<br>Community {comm_id + 1}"

        net.show(str(output_path))

    # Strategy 4: MASSIVE graphs (300+ nodes) - Statistical dashboard
    else:
        if verbose:
            print(f"  [Visualization] Statistical dashboard ({num_nodes} nodes - too large)")

        # Compute centrality for top entities
        centrality = nx.degree_centrality(G)
        top_10 = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]

        # Detect communities
        from networkx.algorithms import community as nx_comm
        communities = list(nx_comm.greedy_modularity_communities(G))

        # Generate HTML statistics dashboard
        html = f"""
        <!DOCTYPE html>
        <html><head><title>Graph Statistics</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 3px solid #2196F3; padding-bottom: 10px; }}
            h2 {{ color: #555; margin-top: 30px; }}
            .stat {{ margin: 20px 0; padding: 15px; background: #e8f4f8; border-radius: 5px; }}
            .stat-label {{ font-weight: bold; color: #555; font-size: 14px; }}
            .stat-value {{ font-size: 28px; color: #2196F3; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #2196F3; color: white; font-weight: bold; }}
            tr:hover {{ background: #f5f5f5; }}
            .note {{ color: #666; font-style: italic; margin-top: 10px; }}
        </style>
        </head><body>
        <div class="container">
            <h1>Knowledge Graph Statistics</h1>
            <p class="note">Graph too large for visualization ({num_nodes} nodes). Showing statistical summary.</p>

            <div class="stat">
                <div class="stat-label">Total Entities</div>
                <div class="stat-value">{num_nodes:,}</div>
            </div>

            <div class="stat">
                <div class="stat-label">Total Relationships</div>
                <div class="stat-value">{num_edges:,}</div>
            </div>

            <div class="stat">
                <div class="stat-label">Communities Detected</div>
                <div class="stat-value">{len(communities)}</div>
            </div>

            <div class="stat">
                <div class="stat-label">Average Connections per Entity</div>
                <div class="stat-value">{num_edges / num_nodes:.1f}</div>
            </div>

            <h2>Top 10 Most Connected Entities</h2>
            <table>
                <tr><th>Rank</th><th>Entity</th><th>Connections</th></tr>
                {''.join(f'<tr><td>{i+1}</td><td><strong>{node}</strong></td><td>{int(cent * num_nodes)}</td></tr>'
                         for i, (node, cent) in enumerate(top_10))}
            </table>

            <h2>Community Sizes</h2>
            <table>
                <tr><th>Community</th><th>Members</th></tr>
                {''.join(f'<tr><td>Community {i+1}</td><td>{len(comm)}</td></tr>'
                         for i, comm in enumerate(sorted(communities, key=len, reverse=True)[:10]))}
            </table>
        </div>
        </body></html>
        """

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        if verbose:
            print(f"    Top entity: {top_10[0][0]} ({int(top_10[0][1] * num_nodes)} connections)")


# =============================================================================
# Index Command
# =============================================================================

def index_document(input_path: str, visualize: bool = True, verbose: bool = False,
                   enable_entity_resolution: bool = True, verbose_mode: bool = True,
                   dedup_mode: str = 'light'):
    """
    Build a knowledge graph from a text file and save it.

    V3: Light entity deduplication enabled by default, verbose extraction enabled by default.

    Args:
        input_path: Path to the input text file
        visualize: Whether to generate HTML visualization
        verbose: Enable verbose output
        enable_entity_resolution: Enable entity deduplication (default: True in v3)
        verbose_mode: Use verbose extraction prompts (default: True in v3)
        dedup_mode: Deduplication mode: 'none', 'light' (default), or 'standard'
    """
    # Start total timing
    total_start = time.time()

    input_file = Path(input_path)

    # Validate input file exists
    if not input_file.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    # Use the full input path as the index name
    # (get_index_paths will sanitize it for safe storage)
    index_name = input_path

    print(f"Loading document: {input_path}")

    # Initialize LLM
    llm = get_llm()

    # Load the document
    loader = TextLoader(str(input_file))
    documents = loader.load()
    text = documents[0].page_content

    # ==========================================================================
    # Phase 1: Knowledge Extraction Using IncrementalKGBuilder
    # ==========================================================================
    # Note: We use IncrementalKGBuilder instead of GraphIndexCreator because
    # GraphIndexCreator.from_text() has issues in LangChain 0.3.0+
    # IncrementalKGBuilder is the proven implementation used in multi-page mode

    print("Extracting knowledge triples...")
    phase1_start = time.time()

    # Create temporary graph store for single-file indexing
    # We'll extract the graph afterward for compatibility with existing code
    from lib.graph_store import ScalableGraphStore

    # Use a temporary in-memory collection name
    temp_collection = sanitize_path_for_index(index_name).replace('.', '_')
    graph_store = ScalableGraphStore(
        persist_path=str(STORE_DIR / "_temp"),
        graph_name=temp_collection,
        enable_entity_resolution=enable_entity_resolution,
        dedup_mode=dedup_mode
    )

    # Build KG using proven incremental builder
    builder = IncrementalKGBuilder(
        llm=llm,
        graph_store=graph_store,
        chunk_size=500,
        chunk_overlap=50,
        auto_detect_type=False,  # Skip auto-detection for single files
        batch_size=4,  # Process multiple chunks per LLM call
        verbose_mode=verbose_mode,  # V3: verbose extraction enabled by default
        debug=verbose  # Show detailed output in verbose mode
    )

    # Extract triples from text (this also handles inverse relations automatically)
    stats = builder.build_from_text(text, add_inverse=True)

    phase1_time = time.time() - phase1_start

    # Wrap the NetworkX graph in NetworkxEntityGraph for compatibility
    # with existing visualization and save code
    graph = NetworkxEntityGraph(graph=graph_store.graph)

    if verbose:
        print(f"  Extraction time: {phase1_time:.1f}s")
        print(f"  Chunks processed: {stats.get('chunks_processed', 0)}")
        print(f"  Triples extracted: {stats.get('total_triples', 0)}")
        print("\nExtracted triples (first 10):")
        for i, (s, p, o) in enumerate(list(graph.get_triples())[:10]):
            print(f"  ({s}, {p}, {o})")
        if len(list(graph.get_triples())) > 10:
            print(f"  ... and {len(list(graph.get_triples())) - 10} more")

    # ==========================================================================
    # Phase 2: Save the Graph
    # ==========================================================================

    ensure_store_directory()
    gml_path, html_path = get_index_paths(index_name)

    # Save to GML format
    graph.write_to_gml(str(gml_path))
    print(f"Saved graph: {gml_path}")

    # ==========================================================================
    # Phase 3: Generate Visualization (Optional)
    # ==========================================================================

    if visualize:
        try:
            visualize_graph_smart(graph, html_path, verbose=verbose)
            print(f"✓ Visualization saved: {html_path}")
            print(f"📊 View in browser: open {html_path}")
        except ImportError:
            print("Warning: pyvis not installed. Skipping visualization.")
            print("Install with: pip install pyvis")

    print(f"\nIndex '{index_name}' created successfully!")
    print(f"  Nodes: {graph.get_number_of_nodes()}")
    print(f"  Triples: {len(list(graph.get_triples()))}")

    # Show total time in verbose mode
    total_time = time.time() - total_start
    if verbose:
        print(f"  Total time: {total_time:.1f}s")


# =============================================================================
# Multi-Page Index Command
# =============================================================================

def index_directory(directory: str, collection_name: str, file_pattern: str = "*.txt",
                    enable_entity_resolution: bool = True, visualize: bool = True,
                    verbose: bool = False, viz3d: bool = False, force_rebuild: bool = False,
                    verbose_mode: bool = True, dedup_mode: str = 'light'):
    """
    Index a directory of page files using HybridDocumentProcessor.

    Creates both:
    - RAG vector store for semantic search
    - Knowledge graph for relationship queries

    V3: Light entity deduplication enabled by default, verbose extraction enabled by default.

    Args:
        directory: Path to directory containing page files
        collection_name: Collection name for storage
        file_pattern: Glob pattern for page files (default: "*.txt")
        enable_entity_resolution: Enable entity deduplication (default: True in v3)
        visualize: Generate visualization (default: True, but skipped for multi-page)
        verbose: Enable verbose output
        viz3d: Generate 3D visualization
        force_rebuild: Delete existing index before rebuilding (default: False)
        verbose_mode: Use verbose extraction prompts (default: True in v3)
        dedup_mode: Deduplication mode: 'none', 'light' (default), or 'standard'
    """
    total_start = time.time()

    print(f"\n{'='*60}")
    print("MULTI-PAGE INDEXING MODE (V3 - Verbose Extraction)")
    print(f"{'='*60}")
    print(f"Directory: {directory}")
    print(f"Collection: {collection_name}")
    print(f"File pattern: {file_pattern}")
    print(f"Verbose extraction: {'enabled' if verbose_mode else 'disabled'}")
    print(f"Entity resolution: {'enabled' if enable_entity_resolution else 'disabled'}")
    print(f"Dedup mode: {dedup_mode}")
    print(f"Force rebuild: {'enabled' if force_rebuild else 'disabled'}")
    print(f"{'='*60}\n")

    # Get LLM instance
    llm = get_llm()

    # Get embeddings (need to initialize - for now using OpenAI embeddings)
    # TODO: Make this configurable or use HuggingFace embeddings
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()

    # Create hybrid document processor
    processor = HybridDocumentProcessor(
        llm=llm,
        embeddings=embeddings,
        persist_path=f"./kg_store/{collection_name}",
        chunk_size=500,
        chunk_overlap=50
    )

    # Define progress callback
    def progress_callback(stage, current, total, stats):
        if stage == 'rag_indexing':
            print(f"  [RAG] Processing page {current}/{total}...")
        elif stage == 'kg_building':
            print(f"  [KG] Extracting triples from page {current}/{total}...")

    try:
        # Process directory
        print("\n[1/2] Building RAG vector store and knowledge graph...")
        hybrid = processor.process_directory(
            directory=directory,
            collection_name=collection_name,
            file_pattern=file_pattern,
            progress_callback=progress_callback if verbose else None,
            verbose=verbose,
            force_rebuild=force_rebuild,
            verbose_mode=verbose_mode,
            enable_entity_resolution=enable_entity_resolution,
            dedup_mode=dedup_mode
        )

        # Get statistics
        stats = processor.last_stats
        if stats:
            print(f"\n[Processing Complete]")
            print(f"  Pages processed: {stats.get('pages', 0)}")
            print(f"  RAG chunks: {stats.get('rag_chunks', 0)}")
            print(f"  KG triples: {stats.get('kg_triples', 0)}")
            print(f"  KG entities: {stats.get('kg_entities', 0)}")
            print(f"  RAG time: {stats.get('rag_time', 0):.1f}s")
            print(f"  KG time: {stats.get('kg_time', 0):.1f}s")

        # Show entity deduplication stats if enabled
        if enable_entity_resolution:
            print("\n[2/2] Entity Deduplication Statistics...")
            dedup_stats = hybrid.graph_store.get_deduplication_stats()
            if dedup_stats.get('enabled'):
                print(f"  Canonical entities: {dedup_stats['total_entities']}")
                print(f"  Total aliases: {dedup_stats['total_aliases']}")
                print(f"  Total mentions: {dedup_stats.get('total_mentions', 0)}")
                print(f"  Deduplication ratio: {dedup_stats['deduplication_ratio']:.2f}")

                if dedup_stats.get('top_merged'):
                    print("  Most merged entities:")
                    for entity, count in dedup_stats['top_merged'][:5]:
                        print(f"    - {entity}: {count} mentions")

        # Note: Multi-page visualization is skipped during indexing
        # Visualization will be generated during query based on query results
        if visualize:
            print("\n[Visualization]")

            # Check if 3D visualization requested
            if viz3d:
                try:
                    from lib.viz3d import visualize_graph_3d

                    viz_path = f"./kg_store/{collection_name}_3d.html"
                    success = visualize_graph_3d(
                        graph=hybrid.graph_store.graph,
                        output_path=viz_path,
                        title=f"Knowledge Graph: {collection_name}",
                        entity_resolver=hybrid.graph_store.entity_resolver if hasattr(hybrid.graph_store, 'entity_resolver') else None
                    )
                    if success:
                        from pathlib import Path
                        full_path = Path(viz_path).resolve()
                        print(f"  ✓ 3D visualization saved: {viz_path}")
                        print(f"  📊 View in browser: open {viz_path}")
                        print(f"  📁 Full path: {full_path}")
                except ImportError:
                    print("  Warning: plotly not installed. Skipping 3D visualization.")
                    print("  Install with: pip install plotly")
                except Exception as e:
                    print(f"  Warning: 3D visualization failed: {e}")
            else:
                print("  Note: Full graph visualization skipped for multi-page documents.")
                print("  Use --viz3d flag for 3D multi-page visualization.")
                print("  Query-specific visualizations will be generated during queries.")

        total_time = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"Index '{collection_name}' created successfully!")
        print(f"  Total time: {total_time:.1f}s")
        print(f"{'='*60}\n")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during indexing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# Query Command - Helper Functions
# =============================================================================

def get_subgraph_triples(graph, entities: list, hops: int = 2, max_triples: int = 100) -> list:
    """
    Extract triples from the graph around specified entities using BFS.

    Args:
        graph: NetworkxEntityGraph instance
        entities: List of entity names to start from
        hops: Number of hops to traverse (default: 2)
        max_triples: Maximum number of triples to return (default: 100)

    Returns:
        List of (subject, predicate, object) tuples
    """
    import networkx as nx
    from collections import deque

    # Get the underlying NetworkX graph
    G = graph._graph if hasattr(graph, '_graph') else graph

    # BFS to find all nodes within N hops of seed entities
    visited = set()
    queue = deque()

    # Initialize queue with seed entities
    for entity in entities:
        if entity in G.nodes():
            queue.append((entity, 0))
            visited.add(entity)

    # BFS traversal
    while queue:
        node, depth = queue.popleft()

        if depth < hops:
            # Explore outgoing edges
            for neighbor in G.successors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

            # Explore incoming edges
            for neighbor in G.predecessors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

    # Extract triples involving visited nodes
    triples = []
    for u, v, data in G.edges(data=True):
        if u in visited or v in visited:
            predicate = data.get('relation', data.get('label', 'RELATED_TO'))
            triples.append((u, predicate, v))

            if len(triples) >= max_triples:
                break

    return triples


def format_kg_context(triples: list) -> str:
    """
    Format knowledge graph triples as readable context for LLM.

    Args:
        triples: List of (subject, predicate, object) tuples

    Returns:
        Formatted string with relationships
    """
    if not triples:
        return "[No relevant knowledge found in the graph]"

    lines = []
    for subj, pred, obj in triples:
        # Make predicates more readable
        readable_pred = pred.replace('_', ' ').lower()
        lines.append(f"- {subj} {readable_pred} {obj}")

    return "\n".join(lines)


# =============================================================================
# Query Command
# =============================================================================

def query_graph(index_name: str, question: str, verbose: bool = False):
    """
    Load a knowledge graph and answer a question.

    Uses enhanced EntityExtractor for better entity matching from questions.

    Args:
        index_name: Path/name of the index to query (same as used for indexing)
        question: The question to ask
        verbose: Enable verbose output
    """
    gml_path, _ = get_index_paths(index_name)

    # Check if index exists
    if not gml_path.exists():
        print(f"Error: Index not found for '{index_name}'.")
        print(f"  Expected file: {gml_path}")
        print(f"\nPlease build the index first with:")
        print(f"  python kg_rag_cli.py --mode index --input {index_name}")
        sys.exit(1)

    print(f"Loading graph: {gml_path}")

    # Load the graph from GML file
    graph = NetworkxEntityGraph.from_gml(str(gml_path))

    if verbose:
        print(f"  Nodes: {graph.get_number_of_nodes()}")
        print(f"  Triples: {len(list(graph.get_triples()))}")

    # NEW: Enhanced entity extraction using EntityExtractor
    # Get all graph node labels as known entities for matching
    from lib.entity_extractor import EntityExtractor

    # Get LLM FIRST (needed for EntityExtractor initialization)
    llm = get_llm()

    # Get all graph node labels as known entities for matching
    graph_entities = list(graph._graph.nodes())  # Get all node labels from NetworkX graph
    entity_extractor = EntityExtractor(llm=llm, known_entities=graph_entities)

    # Extract entities from question using 3-tier strategy (pattern + fuzzy + LLM)
    extracted_entities = entity_extractor.extract(question, use_llm=True)

    if verbose:
        print(f"\nEntity Extraction:")
        print(f"  Known entities in graph: {len(graph_entities)}")
        print(f"  Entities extracted from question: {extracted_entities if extracted_entities else 'NONE'}")

    # Extract subgraph around matched entities (use all graph triples if no entities found)
    if extracted_entities:
        subgraph_triples = get_subgraph_triples(graph, extracted_entities, hops=2, max_triples=100)
    else:
        # Fallback: use all triples from the graph (limited)
        all_triples = list(graph.get_triples())[:50]
        subgraph_triples = all_triples
        if verbose:
            print("  No entities matched, using full graph context (limited to 50 triples)")

    # Format KG context
    kg_context = format_kg_context(subgraph_triples)

    if verbose:
        print(f"\nKnowledge Graph Context:")
        print(f"  Triples used: {len(subgraph_triples)}")
        if subgraph_triples:
            print("  Sample triples:")
            for s, p, o in subgraph_triples[:5]:
                print(f"    ({s}) --[{p}]--> ({o})")
            if len(subgraph_triples) > 5:
                print(f"    ... and {len(subgraph_triples) - 5} more")

    # Build prompt with KG context
    prompt = f"""Based on the following knowledge extracted from the document's knowledge graph, answer the question.
Use ONLY the information provided in the Knowledge Graph Context below. If the answer cannot be determined from the context, say so.

Knowledge Graph Context:
{kg_context}

Question: {question}

Answer:"""

    # Invoke LLM directly with KG context
    print(f"\nQuestion: {question}")

    try:
        response = llm.invoke(prompt)
        # Handle different response types
        if hasattr(response, 'content'):
            answer = response.content
        elif isinstance(response, dict):
            answer = response.get('result', str(response))
        else:
            answer = str(response)

        print(f"\nAnswer: {answer}")

    except Exception as e:
        print(f"\nError generating answer: {e}")


# =============================================================================
# Display Helper Functions
# =============================================================================

def _display_source_excerpts(excerpts, max_display=5, verbose=False):
    """Display source excerpts in formatted CLI output."""
    if not excerpts:
        print("  No source excerpts available")
        return

    display_count = len(excerpts) if verbose else min(max_display, len(excerpts))

    for i, excerpt in enumerate(excerpts[:display_count]):
        page = excerpt.get('page', 'unknown')
        page_num = excerpt.get('page_num', '?')
        chunk_num = excerpt.get('chunk_num', '?')
        source = excerpt.get('source', 'rag').upper()
        text = excerpt.get('excerpt', '')
        full_len = excerpt.get('full_length', len(text))

        print(f"\n  [{i+1}] {page} (Page {page_num}, Chunk {chunk_num}) [{source}]")

        # Show preview (max 150 chars in verbose, 50 chars in compact)
        max_chars = 150 if verbose else 50
        preview = text[:max_chars].replace('\n', ' ')
        if len(text) > max_chars:
            preview += "..."
        print(f"      {preview}")

        if full_len > len(text):
            print(f"      (Full chunk: {full_len} chars)")

    if len(excerpts) > display_count:
        remaining = len(excerpts) - display_count
        print(f"\n  ... and {remaining} more excerpts (use --verbose to see all)")


def _display_source_pages_summary(source_pages, page_provenance=None, max_preview=50, verbose=False):
    """Display summary of source pages with text previews and provenance labels."""
    if not source_pages:
        print("  No source pages available")
        return

    print(f"  Referenced pages: {len(source_pages)}")

    for page_num in sorted(source_pages.keys()):
        page_text = source_pages[page_num]

        # Get provenance info (if available)
        if page_provenance and page_num in page_provenance:
            prov = page_provenance[page_num]
            source_label = prov['source'].upper()
            kg_edges = prov.get('kg_edges', [])
            kg_count = len(kg_edges)

            # Build label
            if prov['source'] == 'kg' and kg_count > 0:
                label = f"[{source_label}: {kg_count} relationships]"
            elif prov['source'] == 'rag' and kg_count > 0:
                label = f"[{source_label} + {kg_count} KG relationships]"
            else:
                label = f"[{source_label}]"
        else:
            # Fallback if no provenance data
            label = ""
            kg_edges = []
            kg_count = 0

        if page_text.startswith('['):
            # Error or special message
            print(f"  • Page {page_num} {label}: {page_text}")
        else:
            # Show preview (max 150 chars in verbose, 50 chars in compact)
            max_chars = 150 if verbose else 50
            preview = page_text[:max_chars].replace('\n', ' ')
            if len(page_text) > max_chars:
                preview += "..."
            print(f"  • Page {page_num} ({len(page_text)} chars) {label}: \"{preview}\"")

            # Show relationships in verbose mode
            if verbose and kg_count > 0:
                print(f"      KG relationships:")
                for u, rel, v in kg_edges[:5]:  # Show first 5
                    print(f"        - {u} --[{rel}]--> {v}")
                if kg_count > 5:
                    print(f"        ... and {kg_count - 5} more")


# =============================================================================
# Multi-Page Query Command
# =============================================================================

def query_directory(collection_name: str, question: str,
                    source_directory: str = None,
                    similarity_threshold: float = 0.75, verbose: bool = False):
    """
    Query a multi-page index using hybrid RAG+KG system.

    Args:
        collection_name: Name of the collection to query
        question: The question to ask
        source_directory: Original source directory (for page loading)
        similarity_threshold: Minimum similarity score for RAG chunks (default: 0.75)
        verbose: Enable verbose output
    """
    print(f"\n{'='*60}")
    print("MULTI-PAGE QUERY MODE")
    print(f"{'='*60}")
    print(f"Collection: {collection_name}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")

    # Get LLM instance
    llm = get_llm()

    # Get embeddings
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()

    # Create hybrid document processor to load existing index
    processor = HybridDocumentProcessor(
        llm=llm,
        embeddings=embeddings,
        persist_path=f"./kg_store/{collection_name}"
    )

    try:
        # Load existing hybrid system
        print("[1/3] Loading hybrid RAG+KG system...")
        hybrid = processor.load_existing(collection_name, source_directory=source_directory)

        # Query the hybrid system
        print("\n[2/3] Querying hybrid system...")
        result = hybrid.query(
            question,
            kg_hops=2,   # Number of KG hops for subgraph extraction
            max_kg_nodes=50,  # Limit subgraph size
            score_threshold=similarity_threshold,  # Use parameter (default: 0.75)
            verbose=verbose
        )

        # Display results
        print(f"\n{'='*60}")
        print("RESPONSE")
        print(f"{'='*60}")
        print(result['response'])

        # Show query details
        print(f"\n{'='*60}")
        print("QUERY DETAILS")
        print(f"{'='*60}")
        print(f"  Query entities found: {result.get('query_entities', [])}")
        print(f"  RAG chunks used: {result.get('rag_chunks', 0)} (from {result.get('rag_candidates', 0)} candidates)")
        if result.get('rag_duplicates', 0) > 0:
            print(f"  RAG duplicates removed: {result.get('rag_duplicates', 0)} ({result.get('rag_unique', 0)} unique)")
        print(f"  RAG tokens used: ~{result.get('rag_tokens_used', 0)} / {result.get('context_budget', 0)} available")
        print(f"  KG nodes used: {result.get('subgraph_nodes', 0)}")
        print(f"  KG edges used: {result.get('subgraph_edges', 0)}")

        # Display source pages summary with provenance
        if 'source_pages' in result and result['source_pages']:
            print(f"\n{'='*60}")
            print("SOURCE PAGES")
            print(f"{'='*60}")
            _display_source_pages_summary(
                result['source_pages'],
                page_provenance=result.get('page_provenance'),
                verbose=verbose
            )

        print(f"\n{'='*60}\n")

    except FileNotFoundError:
        print(f"Error: Collection '{collection_name}' not found.")
        print(f"\nPlease index the directory first with:")
        print(f"  python kg_rag_cli.py --mode index --input-dir <directory> --collection {collection_name}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during query: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate input arguments
    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir is required")

    if args.input and args.input_dir:
        parser.error("Cannot use both --input and --input-dir. Choose one mode.")

    # Determine mode: single-file or multi-page
    is_multi_page = bool(args.input_dir)

    # Validate query mode arguments
    if args.mode == "query" and not args.question:
        parser.error("--question is required when using --mode query")

    # Auto-derive collection name for multi-page mode if not provided
    if is_multi_page:
        if not args.collection:
            # Auto-derive collection name from directory path
            # Remove trailing slashes and sanitize
            dir_path = args.input_dir.rstrip('/\\')
            args.collection = sanitize_path_for_index(dir_path)
            print(f"[Auto-derived collection name: {args.collection}]")

    # Dispatch to appropriate handler
    if args.mode == "index":
        if is_multi_page:
            # V3: Determine verbose_mode (enabled by default, disabled if --compact-extraction)
            use_verbose_mode = args.verbose_extraction and not args.compact_extraction

            # V3: Determine dedup_mode based on flags
            # Priority: --no-entity-resolution > --enable-entity-resolution > light (default)
            if args.no_entity_resolution:
                dedup_mode = 'none'  # Explicitly disabled
                enable_entity_resolution = False
            elif args.enable_entity_resolution:
                dedup_mode = 'standard'  # Full dedup
                enable_entity_resolution = True
            else:
                dedup_mode = 'light'  # Light dedup (v3 default)
                enable_entity_resolution = True

            # Multi-page indexing
            index_directory(
                directory=args.input_dir,
                collection_name=args.collection,
                file_pattern=args.file_pattern,
                enable_entity_resolution=enable_entity_resolution,
                visualize=not args.no_visualize,
                verbose=args.verbose,
                viz3d=args.viz3d,
                force_rebuild=args.force_rebuild,
                verbose_mode=use_verbose_mode,
                dedup_mode=dedup_mode
            )
        else:
            # V3: Determine verbose_mode (enabled by default, disabled if --compact-extraction)
            use_verbose_mode = args.verbose_extraction and not args.compact_extraction

            # V3: Determine dedup_mode based on flags
            # Priority: --no-entity-resolution > --enable-entity-resolution > light (default)
            if args.no_entity_resolution:
                dedup_mode = 'none'  # Explicitly disabled
                enable_entity_resolution = False
            elif args.enable_entity_resolution:
                dedup_mode = 'standard'  # Full dedup
                enable_entity_resolution = True
            else:
                dedup_mode = 'light'  # Light dedup (v3 default)
                enable_entity_resolution = True

            # Single-file indexing
            index_document(
                args.input,
                visualize=not args.no_visualize,
                verbose=args.verbose,
                enable_entity_resolution=enable_entity_resolution,
                verbose_mode=use_verbose_mode,
                dedup_mode=dedup_mode
            )
    elif args.mode == "query":
        if is_multi_page:
            # Multi-page querying
            query_directory(
                collection_name=args.collection,
                question=args.question,
                source_directory=args.input_dir,  # Pass original directory for file loading
                similarity_threshold=args.similarity_threshold,
                verbose=args.verbose
            )
        else:
            # Single-file querying
            query_graph(
                args.input,
                args.question,
                verbose=args.verbose
            )


if __name__ == "__main__":
    main()
