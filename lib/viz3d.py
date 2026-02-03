"""
3D Multi-Page Knowledge Graph Visualization with Plotly

Visualizes multi-page document graphs in 3D space:
- Each page forms a horizontal layer (Z = page number)
- Same entity on different pages shown as linked nodes (identity edges)
- Interactive layer toggling and entity highlighting

Dependencies: plotly (pip install plotly)
"""
import networkx as nx
import json
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Node3D:
    """Represents a node in 3D space (entity instance on a specific page)."""
    entity: str           # Canonical entity name
    page_num: int         # Page where this instance appears
    x: float              # 2D layout X coordinate
    y: float              # 2D layout Y coordinate
    z: float              # Z = layer height (computed from page_num)
    is_common: bool       # True if entity appears on multiple pages
    node_id: str          # Unique ID: f"{entity}__page{page_num}"


@dataclass
class Edge3D:
    """Represents an edge in 3D space."""
    source_id: str        # Source node ID
    target_id: str        # Target node ID
    relation: str         # Relationship type
    is_identity: bool     # True if cross-layer identity edge
    page_num: Optional[int]  # Page for intra-layer edges, None for identity


class MultiPageGraph3D:
    """
    Processes NetworkX graph for 3D multi-page visualization.

    Approach: "Linked Nodes" - same entity appears as separate nodes
    on each page where it appears, connected by vertical identity edges.
    """

    def __init__(self, graph: nx.DiGraph, entity_resolver=None):
        """
        Initialize 3D graph processor.

        Args:
            graph: NetworkX DiGraph with page metadata on edges
            entity_resolver: Optional EntityResolver for detecting common entities
        """
        self.graph = graph
        self.entity_resolver = entity_resolver
        self.nodes_3d: Dict[str, Node3D] = {}
        self.edges_3d: List[Edge3D] = []
        self.pages: Set[int] = set()
        self.common_entities: Set[str] = set()

    def process(self, layer_spacing: float = 1.0) -> Tuple[Dict[str, Node3D], List[Edge3D]]:
        """
        Process the graph into 3D node/edge structures.

        Args:
            layer_spacing: Z-axis spacing between page layers

        Returns:
            Tuple of (nodes_3d dict, edges_3d list)
        """
        # Step 1: Identify which pages each entity appears on
        entity_pages = self._build_entity_page_map()

        # Step 2: Identify common entities (appear on 2+ pages)
        self.common_entities = {
            entity for entity, pages in entity_pages.items()
            if len(pages) >= 2
        }

        # Step 3: Compute per-page 2D layouts
        page_layouts = self._compute_page_layouts(entity_pages)

        # Step 4: Create 3D nodes
        self._create_3d_nodes(entity_pages, page_layouts, layer_spacing)

        # Step 5: Create intra-layer edges (within same page)
        self._create_intra_layer_edges()

        # Step 6: Create identity edges (cross-layer for common entities)
        self._create_identity_edges(entity_pages)

        return self.nodes_3d, self.edges_3d

    def _build_entity_page_map(self) -> Dict[str, Set[int]]:
        """
        Build mapping of entity -> set of page numbers.

        Uses edge metadata 'page_num' to determine which pages
        each entity appears on.
        """
        entity_pages = defaultdict(set)

        for u, v, data in self.graph.edges(data=True):
            page_num = data.get('page_num', 1)  # Default to page 1
            entity_pages[u].add(page_num)
            entity_pages[v].add(page_num)
            self.pages.add(page_num)

        # Also check node attributes if available
        for node, attrs in self.graph.nodes(data=True):
            if 'pages' in attrs:
                pages = attrs['pages']
                if isinstance(pages, str):
                    # Deserialize from GML format
                    pages = {int(p) for p in pages.split(',') if p}
                elif isinstance(pages, (int, float)):
                    pages = {int(pages)}
                elif isinstance(pages, set):
                    pass  # Already a set
                else:
                    pages = set()
                entity_pages[node].update(pages)
                self.pages.update(pages)

        return dict(entity_pages)

    def _compute_page_layouts(self, entity_pages: Dict[str, Set[int]]) -> Dict[int, Dict[str, Tuple[float, float]]]:
        """
        Compute 2D layout for each page using per-page subgraphs.

        Returns:
            Dict mapping page_num -> {entity: (x, y)}
        """
        page_layouts = {}

        for page_num in sorted(self.pages):
            # Extract subgraph for this page
            page_entities = {e for e, pages in entity_pages.items() if page_num in pages}
            subgraph = self.graph.subgraph(page_entities).copy()

            if len(subgraph) == 0:
                page_layouts[page_num] = {}
                continue

            # Compute layout using NetworkX
            # Use spring_layout for organic arrangement
            try:
                k_param = 2.0 / (len(subgraph) ** 0.5) if len(subgraph) > 1 else 1.0
                layout = nx.spring_layout(
                    subgraph,
                    k=k_param,
                    iterations=50,
                    seed=42  # Reproducible layouts
                )
            except Exception:
                # Fallback to circular layout
                layout = nx.circular_layout(subgraph)

            # Scale layout to [0, 1] range
            positions = list(layout.values())
            if positions:
                min_x = min(p[0] for p in positions)
                max_x = max(p[0] for p in positions)
                min_y = min(p[1] for p in positions)
                max_y = max(p[1] for p in positions)

                scale_x = (max_x - min_x) if max_x != min_x else 1
                scale_y = (max_y - min_y) if max_y != min_y else 1

                page_layouts[page_num] = {
                    entity: (
                        (pos[0] - min_x) / scale_x,
                        (pos[1] - min_y) / scale_y
                    )
                    for entity, pos in layout.items()
                }
            else:
                page_layouts[page_num] = {}

        return page_layouts

    def _create_3d_nodes(self, entity_pages: Dict[str, Set[int]],
                         page_layouts: Dict[int, Dict[str, Tuple[float, float]]],
                         layer_spacing: float):
        """Create Node3D objects for all entity instances."""
        for entity, pages in entity_pages.items():
            is_common = len(pages) >= 2

            for page_num in pages:
                node_id = f"{entity}__page{page_num}"

                # Get position from page layout
                layout = page_layouts.get(page_num, {})
                x, y = layout.get(entity, (0.5, 0.5))
                z = (page_num - 1) * layer_spacing

                self.nodes_3d[node_id] = Node3D(
                    entity=entity,
                    page_num=page_num,
                    x=x,
                    y=y,
                    z=z,
                    is_common=is_common,
                    node_id=node_id
                )

    def _create_intra_layer_edges(self):
        """Create edges within the same page layer."""
        for u, v, data in self.graph.edges(data=True):
            page_num = data.get('page_num', 1)
            relation = data.get('relation', 'RELATED_TO')

            source_id = f"{u}__page{page_num}"
            target_id = f"{v}__page{page_num}"

            # Only create if both nodes exist on this page
            if source_id in self.nodes_3d and target_id in self.nodes_3d:
                self.edges_3d.append(Edge3D(
                    source_id=source_id,
                    target_id=target_id,
                    relation=relation,
                    is_identity=False,
                    page_num=page_num
                ))

    def _create_identity_edges(self, entity_pages: Dict[str, Set[int]]):
        """
        Create vertical identity edges connecting same entity across pages.

        For entities appearing on multiple pages, connect consecutive
        page instances with dashed vertical edges.
        """
        for entity in self.common_entities:
            pages = sorted(entity_pages[entity])

            # Connect consecutive pages
            for i in range(len(pages) - 1):
                source_id = f"{entity}__page{pages[i]}"
                target_id = f"{entity}__page{pages[i + 1]}"

                self.edges_3d.append(Edge3D(
                    source_id=source_id,
                    target_id=target_id,
                    relation="SAME_ENTITY",
                    is_identity=True,
                    page_num=None
                ))


def visualize_graph_3d(
    graph: nx.DiGraph,
    output_path: str,
    title: str = "Multi-Page Knowledge Graph",
    entity_resolver=None,
    layer_spacing: float = 1.0,
    node_size: int = 8,
    common_node_size: int = 12,
    show_labels: bool = True,
    edge_width: float = 1.5,
    identity_edge_width: float = 2.0,
    colorscale: str = "Viridis"
) -> bool:
    """
    Create interactive 3D visualization of multi-page knowledge graph.

    Args:
        graph: NetworkX DiGraph with page metadata
        output_path: Path to save HTML file
        title: Plot title
        entity_resolver: Optional EntityResolver for common entity detection
        layer_spacing: Z-axis spacing between page layers
        node_size: Default node size
        common_node_size: Size for common entities (appear on multiple pages)
        show_labels: Whether to show entity labels
        edge_width: Width for intra-layer edges
        identity_edge_width: Width for identity (cross-layer) edges
        colorscale: Plotly colorscale for page-based coloring

    Returns:
        True if visualization was created successfully
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[Viz3D] Error: plotly not installed. Run: pip install plotly")
        return False

    # Process graph into 3D structure
    processor = MultiPageGraph3D(graph, entity_resolver)
    nodes_3d, edges_3d = processor.process(layer_spacing)

    if not nodes_3d:
        print("[Viz3D] Warning: No nodes to visualize")
        return False

    pages = sorted(processor.pages)
    common_entities = processor.common_entities

    # Create Plotly figure
    fig = go.Figure()

    # =========================================================================
    # Create edge traces (one per page layer + one for identity edges)
    # =========================================================================

    # Intra-layer edges by page
    for page_num in pages:
        page_edges = [e for e in edges_3d if e.page_num == page_num and not e.is_identity]

        if page_edges:
            edge_x, edge_y, edge_z = [], [], []

            for edge in page_edges:
                src = nodes_3d[edge.source_id]
                tgt = nodes_3d[edge.target_id]

                # Line from source to target with None separator
                edge_x.extend([src.x, tgt.x, None])
                edge_y.extend([src.y, tgt.y, None])
                edge_z.extend([src.z, tgt.z, None])

            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                name=f'Page {page_num} edges',
                line=dict(color='rgba(100, 100, 100, 0.5)', width=edge_width),
                hoverinfo='skip',
                legendgroup=f'page_{page_num}',
                showlegend=False
            ))

    # Identity edges (vertical, dashed appearance)
    identity_edges = [e for e in edges_3d if e.is_identity]
    if identity_edges:
        identity_x, identity_y, identity_z = [], [], []

        for edge in identity_edges:
            src = nodes_3d[edge.source_id]
            tgt = nodes_3d[edge.target_id]

            identity_x.extend([src.x, tgt.x, None])
            identity_y.extend([src.y, tgt.y, None])
            identity_z.extend([src.z, tgt.z, None])

        fig.add_trace(go.Scatter3d(
            x=identity_x, y=identity_y, z=identity_z,
            mode='lines',
            name='Identity links',
            line=dict(color='rgba(255, 165, 0, 0.8)', width=identity_edge_width),
            hoverinfo='skip'
        ))

    # =========================================================================
    # Create node traces (one per page layer)
    # =========================================================================

    for page_num in pages:
        page_nodes = [n for n in nodes_3d.values() if n.page_num == page_num]

        if not page_nodes:
            continue

        # Separate common and regular nodes
        common_nodes = [n for n in page_nodes if n.is_common]
        regular_nodes = [n for n in page_nodes if not n.is_common]

        # Regular nodes for this page
        if regular_nodes:
            fig.add_trace(go.Scatter3d(
                x=[n.x for n in regular_nodes],
                y=[n.y for n in regular_nodes],
                z=[n.z for n in regular_nodes],
                mode='markers+text' if show_labels else 'markers',
                name=f'Page {page_num}',
                text=[n.entity for n in regular_nodes] if show_labels else None,
                textposition='top center',
                textfont=dict(size=8),
                marker=dict(
                    size=node_size,
                    color=page_num,
                    colorscale=colorscale,
                    cmin=min(pages),
                    cmax=max(pages),
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                hovertext=[f"{n.entity}<br>Page {n.page_num}" for n in regular_nodes],
                hoverinfo='text',
                legendgroup=f'page_{page_num}'
            ))

        # Common nodes (highlighted)
        if common_nodes:
            fig.add_trace(go.Scatter3d(
                x=[n.x for n in common_nodes],
                y=[n.y for n in common_nodes],
                z=[n.z for n in common_nodes],
                mode='markers+text' if show_labels else 'markers',
                name=f'Page {page_num} (common)',
                text=[n.entity for n in common_nodes] if show_labels else None,
                textposition='top center',
                textfont=dict(size=9, color='red'),
                marker=dict(
                    size=common_node_size,
                    color='red',
                    symbol='diamond',
                    opacity=1.0,
                    line=dict(width=2, color='gold')
                ),
                hovertext=[f"<b>{n.entity}</b><br>Page {n.page_num}<br>(Common entity)" for n in common_nodes],
                hoverinfo='text',
                legendgroup=f'page_{page_num}',
                customdata=[n.entity for n in common_nodes]  # For click callbacks
            ))

    # =========================================================================
    # Add layer planes (optional visual guides)
    # =========================================================================

    for page_num in pages:
        z = (page_num - 1) * layer_spacing
        fig.add_trace(go.Mesh3d(
            x=[0, 1, 1, 0],
            y=[0, 0, 1, 1],
            z=[z, z, z, z],
            opacity=0.1,
            color='lightblue',
            name=f'Layer {page_num}',
            showlegend=False,
            hoverinfo='skip'
        ))

    # =========================================================================
    # Layout configuration
    # =========================================================================

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        scene=dict(
            xaxis=dict(title='X', showbackground=False, showticklabels=False),
            yaxis=dict(title='Y', showbackground=False, showticklabels=False),
            zaxis=dict(
                title='Page',
                tickmode='array',
                tickvals=[(p - 1) * layer_spacing for p in pages],
                ticktext=[f'Page {p}' for p in pages]
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        showlegend=True,
        hovermode='closest'
    )

    # =========================================================================
    # Add updatemenus for layer toggling
    # =========================================================================

    # Create visibility controls
    layer_buttons = []

    # "Show All" button
    layer_buttons.append(dict(
        args=[{'visible': [True] * len(fig.data)}],
        label='Show All',
        method='update'
    ))

    # Per-page toggle buttons
    for page_num in pages:
        visible = []
        for trace in fig.data:
            # Check if trace belongs to this page
            trace_name = getattr(trace, 'name', '') or ''
            legendgroup = getattr(trace, 'legendgroup', '') or ''

            if legendgroup == f'page_{page_num}':
                visible.append(True)
            elif f'Page {page_num}' in trace_name:
                visible.append(True)
            elif trace_name == 'Identity links':
                visible.append(True)  # Always show identity links
            elif trace_name.startswith('Layer'):
                # Show corresponding layer plane
                if f'Layer {page_num}' in trace_name:
                    visible.append(True)
                else:
                    visible.append(False)
            else:
                visible.append(False)

        layer_buttons.append(dict(
            args=[{'visible': visible}],
            label=f'Page {page_num}',
            method='update'
        ))

    fig.update_layout(
        updatemenus=[
            dict(
                type='dropdown',
                direction='down',
                x=0.01,
                y=0.99,
                xanchor='left',
                yanchor='top',
                buttons=layer_buttons,
                bgcolor='white',
                font=dict(size=11)
            )
        ]
    )

    # =========================================================================
    # Add JavaScript for common node highlighting
    # =========================================================================

    # Create mapping of entity -> all trace indices for highlighting
    entity_to_traces = defaultdict(list)
    for i, trace in enumerate(fig.data):
        customdata = getattr(trace, 'customdata', None)
        if customdata is not None:
            for entity in customdata:
                entity_to_traces[entity].append(i)

    highlight_js = f"""
    <script>
    var entityMap = {json.dumps(dict(entity_to_traces))};

    document.addEventListener('DOMContentLoaded', function() {{
        var plots = document.querySelectorAll('.plotly-graph-div');
        plots.forEach(function(plot) {{
            plot.on('plotly_click', function(data) {{
                var point = data.points[0];
                if (point.customdata) {{
                    var entity = point.customdata;
                    var traces = entityMap[entity] || [];

                    // Highlight all instances of this entity
                    var updateMarkerLine = {{}};
                    for (var i = 0; i < plot.data.length; i++) {{
                        if (traces.includes(i)) {{
                            updateMarkerLine[i] = {{'marker.line.width': 4, 'marker.line.color': 'yellow'}};
                        }}
                    }}

                    for (var idx in updateMarkerLine) {{
                        Plotly.restyle(plot, updateMarkerLine[idx], [parseInt(idx)]);
                    }}
                }}
            }});

            plot.on('plotly_doubleclick', function() {{
                // Reset highlighting
                for (var i = 0; i < plot.data.length; i++) {{
                    if (plot.data[i].marker) {{
                        var resetStyle = {{'marker.line.width': plot.data[i].marker.symbol === 'diamond' ? 2 : 1}};
                        resetStyle['marker.line.color'] = plot.data[i].marker.symbol === 'diamond' ? 'gold' : 'white';
                        Plotly.restyle(plot, resetStyle, [i]);
                    }}
                }}
            }});
        }});
    }});
    </script>
    """

    # Save HTML with custom JavaScript
    html = fig.to_html(full_html=True, include_plotlyjs=True)

    # Inject custom JavaScript before closing body tag
    html = html.replace('</body>', f'{highlight_js}</body>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"[Viz3D] Saved 3D visualization to {output_path}")
    print(f"[Viz3D] {len(nodes_3d)} nodes, {len(edges_3d)} edges, {len(pages)} pages")
    print(f"[Viz3D] {len(common_entities)} common entities highlighted")

    return True
