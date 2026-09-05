"""Interactive Knowledge Graph Explorer page."""

import streamlit as st
import os
import sys
import tempfile
from collections import Counter

st.set_page_config(
    page_title="Knowledge Graph Explorer",
    page_icon="",
    layout="wide",
)

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, APP_DIR)


@st.cache_resource(show_spinner="Loading knowledge graph...")
def load_graph():
    from backend.core.config import GRAPH_CACHE_PATH
    from backend.core.graph import load_graph as _load_graph
    return _load_graph(GRAPH_CACHE_PATH)


G = load_graph()

# --- Precompute stats ---
@st.cache_data(show_spinner=False)
def get_graph_stats():
    nodes = list(G.nodes())
    degrees = dict(G.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

    edge_labels = Counter()
    for _, _, d in G.edges(data=True):
        for label in d.get("labels", []):
            edge_labels[label] += 1

    relation_assertions = sum(edge_labels.values())

    return {
        "nodes": nodes,
        "degrees": degrees,
        "top_nodes": top_nodes,
        "edge_labels": edge_labels,
        "relation_assertions": relation_assertions,
    }


stats = get_graph_stats()

# --- Header ---
st.title("Knowledge Graph Explorer")
st.markdown("*Interactive visualisation of the regulatory knowledge graph extracted from MiFID II, MiFIR, and Delegated Regulation 2017/565*")

# --- Overview metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Nodes", f"{G.number_of_nodes():,}")
col2.metric("Edge Pairs", f"{G.number_of_edges():,}")
col3.metric("Relation Assertions", f"{stats['relation_assertions']:,}")
col4.metric("Relation Types", len(stats["edge_labels"]))

st.markdown("---")

# --- Relation distribution ---
st.markdown("### Relation Type Distribution")

import pandas as pd

rel_df = pd.DataFrame(
    [(rel, count) for rel, count in stats["edge_labels"].most_common()],
    columns=["Relation", "Count"],
)
st.bar_chart(rel_df.set_index("Relation"))

# --- Top entities ---
st.markdown("### Top Entities by Connectivity")

top_n = st.slider("Number of entities to show", 10, 50, 20)
top_df = pd.DataFrame(
    stats["top_nodes"][:top_n],
    columns=["Entity", "Degree"],
)
st.bar_chart(top_df.set_index("Entity"))

st.markdown("---")

# --- Entity Explorer ---
st.markdown("### Entity Explorer")
st.markdown("Select an entity to see all its regulatory relationships.")

# Searchable entity selector
search_term = st.text_input("Search for an entity", placeholder="e.g., investment firm, esma, compliance")

if search_term:
    matching = [n for n in stats["nodes"] if search_term.lower() in n.lower()]
    matching.sort(key=lambda n: stats["degrees"].get(n, 0), reverse=True)
else:
    matching = [n for n, _ in stats["top_nodes"][:50]]

if not matching:
    st.warning("No entities found matching your search.")
    st.stop()

selected_entity = st.selectbox(
    "Select entity",
    matching[:100],
    format_func=lambda x: f"{x} (degree: {stats['degrees'].get(x, 0)})",
)

if selected_entity:
    st.markdown(f"#### Relationships for: `{selected_entity}`")

    # Outgoing edges
    outgoing = []
    for _, target, data in G.out_edges(selected_entity, data=True):
        for label in data.get("labels", []):
            sources = data.get("source_ids", [])
            outgoing.append({
                "Direction": "outgoing",
                "Relation": label,
                "Target": target,
                "Sources": ", ".join(sources[:3]) + ("..." if len(sources) > 3 else ""),
            })

    # Incoming edges
    incoming = []
    for source, _, data in G.in_edges(selected_entity, data=True):
        for label in data.get("labels", []):
            sources = data.get("source_ids", [])
            incoming.append({
                "Direction": "incoming",
                "Relation": label,
                "Source": source,
                "Sources": ", ".join(sources[:3]) + ("..." if len(sources) > 3 else ""),
            })

    col_out, col_in = st.columns(2)

    with col_out:
        st.markdown(f"**Outgoing** ({len(outgoing)})")
        if outgoing:
            for edge in outgoing:
                st.markdown(f"- `{selected_entity}` **—[{edge['Relation']}]→** `{edge['Target']}`")
        else:
            st.markdown("*No outgoing relationships*")

    with col_in:
        st.markdown(f"**Incoming** ({len(incoming)})")
        if incoming:
            for edge in incoming:
                st.markdown(f"- `{edge['Source']}` **—[{edge['Relation']}]→** `{selected_entity}`")
        else:
            st.markdown("*No incoming relationships*")

    # Source provenance
    node_sources = G.nodes[selected_entity].get("source_ids", [])
    if node_sources:
        with st.expander(f"Source provenance ({len(node_sources)} chunks)", expanded=False):
            for sid in node_sources:
                st.code(sid)

st.markdown("---")

# --- Interactive Graph Visualisation ---
st.markdown("### Interactive Graph Visualisation")
st.markdown("Explore a neighbourhood of the graph around a selected entity.")

vis_entity = st.selectbox(
    "Centre entity for visualisation",
    matching[:100],
    format_func=lambda x: f"{x} (degree: {stats['degrees'].get(x, 0)})",
    key="vis_entity",
)

vis_hops = st.slider("Expansion hops", 1, 3, 1)
max_nodes = st.slider("Max nodes to display", 20, 150, 50)

# Colour map for relation types
RELATION_COLORS = {
    "REQUIRES": "#e74c3c",
    "PROHIBITS": "#c0392b",
    "PERMITS": "#27ae60",
    "GOVERNS": "#2980b9",
    "APPLIES_TO": "#8e44ad",
    "HAS_OBLIGATION": "#d35400",
    "SUPERVISES": "#f39c12",
    "REPORTS_TO": "#1abc9c",
    "AUTHORIZES": "#2ecc71",
    "ENSURES": "#3498db",
    "EXEMPTS": "#e67e22",
    "DEFINES": "#9b59b6",
    "SUBJECT_TO": "#34495e",
    "DISCLOSES_TO": "#16a085",
    "NOTIFIES": "#f1c40f",
    "COOPERATES_WITH": "#7f8c8d",
    "RELATED_TO": "#bdc3c7",
}

if vis_entity:
    from pyvis.network import Network
    import networkx as nx

    # BFS to collect neighbourhood
    visited = {vis_entity}
    frontier = {vis_entity}
    subgraph_nodes = {vis_entity}

    for _ in range(vis_hops):
        next_frontier = set()
        for node in frontier:
            for neighbor in list(G.successors(node)) + list(G.predecessors(node)):
                if neighbor not in visited:
                    next_frontier.add(neighbor)
                    subgraph_nodes.add(neighbor)
                    if len(subgraph_nodes) >= max_nodes:
                        break
            visited.update(next_frontier)
            if len(subgraph_nodes) >= max_nodes:
                break
        frontier = next_frontier
        if len(subgraph_nodes) >= max_nodes:
            break

    # Build pyvis network
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#0e1117",
        font_color="white",
        directed=True,
        notebook=False,
    )

    net.set_options("""
    {
        "nodes": {
            "font": {"size": 14, "color": "white"},
            "borderWidth": 2
        },
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
            "font": {"size": 10, "color": "#aaaaaa", "align": "middle"},
            "smooth": {"type": "continuous"}
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
    }
    """)

    # Add nodes
    for node in subgraph_nodes:
        degree = stats["degrees"].get(node, 0)
        size = min(10 + degree * 1.5, 50)
        color = "#e74c3c" if node == vis_entity else "#3498db"
        net.add_node(
            node,
            label=node[:30] + ("..." if len(node) > 30 else ""),
            title=f"{node}\nDegree: {degree}\nSources: {len(G.nodes[node].get('source_ids', []))}",
            size=size,
            color=color,
        )

    # Add edges
    for u, v, data in G.edges(data=True):
        if u in subgraph_nodes and v in subgraph_nodes:
            labels = data.get("labels", [])
            sources = data.get("source_ids", [])
            for label in labels:
                color = RELATION_COLORS.get(label, "#95a5a6")
                net.add_edge(
                    u, v,
                    label=label,
                    title=f"{u} —[{label}]→ {v}\nSources: {', '.join(sources[:3])}",
                    color=color,
                    width=2,
                )

    # Render
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        with open(f.name, "r") as html_file:
            html_content = html_file.read()
        os.unlink(f.name)

    st.components.v1.html(html_content, height=620, scrolling=False)

    st.caption(
        f"Showing {len(subgraph_nodes)} nodes around `{vis_entity}` "
        f"({vis_hops} hop{'s' if vis_hops > 1 else ''}). "
        f"Hover over nodes and edges for details. Drag to rearrange."
    )

    # Legend
    st.markdown("#### Relation Colour Legend")
    legend_cols = st.columns(4)
    for i, (rel, color) in enumerate(RELATION_COLORS.items()):
        col_idx = i % 4
        legend_cols[col_idx].markdown(
            f'<span style="color:{color}">&#9632;</span> {rel}',
            unsafe_allow_html=True,
        )
