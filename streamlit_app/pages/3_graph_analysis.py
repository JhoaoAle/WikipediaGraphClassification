import streamlit as st
import networkx as nx
from pyvis.network import Network
import tempfile
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import json

st.title("📁 Interactive GEXF Graph Viewer + Path Explorer")

st.write("""
Please allow a few seconds for the graph to load.
""")

# Automatically load the GEXF file from a relative path
CURRENT_DIR = Path(__file__).parent
GEXF_PATH = (CURRENT_DIR.parent / "data_sample" / "wikipedia_top200_louvain.gexf").resolve()

if not GEXF_PATH.exists():
    st.error(f"GEXF file not found at {GEXF_PATH}")
    st.stop()

# Load the graph
G = nx.read_gexf(GEXF_PATH)

# Ensure the graph is directed
if not G.is_directed():
    G = G.to_directed()

# Optional: Show graph type
st.write(f"Graph type: {'Directed' if G.is_directed() else 'Undirected'}")

# Convert louvain_community to string for consistent coloring
for _, data in G.nodes(data=True):
    if "louvain_community" in data:
        data["community"] = str(data["louvain_community"])

# Extract nodes and community data
all_nodes = list(G.nodes)
communities = nx.get_node_attributes(G, "community")

# UI: Node selection for shortest path
st.sidebar.header("🎯 Path Finder")
source_node = st.sidebar.selectbox("Select source node:", all_nodes)
target_node = st.sidebar.selectbox("Select target node:", all_nodes)
highlight_path = st.sidebar.button("🧭 Highlight Shortest Path")

# Create PyVis network
net = Network(height="600px", width="100%", directed=True, notebook=False)
net.force_atlas_2based()

# Set options with stabilization (stops moving after a few seconds)
net.set_options(json.dumps({
    "physics": {
        "enabled": True,
        "forceAtlas2Based": {
            "gravitationalConstant": -50,
            "centralGravity": 0.01,
            "springLength": 100,
            "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {
            "enabled": True,
            "iterations": 150,
            "updateInterval": 25,
            "onlyDynamicEdges": False,
            "fit": True
        }
    },
    "interaction": {
        "dragNodes": True,
        "dragView": True,
        "zoomView": True
    }
}))

# Community-based coloring
unique_communities = list(set(communities.values()))
color_map = cm.get_cmap('tab20', len(unique_communities))
community_color = {
    comm: mcolors.to_hex(color_map(i)) for i, comm in enumerate(unique_communities)
}

# Compute shortest path if needed
path_nodes = []
if highlight_path:
    try:
        path_nodes = nx.shortest_path(G, source=source_node, target=target_node)
    except nx.NetworkXNoPath:
        st.warning(f"No path found between {source_node} and {target_node}")
        path_nodes = []

# Add nodes (highlight if on path)
for node, data in G.nodes(data=True):
    label = str(node)
    color = "red" if node in path_nodes else community_color.get(data.get("community", ""), "#cccccc")
    size = 25 if node in path_nodes else 15
    net.add_node(node, label=label, color=color, size=size)

# Add edges, highlighting only directed ones in the path
path_edges = set(zip(path_nodes, path_nodes[1:])) if len(path_nodes) >= 2 else set()
for u, v in G.edges():
    if (u, v) in path_edges:
        net.add_edge(u, v, color="red", width=3, arrows="to")
    else:
        net.add_edge(u, v, arrows="to")

# Render to HTML and display
with tempfile.NamedTemporaryFile(mode="w+", suffix=".html", delete=False) as tmp_file:
    tmp_path = tmp_file.name
    net.write_html(tmp_path)

    with open(tmp_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=600, scrolling=True)

# Description
st.markdown(""" 
Above is a small visualization of a 200-sized subset of Simple Wikipedia Articles, 
made with the PyVis library and serving as an interactive visualization 
of a small section of the simple Wikipedia universe.

You can select a source and target node, and click the "Highlight Shortest Path" 
button to visualize the shortest **directed** path between them in red.

The nodes are colored by their Louvain community, and you can click them to see their connections.

Bear in mind this subset is about 5 times smaller than the sample used for graph-related analysis, which itself
was a 15% sample of the full graph.
""")
