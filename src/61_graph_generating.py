import pandas as pd
import networkx as nx
import ast
import umap
import matplotlib.pyplot as plt
import numpy as np
from community import community_louvain
from tqdm import tqdm
import pathlib
from scipy.sparse import csr_matrix

IN_PARQUET = pathlib.Path("data/40_preprocessed/42_mapping/articles.parquet")

# --- Load your DataFrame ---
print("📦 Loading DataFrame...")
df = pd.read_parquet(IN_PARQUET)

# Ensure linked_article_titles is a list
df['linked_article_titles'] = df['linked_article_titles'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# --- Build the graph ---
print("🧱 Building the graph...")
G = nx.Graph()

for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding nodes"):
    G.add_node(row['title'])

article_titles_set = set(df['title'])
for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding edges"):
    for target in row['linked_article_titles']:
        if target in article_titles_set:
            G.add_edge(row['title'], target)

# --- Louvain clustering ---
print("🧠 Running Louvain community detection...")
partition = community_louvain.best_partition(G)
nx.set_node_attributes(G, partition, 'louvain_community')

# --- Create adjacency matrix for layout ---
print("📐 Building adjacency matrix...")
nodes = list(G.nodes())
adj_array = nx.to_scipy_sparse_array(G, nodelist=nodes)
adj = csr_matrix(adj_array)

# --- UMAP layout (ForceAtlas2-like) ---
print("🗺️ Computing UMAP layout...")
embedding = umap.UMAP(n_components=2, metric='euclidean', random_state=42).fit_transform(adj)

# --- Prepare node colors ---
print("🎨 Coloring communities...")
communities = [partition[node] for node in nodes]
unique_communities = list(set(communities))
community_to_color = {comm: idx for idx, comm in enumerate(unique_communities)}
colors = [community_to_color[comm] for comm in communities]

# --- Plot ---
print("🖼️ Plotting and saving SVG...")
plt.figure(figsize=(40, 40))
plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap='tab20', s=2, alpha=0.8, linewidths=0)
plt.axis('off')
plt.tight_layout()
plt.savefig("wikipedia_umap_louvain.svg", format="svg", dpi=300)
print("✅ Done! Saved to wikipedia_umap_louvain.svg")
