import pandas as pd
import networkx as nx
import ast
from community import community_louvain
import pathlib
from tqdm import tqdm

# --- Config ---
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

# Add nodes
for _, row in df.iterrows():
    G.add_node(row['title'])

# Add edges (only when linked article exists in the dataset)
article_titles_set = set(df['title'])
for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding edges"):
    source = row['title']
    for target in row['linked_article_titles']:
        if target in article_titles_set:
            G.add_edge(source, target)

# --- Run Louvain community detection ---
print("🧠 Running Louvain community detection...")
partition = community_louvain.best_partition(G)

# Assign community as node attribute in the full graph
nx.set_node_attributes(G, partition, 'louvain_community')

# Add the community assignment to the DataFrame
df['louvain_community'] = df['title'].map(partition)

# --- Save the full graph with communities ---
print("💾 Saving full graph to GEXF...")
nx.write_gexf(G, "wikipedia_graph.gexf")

# --- Select Top 10k Nodes by Degree ---
print("🔎 Selecting top 10,000 nodes by degree...")
top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10_000]
top_node_names = [node for node, _ in top_nodes]

# --- Induce Subgraph ---
print("🔧 Inducing subgraph...")
G_sub = G.subgraph(top_node_names).copy()

# Attach community attribute to subgraph
nx.set_node_attributes(G_sub, {node: partition[node] for node in G_sub.nodes}, 'louvain_community')

# --- Save subgraph for Gephi ---
print("💾 Saving top-10k subgraph to GEXF...")
nx.write_gexf(G_sub, "wikipedia_top10k_louvain.gexf")

# --- Select Top 10k Nodes by Degree ---
print("🔎 Selecting top 1,000 nodes by degree...")
top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:1_000]
top_node_names = [node for node, _ in top_nodes]

# --- Induce Subgraph ---
print("🔧 Inducing subgraph...")
G_sub = G.subgraph(top_node_names).copy()

# Attach community attribute to subgraph
nx.set_node_attributes(G_sub, {node: partition[node] for node in G_sub.nodes}, 'louvain_community')

# --- Save subgraph for Gephi ---
print("💾 Saving top-10k subgraph to GEXF...")
nx.write_gexf(G_sub, "wikipedia_top1k_louvain.gexf")

# --- Select Top 10k Nodes by Degree ---
print("🔎 Selecting top 500 nodes by degree...")
top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:500]
top_node_names = [node for node, _ in top_nodes]

# --- Induce Subgraph ---
print("🔧 Inducing subgraph...")
G_sub = G.subgraph(top_node_names).copy()

# Attach community attribute to subgraph
nx.set_node_attributes(G_sub, {node: partition[node] for node in G_sub.nodes}, 'louvain_community')

# --- Save subgraph for Gephi ---
print("💾 Saving top-500 subgraph to GEXF...")
nx.write_gexf(G_sub, "wikipedia_top500_louvain.gexf")


print("🔎 Selecting top 100 nodes by degree...")
top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:100]
top_node_names = [node for node, _ in top_nodes]

# --- Induce Subgraph ---
print("🔧 Inducing subgraph...")
G_sub = G.subgraph(top_node_names).copy()

# Attach community attribute to subgraph
nx.set_node_attributes(G_sub, {node: partition[node] for node in G_sub.nodes}, 'louvain_community')

# --- Save subgraph for Gephi ---
print("💾 Saving top-100 subgraph to GEXF...")
nx.write_gexf(G_sub, "wikipedia_top100_louvain.gexf")


print("🔎 Selecting top 200 nodes by degree...")
top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:200]
top_node_names = [node for node, _ in top_nodes]

# --- Induce Subgraph ---
print("🔧 Inducing subgraph...")
G_sub = G.subgraph(top_node_names).copy()

# Attach community attribute to subgraph
nx.set_node_attributes(G_sub, {node: partition[node] for node in G_sub.nodes}, 'louvain_community')

# --- Save subgraph for Gephi ---
print("💾 Saving top-200 subgraph to GEXF...")
nx.write_gexf(G_sub, "wikipedia_top200_louvain.gexf")



print("✅ Done!")
