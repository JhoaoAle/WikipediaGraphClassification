import pandas as pd
import networkx as nx
import ast
import numpy as np
import igraph as ig
import leidenalg
from community import community_louvain
from tqdm import tqdm
import pathlib

IN_PARQUET = pathlib.Path("data/40_preprocessed/42_mapping/articles.parquet")
OUT_PARQUET = pathlib.Path("data/50_clustered/53_louvain_leiden/articles.parquet")

def main():
    print("Loading parquet:", IN_PARQUET)
    df = pd.read_parquet(IN_PARQUET)
    print(f"Loaded {len(df):,} rows")

    # Ensure linked_article_titles is a list
    df['linked_article_titles'] = df['linked_article_titles'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # --- Build the graph ---
    print("Building the NetworkX graph...")
    G = nx.Graph()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding nodes"):
        G.add_node(row['title'])

    article_titles_set = set(df['title'])
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding edges"):
        for target in row['linked_article_titles']:
            if target in article_titles_set:
                G.add_edge(row['title'], target)

    # --- Louvain clustering ---
    print("Running Louvain community detection...")
    partition_louvain = community_louvain.best_partition(G)
    nx.set_node_attributes(G, partition_louvain, 'louvain_community')
    df['louvain_community'] = df['title'].map(partition_louvain)

    # --- Leiden clustering ---
    print("Converting to igraph...")
    mapping = {node: idx for idx, node in enumerate(G.nodes())}
    reverse_mapping = {idx: node for node, idx in mapping.items()}

    edges_igraph = [(mapping[u], mapping[v]) for u, v in G.edges()]
    g_ig = ig.Graph(edges=edges_igraph, directed=False)
    
    print("Running Leiden community detection...")
    leiden_partition = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition)

    partition_leiden = {reverse_mapping[v.index]: comm_id for comm_id, community in enumerate(leiden_partition) for v in community}
    df['leiden_community'] = df['title'].map(partition_leiden)

    # --- Save results ---
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving DataFrame with clusters to {OUT_PARQUET}...")
    df.to_parquet(OUT_PARQUET, index=False)

    # --- Report ---
    print("\nCommunity detection report:")
    print(f"🔵 Louvain: {len(set(partition_louvain.values()))} communities")
    modularity_louvain = community_louvain.modularity(partition_louvain, G)
    print(f"📦 Louvain Modularity: {modularity_louvain:.4f}")

    print(f"🔴 Leiden:  {len(set(partition_leiden.values()))} communities")
    modularity_leiden = leiden_partition.quality()
    print(f"📦 Leiden Modularity: {modularity_leiden:.4f}")
    
    # Optional: Print top Leiden community sizes
    leiden_sizes = pd.Series(list(partition_leiden.values())).value_counts().sort_values(ascending=False)
    print("Leiden community sizes (>10% of total):")
    total_nodes = len(G)
    for comm_id, size in leiden_sizes.items():
        if size > 0.1 * total_nodes:
            print(f"  - Community {comm_id}: {size} nodes")

if __name__ == "__main__":
    main()
