import pandas as pd
import networkx as nx
import ast
import numpy as np
from community import community_louvain
from tqdm import tqdm
import pathlib

IN_PARQUET = pathlib.Path("data/40_preprocessed/42_mapping/articles.parquet")
OUT_PARQUET = pathlib.Path("data/50_clustered/53_louvain/articles.parquet")

def main():
    print("→ Loading parquet:", IN_PARQUET)
    df = pd.read_parquet(IN_PARQUET)
    print(f"✓ Loaded {len(df):,} rows")

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

    # Map partition results back to DataFrame
    df['louvain_community'] = df['title'].map(partition)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)  # Make sure directory exists

    print(f"💾 Saving DataFrame with clusters to {OUT_PARQUET}...")
    df.to_parquet(OUT_PARQUET, index=False)

    # --- Report community detection stats ---
    print("📊 Community detection report:")
    num_communities = len(set(partition.values()))
    community_sizes = pd.Series(list(partition.values())).value_counts().sort_values(ascending=False)
    modularity = community_louvain.modularity(partition, G)

    print(f"🔢 Number of communities: {num_communities}")
    print(f"📈 Modularity: {modularity:.4f}")
    print("📦 Community sizes (>10% of total):")

    total_nodes = len(G)
    threshold = total_nodes * 0.10
    large_communities = community_sizes[community_sizes > threshold]

    for comm_id, size in large_communities.items():
        print(f"  - Community {comm_id}: {size} nodes")

if __name__ == "__main__":
    main()