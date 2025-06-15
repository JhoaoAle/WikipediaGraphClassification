import pandas as pd
import pathlib
import numpy as np
import igraph as ig
from concurrent.futures import ThreadPoolExecutor

IN_PARQUET = pathlib.Path("data/40_preprocessed/42_mapping/graph_dataset.parquet")


def random_walk_sample(g: ig.Graph, target_size: int, jump_prob: float = 0.15) -> list:
    """Random Walk sampling with optional jump to random node"""
    visited = set()
    current = np.random.randint(0, g.vcount())

    while len(visited) < target_size:
        visited.add(current)
        if np.random.rand() < jump_prob:
            current = np.random.randint(0, g.vcount())  # jump
        else:
            neighbors = g.neighbors(current, mode="ALL")
            if neighbors:
                current = np.random.choice(neighbors)
            else:
                current = np.random.randint(0, g.vcount())  # dead end recovery
    return list(visited)

def parallel_calculate(func, graph, nodes=None):
    """Helper function for parallel computation"""
    if nodes is None:
        nodes = range(graph.vcount())
    
    # Create a list of tasks where each task is (graph, node)
    tasks = [(graph, node) for node in nodes]
    
    with ThreadPoolExecutor() as executor:
        # Use a helper function that unpacks the arguments
        results = list(executor.map(lambda args: func(*args), tasks))
    return results

def describe_graph(g: ig.Graph):
    print("📊 Graph Statistics Report")
    print("="*50)

    print(f"🧩 Nodes: {g.vcount()} | Edges: {g.ecount()}")

    # Degree calculation - using built-in parallel methods where possible
    degrees = g.degree()
    avg_degree = np.mean(degrees)
    print(f"🔗 Average Degree: {avg_degree:.2f} (Barabási, 2002)")

    if g.is_directed():
        in_degrees = g.indegree()
        out_degrees = g.outdegree()
        print(f"📥 Avg In-Degree: {np.mean(in_degrees):.2f}")
        print(f"📤 Avg Out-Degree: {np.mean(out_degrees):.2f}")

    # PageRank - using optimized implementation
    print("⭐ Calculating PageRank...")
    pr = g.pagerank(damping=0.85, weights=None, implementation="prpack")
    top_pr = sorted(enumerate(pr), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 nodes by PageRank (Page et al., 1999):")
    for idx, score in top_pr:
        print(f"  Node {idx}: {score:.4f}")

    # Betweenness Centrality - with approximation for large graphs
    print("⏳ Calculating Betweenness Centrality...")
    betweenness = g.betweenness(cutoff=6)  # Limit path length for speed
    top_betw = sorted(enumerate(betweenness), key=lambda x: x[1], reverse=True)[:5]
    print("🔀 Top 5 nodes by Betweenness (Freeman, 1977):")
    for idx, score in top_betw:
        print(f"  Node {idx}: {score:.4f}")

    # Clustering Coefficient
    print("🔁 Calculating Clustering Coefficients...")
    clustering_coeffs = g.transitivity_local_undirected(mode="zero")
    avg_cluster = np.mean(clustering_coeffs)
    print(f"Avg Clustering Coefficient (Watts & Strogatz, 1998): {avg_cluster:.4f}")

    # Giant component analysis
    print("🛣 Analyzing Largest Connected Component...")
    components = g.clusters(mode="weak")
    giant = g.subgraph(components.giant().vs)
    try:
        avg_path_len = giant.average_path_length(directed=False)
        diameter = giant.diameter(directed=False)
    except:
        avg_path_len = diameter = "too large"
    print(f"    Avg Shortest Path Length: {avg_path_len}")
    print(f"    Diameter: {diameter}")

    print(f"🧱 Number of Weakly Connected Components: {len(components)}")

    # Degree Assortativity
    try:
        assort = g.assortativity_degree(directed=True)
        print(f"📊 Degree Assortativity (Newman, 2002): {assort:.3f}")
    except:
        print("📊 Degree Assortativity: Not computable.")

    print("="*50)

def main():
    # Load data with optimized parameters
    df = pd.read_parquet(IN_PARQUET, engine='pyarrow')
    print("✅ DataFrame loaded.")
    
    # Create graph more efficiently
    all_nodes = pd.concat([df['source'], df['target']])
    unique_nodes = pd.unique(all_nodes)
    node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}
    
    # Convert to numpy arrays for faster processing
    sources = df['source'].map(node_to_index).to_numpy()
    targets = df['target'].map(node_to_index).to_numpy()
    
    g = ig.Graph(n=len(unique_nodes), edges=list(zip(sources, targets)), directed=True)

    print(f"🎯 Target sample size (15%): {int(0.15 * g.vcount())}")
    sampled_nodes = random_walk_sample(g, target_size=int(0.15 * g.vcount()))
    g_sample = g.subgraph(sampled_nodes)

    print(f"📉 Sampled subgraph: {g_sample.vcount()} nodes, {g_sample.ecount()} edges")
    

    
    # Set number of threads for igraph (if available)
    try:
        ig.set_blas_nthreads(8)
        ig.set_lapack_nthreads(8)
    except:
        pass
    
    describe_graph(g_sample)

if __name__ == "__main__":
    main()