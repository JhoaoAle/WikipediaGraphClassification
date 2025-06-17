import pandas as pd
import pathlib
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)

IN_PARQUET = pathlib.Path("data/50_clustered/articles.parquet")

def compare_pair(df, col1, col2):
    df_clean = df.dropna(subset=[col1, col2])
    print(f"✅ Comparing '{col1}' vs '{col2}' — valid rows: {len(df_clean):,} of {len(df):,}")
    
    labels1 = df_clean[col1].astype(int)
    labels2 = df_clean[col2].astype(int)

    ari = adjusted_rand_score(labels1, labels2)
    nmi = normalized_mutual_info_score(labels1, labels2)
    homogeneity = homogeneity_score(labels1, labels2)
    completeness = completeness_score(labels1, labels2)
    v_measure = v_measure_score(labels1, labels2)

    print(f"\n🔍 {col1} vs {col2}")
    print(f"  Adjusted Rand Index (ARI):     {ari:.4f}")
    print(f"  Normalized Mutual Info (NMI):  {nmi:.4f}")
    print(f"  Homogeneity:                   {homogeneity:.4f}")
    print(f"  Completeness:                  {completeness:.4f}")
    print(f"  V-Measure:                     {v_measure:.4f}\n")

    return {
        "pair": f"{col1} vs {col2}",
        "ari": ari,
        "nmi": nmi,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure
    }

def analyze_kmeans(df, col="kmeans_cluster"):
    print(f"📊 Analyzing KMeans clustering: '{col}'")

    if col not in df.columns:
        print(f"❌ Column '{col}' not found.")
        return

    cluster_counts = df[col].value_counts().sort_values(ascending=False)
    n_clusters = cluster_counts.shape[0]
    total_points = cluster_counts.sum()

    print(f"✅ Total clusters: {n_clusters}")
    print(f"🧮 Cluster size stats:")
    print(cluster_counts.describe())

    print(f"\n📌 Top 5 largest clusters:")
    for i, (cluster_id, size) in enumerate(cluster_counts.head(5).items(), 1):
        print(f"  {i}. Cluster {cluster_id}: {size} points ({100*size/total_points:.2f}%)")

    return {
        "total_clusters": n_clusters,
        "cluster_size_stats": cluster_counts.describe(),
        "top_clusters": cluster_counts.head(5).to_dict()
    }

def main():
    df = pd.read_parquet(IN_PARQUET)

    results = []
    pairs = [
        ("louvain_community", "kmeans_cluster"),
        ("louvain_community", "leiden_community"),
        ("kmeans_cluster", "leiden_community")
    ]
    
    for col1, col2 in pairs:
        results.append(compare_pair(df, col1, col2))

    kmeans_stats = analyze_kmeans(df)

if __name__ == "__main__":
    main()
