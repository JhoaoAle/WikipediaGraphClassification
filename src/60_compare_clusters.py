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

if __name__ == "__main__":
    main()
