import pandas as pd
import numpy as np
import pathlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

IN_PARQUET = pathlib.Path("data/40_preprocessed/41_classification/articles.parquet")
OUT_PARQUET = pathlib.Path("data/50_clustered/51_k_means/articles.parquet")
N_CLUSTERS = 4500

def main():
    print("→ Loading parquet:", IN_PARQUET)
    df = pd.read_parquet(IN_PARQUET)
    print(f"✓ Loaded {len(df):,} rows")

    # 1. Select only numerical columns excluding article_id
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'article_id' in numerical_cols:
        numerical_cols.remove('article_id')
    print(f"🔢 Using {len(numerical_cols)} numerical columns for clustering")

    # 2. Standardize features
    X = df[numerical_cols].fillna(0)
    print("⚖️ Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Apply KMeans
    print(f"🧠 Running KMeans with {N_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init='auto', verbose=0, random_state=42)
    df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

    # 4. Keep only article_id and cluster assignment
    result_df = df[['article_id', 'kmeans_cluster']]

    # 5. Save results
    print(f"💾 Saving clustered DataFrame to {OUT_PARQUET}")
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(OUT_PARQUET, index=False)

    # 6. Report
    print("📊 Clustering summary:")
    print(f"🔢 Number of clusters: {result_df['kmeans_cluster'].nunique()}")
    print(result_df['kmeans_cluster'].value_counts().sort_index())

if __name__ == "__main__":
    main()
