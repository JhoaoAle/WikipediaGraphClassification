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

    # 1. Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(f"🔢 Found {len(numerical_cols)} numerical columns")

    X = df[numerical_cols].fillna(0)  # Ensure no NaNs
    print("⚖️ Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Apply KMeans
    print(f"🧠 Running KMeans with {N_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init='auto', verbose=0, random_state=42)
    df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

    # 3. Save results
    print(f"💾 Saving clustered DataFrame to {OUT_PARQUET}")
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    # 4. Report
    print("📊 Clustering summary:")
    print(f"🔢 Number of clusters: {df['kmeans_cluster'].nunique()}")
    print(df['kmeans_cluster'].value_counts().sort_index())

if __name__ == "__main__":
    main()
