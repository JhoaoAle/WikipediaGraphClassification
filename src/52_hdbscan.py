import pandas as pd
import numpy as np
import pathlib
import hdbscan
from sklearn.preprocessing import StandardScaler

IN_PARQUET = pathlib.Path("data/40_preprocessed/41_classification/articles.parquet")
OUT_PARQUET = pathlib.Path("data/50_clustered/52_hdbscan/articles.parquet")

# HDBSCAN Parameters (adjust as needed)
MIN_CLUSTER_SIZE = 10
MIN_SAMPLES = None  # If None, defaults to MIN_CLUSTER_SIZE

def main():
    print("→ Loading parquet:", IN_PARQUET)
    df = pd.read_parquet(IN_PARQUET)
    print(f"✓ Loaded {len(df):,} rows")

    # 1. Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(f"🔢 Found {len(numerical_cols)} numerical columns")

    X = df[numerical_cols].fillna(0)
    print("⚖️ Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Apply HDBSCAN
    print(f"🧠 Running HDBSCAN with min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES}...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                                min_samples=MIN_SAMPLES,
                                prediction_data=True,
                                core_dist_n_jobs=-1)
    df['hdbscan_cluster'] = clusterer.fit_predict(X_scaled)

    # 3. Save results
    print(f"💾 Saving clustered DataFrame to {OUT_PARQUET}")
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    # 4. Report
    labels = df['hdbscan_cluster']
    num_clusters = len(set(labels)) - (1 if -1 in labels.values else 0)
    num_noise = (labels == -1).sum()

    print("📊 Clustering summary:")
    print(f"🔢 Number of clusters found: {num_clusters}")
    print(f"❌ Noise points (label = -1): {num_noise}")
    print(labels.value_counts().sort_index())

if __name__ == "__main__":
    main()
