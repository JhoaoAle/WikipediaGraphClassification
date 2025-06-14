import pandas as pd
import pathlib
import subprocess
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Paths to the scripts
SCRIPT_KMEANS = "src/51_kmeans.py"
SCRIPT_LOUVAIN = "src/53_louvain.py"

# Input and output parquet files
IN_PARQUET_KMEANS = pathlib.Path("data/50_clustered/51_k_means/articles.parquet")
IN_PARQUET_LOUVAIN = pathlib.Path("data/50_clustered/53_louvain/articles.parquet")
OUT_PARQUET = pathlib.Path("data/50_clustered/articles.parquet")

def run_script(script_path):
    print(f"🚀 Running script: {script_path}")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error running {script_path}:\n{result.stderr}")
        raise RuntimeError(f"Script failed: {script_path}")
    print(f"✅ Finished {script_path}")

def main():
    print(f"🐍 Python path in use: {sys.executable}")
    # Step 1: Generate the cluster files
    run_script(SCRIPT_KMEANS)
    run_script(SCRIPT_LOUVAIN)

    # Step 2: Load the generated parquet files
    print("📦 Loading K-Means clusters:", IN_PARQUET_KMEANS)
    df_kmeans = pd.read_parquet(IN_PARQUET_KMEANS)

    print("📦 Loading Louvain clusters:", IN_PARQUET_LOUVAIN)
    df_louvain = pd.read_parquet(IN_PARQUET_LOUVAIN)

    # Step 3: Merge KMeans into Louvain
    df_combined = pd.merge(
        df_louvain,
        df_kmeans[['article_id', 'kmeans_cluster']],
        on='article_id',
        how='left'
    )

    # Step 4: Save output
    print(f"💾 Saving combined DataFrame to: {OUT_PARQUET}")
    df_combined.to_parquet(OUT_PARQUET, index=False)

if __name__ == "__main__":
    main()
