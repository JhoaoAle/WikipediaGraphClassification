import pandas as pd
import pathlib  
from typing import List, Dict
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import string
from nltk.corpus import stopwords
import re
from tqdm import tqdm
from functools import partial
tqdm.pandas()

IN_PARQUET = pathlib.Path("data/30_embedded/articles.parquet")
OUT_PARQUET = pathlib.Path("data/40_preprocessed/41_classification/articles.parquet")

def main():
    print("→ Loading parquet:", IN_PARQUET)
    df = pd.read_parquet(IN_PARQUET)
    print(f"✓ Loaded {len(df):,} rows")

    # === Clustering Dataset ===
    # Step 1: Select embedding columns
    embedding_cols = [col for col in df.columns if col.startswith('emb_')]
    X = df[embedding_cols].values

    # Step 3: Apply PCA to reduce dimensions (you can change n_components as needed)
    pca = PCA(n_components=200)  
    X_pca = pca.fit_transform(X)

    # Add PCA columns to dataframe
    pca_df = pd.DataFrame(X_pca, columns=[f'pca_{i}' for i in range(X_pca.shape[1])], index=df.index)
    df = pd.concat([df, pca_df], axis=1)

    # Create mask of rows that do NOT start with 'redirect'
    mask = ~df['cleaned_article_body'].str.startswith('redirect', na=False)

    # Apply mask to dataframe to filter out 'redirect' rows
    df = df[mask]

    # Drop original embedding columns
    df.drop(columns=embedding_cols + ['cleaned_article_body', 'linked_article_titles'], inplace=True)

    # Save preprocessed dataframe
    df.to_parquet(OUT_PARQUET, index=False)

    print("✓ wrote", OUT_PARQUET, len(df), "rows")

if __name__ == "__main__":
    main()
    