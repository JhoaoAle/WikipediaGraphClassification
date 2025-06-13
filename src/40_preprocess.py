import pandas as pd
import pathlib  
from typing import Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA, TruncatedSVD
import string
from nltk.corpus import stopwords
import re
from tqdm import tqdm
from functools import partial

tqdm.pandas()

IN_PARQUET = pathlib.Path("data/30_embedded/articles.parquet")
IN_PARQUET_2 = pathlib.Path("data/31_tf_idf/articles.parquet")
OUT_PARQUET = pathlib.Path("data/40_preprocessed/41_classification/articles.parquet")

N_COMPONENTS = 200

def run_reduction(name: str, X: np.ndarray, method: str) -> Tuple[str, str, float, np.ndarray]:
    if method == "pca":
        print(f"⏳ Running PCA on {name}...")
        model = PCA(n_components=N_COMPONENTS)
        X_reduced = model.fit_transform(X)
        score = model.explained_variance_ratio_.sum()
    elif method == "svd":
        print(f"⏳ Running TruncatedSVD on {name}...")
        model = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
        X_reduced = normalize(model.fit_transform(X))
        score = model.explained_variance_ratio_.sum()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"✓ {method.upper()} on {name} → Explained variance: {score:.4f}")
    return name, method, score, X_reduced

def main():
    print("→ Loading embeddings:", IN_PARQUET)
    df_emb = pd.read_parquet(IN_PARQUET)
    print("→ Loading TF-IDF:", IN_PARQUET_2)
    df_tfidf = pd.read_parquet(IN_PARQUET_2)
    
    assert len(df_emb) == len(df_tfidf), "DataFrames must have same number of rows"
    
    # Get embedding and tf-idf matrices
    emb_cols = [col for col in df_emb.columns if col.startswith('emb_')]
    tfidf_cols = [col for col in df_tfidf.columns if col.startswith('tfidf_')]
    
    X_emb = df_emb[emb_cols].values
    X_tfidf = df_tfidf[tfidf_cols].values
    
    # Run all 4 combinations
    results = []
    results.append(run_reduction("embedding", X_emb, "pca"))
    results.append(run_reduction("embedding", X_emb, "svd"))
    results.append(run_reduction("tfidf", X_tfidf, "pca"))
    results.append(run_reduction("tfidf", X_tfidf, "svd"))
    
    # Select best result
    best_result = max(results, key=lambda x: x[2])
    source, method, score, best_X = best_result
    print(f"\n🏆 Best result: {method.upper()} on {source} with explained variance {score:.4f}")
    
    # Add reduced features to df_emb (use it as base)
    reduced_df = pd.DataFrame(best_X, columns=[f'rdc_{i}' for i in range(best_X.shape[1])], index=df_emb.index)
    df_final = pd.concat([df_emb, reduced_df], axis=1)
    
    # Filter out redirects
    mask = ~df_final['cleaned_article_body'].str.startswith('redirect', na=False)
    df_final = df_final[mask]

    # Drop unnecessary columns
    df_final.drop(columns=emb_cols + ['cleaned_article_body', 'linked_article_titles'], inplace=True)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(OUT_PARQUET, index=False)
    print(f"✓ Wrote {len(df_final):,} rows to {OUT_PARQUET}")

if __name__ == "__main__":
    main()
