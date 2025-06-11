import pandas as pd
import numpy as np
import pathlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# tqdm setup
tqdm.pandas()

# Configuration
IN_PARQUET = pathlib.Path("data/20_transformed/articles.parquet")
OUT_PARQUET = pathlib.Path("data/31_tf_idf/articles.parquet")

def main():
    # Step 1: Load data
    df = pd.read_parquet(IN_PARQUET)
    print(f"✓ Loaded {len(df):,} rows from {IN_PARQUET}")
    
    # Step 2: TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1, 2)
    )

    print("⏳ Performing TF-IDF vectorization...")
    X_tfidf = vectorizer.fit_transform(tqdm(df['cleaned_article_body'].fillna(""), desc="TF-IDF Input"))
    print(f"✓ TF-IDF matrix shape: {X_tfidf.shape}")

    # Step 3: Dimensionality Reduction with progress
    svd = TruncatedSVD(n_components=300, random_state=42)
    print("⏳ Reducing dimensionality with TruncatedSVD...")
    X_reduced = svd.fit_transform(X_tfidf)
    X_reduced = normalize(X_reduced)
    print(f"✓ Reduced TF-IDF shape: {X_reduced.shape}")

    # Step 4: Add reduced TF-IDF features to DataFrame
    print("⏳ Constructing TF-IDF feature DataFrame...")
    tfidf_df = pd.DataFrame(
        tqdm(X_reduced, desc="Building DataFrame"),
        columns=[f"tfidf_{i}" for i in range(X_reduced.shape[1])]
    )
    df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

    # Step 5: Save result
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"✓ Saved TF-IDF features to {OUT_PARQUET}")

    print(f"Explained variance (sum): {svd.explained_variance_ratio_.sum():.3f}")

if __name__ == "__main__":
    main()
