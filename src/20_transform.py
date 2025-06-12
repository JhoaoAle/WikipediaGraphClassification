#!/usr/bin/env python
"""Stage 2 – read Wikitext rows ⇒ add destination_articles + clean_body"""

import pathlib
import pandas as pd
from utils.wikiclean import (
    extract_links,
    extract_text_features,
    extract_categories,
    clean_linked_articles,
    parallel_clean_wiki_text
)
from utils.textclean import clean_for_embedding
from tqdm import tqdm
from functools import partial
from nltk.corpus import stopwords

def main():
    # Register tqdm with pandas
    tqdm.pandas()
    stop_words = set(stopwords.words('english'))

    IN_PARQUET = pathlib.Path("data/10_parsed/articles.parquet")
    OUT_PARQUET = pathlib.Path("data/20_transformed/articles.parquet")
    OUT_PARQUET_MAPPING = pathlib.Path("data/40_preprocessed/42_mapping/articles.parquet")
    
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    OUT_PARQUET_MAPPING.parent.mkdir(parents=True, exist_ok=True)

    print("→ Loading parquet:", IN_PARQUET)
    df = pd.read_parquet(IN_PARQUET)
    print(f"✓ Loaded {len(df):,} rows")

    print("\n→ Lowercasing article titles...")
    df['title'] = df['title'].str.lower()
    
    print("\n→ Extracting linked article titles...")
    df['linked_article_titles'] = df['body'].progress_apply(extract_links)

    print("\n→ Cleaning linked article titles...")
    df['linked_article_titles'] = df['linked_article_titles'].progress_apply(clean_linked_articles)

    print("\n→ Filtering linked articles to valid titles...")
    valid_titles = set(df['title'])
    df['linked_article_titles'] = df['linked_article_titles'].progress_apply(
        lambda titles: [t for t in titles if t in valid_titles]
    )

    print("\n→ Counting section titles...")
    df['sections_count'] = df['body'].progress_apply(lambda text: text.count("=="))

    print("\n→ Extracting categories...")
    df['categories'] = df['body'].progress_apply(extract_categories)

    print("\n→ Cleaning Wikitext in parallel...")
    df["cleaned_article_body"] = parallel_clean_wiki_text(df["body"])

    print("\n→ Extracting text features from cleaned body...")
    feature_extractor = partial(extract_text_features, stop_words=stop_words)
    df_text_features = df['cleaned_article_body'].progress_apply(feature_extractor)

    print("\n→ Dropping raw body column...")
    df.drop(['body'], axis=1, inplace=True)

    print("\n→ Merging extracted text features...")
    df = pd.concat([df, df_text_features], axis=1)

    print("\n→ Applying semantic cleaning for embeddings...")
    df["cleaned_article_body"] = df["cleaned_article_body"].progress_apply(clean_for_embedding)
    

    # Add article_id column for future merging
    print("\n→ Adding unique identifier for articles...")
    
    df['article_id'] = range(len(df))
    # === Network Analysis Dataset ===
    print("\n→ Generating dataset for network analysis...")
    
    df_export = df[['article_id', 'title', 'linked_article_titles']]
    df_export.to_parquet(OUT_PARQUET_MAPPING, index=False)
    print("✓ wrote", OUT_PARQUET_MAPPING, len(df_export), "rows")
    
    ## === Embeddings Generation Dataset ===
    # Save
    print("\n→ Generating fitlered dataset for clustering analysis...")
    
    mask = ~df['cleaned_article_body'].str.startswith('redirect', na=False)
    df = df[mask]
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"✓ Wrote {OUT_PARQUET}, {len(df):,} rows")

if __name__ == "__main__":
    main()
