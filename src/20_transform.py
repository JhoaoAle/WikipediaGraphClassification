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
    tqdm.pandas()
    stop_words = set(stopwords.words('english'))

    IN_PARQUET = pathlib.Path("data/10_parsed/articles.parquet")
    OUT_PARQUET = pathlib.Path("data/20_transformed/articles.parquet")
    OUT_PARQUET_MAPPING = pathlib.Path("data/40_preprocessed/42_mapping/articles.parquet")
    OUT_PARQUET_GRAPH = pathlib.Path("data/40_preprocessed/42_mapping/graph_dataset.parquet")
    
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    OUT_PARQUET_MAPPING.parent.mkdir(parents=True, exist_ok=True)
    OUT_PARQUET_GRAPH.parent.mkdir(parents=True, exist_ok=True)

    # Loading parquet
    df = pd.read_parquet(IN_PARQUET)
    print(f"✓ Loaded {len(df):,} rows")
    #Lowercasing
    df['title'] = df['title'].str.lower()
    #Extracting linked articles
    df['linked_article_titles'] = df['body'].progress_apply(extract_links)
    # Cleaning linked articles titles
    df['linked_article_titles'] = df['linked_article_titles'].progress_apply(clean_linked_articles)
    #Filtering linked articles to valid titles
    valid_titles = set(df['title'])
    df['linked_article_titles'] = df['linked_article_titles'].progress_apply(
        lambda titles: [t for t in titles if t in valid_titles]
    )
    # Counting section titles
    df['sections_count'] = df['body'].progress_apply(lambda text: text.count("=="))
    # Extracting categories
    df['categories'] = df['body'].progress_apply(extract_categories)
    # Cleaning Wikitext in parallel
    df["cleaned_article_body"] = parallel_clean_wiki_text(df["body"])
    #Extracting text features from cleaned body
    feature_extractor = partial(extract_text_features, stop_words=stop_words)
    df_text_features = df['cleaned_article_body'].progress_apply(feature_extractor)
    df.drop(['body'], axis=1, inplace=True)
    df = pd.concat([df, df_text_features], axis=1)
    df["cleaned_article_body"] = df["cleaned_article_body"].progress_apply(clean_for_embedding)
    df['article_id'] = range(len(df))

    # === Network Analysis Dataset ===
    df_export = df[['article_id', 'title', 'linked_article_titles']]
    df_export.to_parquet(OUT_PARQUET_MAPPING, index=False)
    # Build title → article_id mapping
    title_to_id = dict(zip(df_export["title"], df_export["article_id"]))
    # Build edge list: for each row, generate (source_id, dest_id) pairs
    edges = []
    for source_title, linked_titles in zip(df_export["title"], df_export["linked_article_titles"]):
        source_id = title_to_id[source_title]
        for dest_title in linked_titles:
            dest_id = title_to_id.get(dest_title)
            if dest_id is not None:
                edges.append((source_id, dest_id))
    # Create edge DataFrame
    df_edges = pd.DataFrame(edges, columns=["source", "target"])
    # Save edge list
    df_edges.to_parquet(OUT_PARQUET_GRAPH, index=False)
    print(f"✓ Wrote edge list to {OUT_PARQUET_GRAPH} with {len(df_edges):,} edges")

    ## === Embeddings Generation Dataset ===
    # Save
    mask = ~df['cleaned_article_body'].str.startswith('redirect', na=False)
    df = df[mask]
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"✓ Wrote {OUT_PARQUET}, {len(df):,} rows")

if __name__ == "__main__":
    main()
