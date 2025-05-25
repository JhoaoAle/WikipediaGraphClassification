#!/usr/bin/env python
"""Stage 2 – read Wikitext rows ⇒ add destination_articles + clean_body"""

import pathlib
import pandas as pd
from utils.wikiclean import (
    extract_links,
    clean_wiki_text,
    extract_categories,
    clean_linked_articles,
    parallel_clean_wiki_text
)
from tqdm import tqdm


def main():
    # Register tqdm with pandas
    tqdm.pandas()

    IN_PARQUET = pathlib.Path("data/10_parsed/articles.parquet")
    OUT_PARQUET = pathlib.Path("data/20_transformed/articles.parquet")
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(IN_PARQUET)

    # Use progress_apply to check progress in terminal
    df['linked_article_titles'] = df['body'].progress_apply(extract_links)

    # Count how many times a "== Title ==" style section appears
    df['sections_count'] = df['body'].str.count(r'==\s*[^=]+?\s*==')

    # Applying functions
    df['categories'] = df['body'].progress_apply(extract_categories)

    # Parallel clean_wiki_text
    df["cleaned_article_body"] = parallel_clean_wiki_text(df["body"])
    
    # Drop raw body
    df.drop(['body'], inplace=True, axis=1)

    # Lowercase titles
    df['title'] = df['title'].str.lower()

    # Clean linked titles
    df['linked_article_titles'] = df['linked_article_titles'].progress_apply(clean_linked_articles)

    # Precompute valid titles only once
    valid_titles = set(df['title'])

    # Filter linked articles by valid titles
    df['linked_article_titles'] = df['linked_article_titles'].progress_apply(
        lambda titles: [t for t in titles if t in valid_titles]
    )

    df.to_parquet(OUT_PARQUET, index=False)
    print("✓ wrote", OUT_PARQUET, len(df), "rows")


if __name__ == "__main__":
    main()
